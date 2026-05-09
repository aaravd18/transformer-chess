"""Mini chess transformer with a Leela-style policy head.

This is an interpretability-research model. The architecture intentionally
mirrors the structure of Leela Chess Zero's policy network so that the same
analyses (activation patching on squares, bilinear probes on the residual
stream, etc.) translate over. Where Leela uses domain-specific tricks
(smolgen, post-norm, custom positional encodings) we use the standard
choices that train stably at small scale.

Key structural facts to keep in mind when interpreting activations:

- Sequence length is 68. Tokens 0..63 are the 64 chessboard squares in
  python-chess order (a1=0, b1=1, ..., h8=63). Tokens 64..66 are the
  castling/en-passant/halfmove-clock board metadata. Token 67 is a [CLS]
  register pooled by the value head.

- The board is canonicalized to "white to move" upstream (see data.py).
  The model never sees a black-to-move position.

- Attention is fully bidirectional. Every token attends to every other.

- No value head. The Leela paper's main results (Section 3) are all
  computed against the policy output, and Appendix C confirms value-head
  results mirror them because the heads share the transformer body. We
  keep the [CLS] token in the input anyway as a cheap pooling slot for
  future position-level probes.

- The policy head computes a 64x64 logit matrix. logit[i, j] is the logit
  for the move (from=i, to=j). Critically, that logit depends ONLY on the
  final residual-stream activations at squares i and j -- the same
  property that lets the Leela paper localize move information to
  specific squares.

Shapes throughout:
    B = batch size
    L = sequence length = 68
    D = model width
    S = 64 (number of board squares)
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.tokenizer import *

# Config
@dataclass
class ModelConfig:
    """Hyperparameters for the chess transformer"""
    d_model:     int   = 256
    n_heads:     int   = 8
    n_layers:    int   = 8
    d_ff:        int   = 1024
    dropout:     float = 0.1
    n_promotions: int  = 5      # don't change this


class ChessEmbedding(nn.Module):
    """Token + positional embedding.

    Different sequence positions carry different "kinds" of information
    (piece IDs vs. castling-rights bitmask vs. en-passant square index
    vs. halfmove bucket vs. CLS), so each gets its own embedding table.
    A learned positional embedding is then added so the model can tell
    a1 from b1 etc.

    The vocabularies don't overlap semantically -- token id 3 means
    something completely different at the piece slots vs. the castling
    slot -- so giving each its own table is both cleaner and more
    parameter-efficient than a single (max_vocab x d_model) table.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model

        self.piece_emb     = nn.Embedding(VOCAB_PIECE,    D)
        self.castling_emb  = nn.Embedding(VOCAB_CASTLING, D)
        self.ep_emb        = nn.Embedding(VOCAB_EP,       D)
        self.halfmove_emb  = nn.Embedding(VOCAB_HALFMOVE, D)
        self.cls_emb       = nn.Embedding(VOCAB_CLS,      D)

        self.pos_emb = nn.Embedding(SEQ_LEN, D)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 68) long. Returns (B, 68, D)."""
        B, L = tokens.shape
        assert L == SEQ_LEN, f"expected seq len {SEQ_LEN}, got {L}"

        out = torch.empty(B, L, self.piece_emb.embedding_dim,
                          device=tokens.device, dtype=self.piece_emb.weight.dtype)

        # Slice the sequence by field and run each through its own table.
        out[:, PIECE_TOKENS]   = self.piece_emb(tokens[:, PIECE_TOKENS])
        out[:, CASTLING_TOKEN] = self.castling_emb(tokens[:, CASTLING_TOKEN])
        out[:, EP_TOKEN]       = self.ep_emb(tokens[:, EP_TOKEN])
        out[:, HALFMOVE_TOKEN] = self.halfmove_emb(tokens[:, HALFMOVE_TOKEN])
        out[:, CLS_TOKEN]      = self.cls_emb(tokens[:, CLS_TOKEN])

        # Add positional embedding (broadcast over batch).
        positions = torch.arange(L, device=tokens.device)
        out = out + self.pos_emb(positions)
        return out

class MultiHeadAttention(nn.Module):
    """Bidirectional multi-head self-attention, hand-rolled.

    Why not nn.MultiheadAttention: that module fuses Q, K, V into one
    linear with concatenated weights, which makes per-head ablation and
    bilinear-probe analysis annoying. Here Q/K/V/O are separate Linears
    and we keep heads as an explicit dimension so you can hook anything.

    Shapes:
        x:           (B, L, D)
        q, k, v:     (B, n_heads, L, head_dim)
        attn_logits: (B, n_heads, L, L)
        attn_probs:  (B, n_heads, L, L)
        out:         (B, L, D)

    Hooks (set externally before forward):
        self._save_attn_probs = True  -> stores last (B, H, L, L) on self.last_attn_probs
        self._save_per_head   = True  -> stores per-head output (B, H, L, head_dim)
                                         BEFORE o_proj on self.last_per_head_out
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model
        self.scale    = self.head_dim ** -0.5

        # Separate Linears -- this is the whole point of doing it manually.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)

        # Hook flags & storage. Default off so training has zero overhead.
        self._save_attn_probs = False
        self._save_per_head   = False
        self.last_attn_probs  = None
        self.last_per_head_out = None

        # Patching. When set, _patch_per_head should be a dict:
        #   {
        #       "heads": LongTensor of head indices to patch,
        #       "values": Tensor of shape (n_patched, L, head_dim) -- the
        #                 replacement output for those heads, broadcast
        #                 over batch.
        #   }
        # When None (default), forward is unchanged.
        self._patch_per_head = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, head_dim)."""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, head_dim) -> (B, L, D)."""
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project then split into heads.
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Scaled dot-product. (B,H,L,Dh) @ (B,H,Dh,L) -> (B,H,L,L)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs  = F.softmax(attn_logits, dim=-1)
        attn_probs  = self.dropout(attn_probs)

        if self._save_attn_probs:
            # Detach so storing doesn't keep the autograd graph alive.
            self.last_attn_probs = attn_probs.detach()

        # Weighted values. (B,H,L,L) @ (B,H,L,Dh) -> (B,H,L,Dh)
        per_head_out = torch.matmul(attn_probs, v)

        # Mean-ablation patch 
        # If a patch is registered, overwrite the targeted heads with the
        # supplied per-token replacement. Broadcasts over the batch dim.
        if self._patch_per_head is not None:
            heads = self._patch_per_head["heads"]
            values = self._patch_per_head["values"]   # (P, L, Dh)
            B = per_head_out.shape[0]
            per_head_out = per_head_out.clone()  # avoid in-place on autograd tensor
            per_head_out[:, heads] = values.unsqueeze(0).expand(B, -1, -1, -1)

        if self._save_per_head:
            self.last_per_head_out = per_head_out.detach()

        return self.o_proj(self._merge_heads(per_head_out))



# Transformer block (pre-norm, bidirectional attention)
class TransformerBlock(nn.Module):
    """Standard pre-norm block: x -> x + Attn(LN(x)); x -> x + MLP(LN(x)).

    Pre-norm trains more stably at small scale than the post-norm variant
    Leela uses; we keep it simple and standard.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D). No mask -- attention is bidirectional."""
        h = self.ln1(x)
        attn_out = self.attn(h)
        x = x + self.dropout(attn_out)

        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x


# ---------------------------------------------------------------------------
# Policy head: source MLP, target MLP, dot product
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Leela-style policy head.

    Two 2-layer MLPs share their first linear layer (parameter efficiency
    plus a useful symmetry: both heads start from the same "what is this
    square about" representation, then diverge). Each MLP is applied
    independently to every square's final embedding -- there is no
    cross-square mixing in the head; all of that has already happened in
    the transformer body.

        S = source_mlp(h_squares)   # (B, 64, D)
        T = target_mlp(h_squares)   # (B, 64, D)
        logits[i, j] = <S[i], T[j]>

    The interpretability consequence: logit(i->j) is a function of ONLY
    h_i and h_j. If the model has learned to represent a future move,
    information about it has nowhere to live except on its source or
    target square's residual stream.

    Promotion: the (from, to) grid alone can't distinguish promoting to
    a knight vs a queen. We add a small promotion head conditioned on
    the source square's final embedding. At training time we supervise
    promotion only on actual promotion moves (caller's responsibility).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model

        # Shared first layer. Per-square (no mixing).
        self.shared = nn.Linear(D, D)
        self.act = nn.GELU()

        # Separate second layers.
        self.source_proj = nn.Linear(D, D)
        self.target_proj = nn.Linear(D, D)

        # Promotion: 5-way classifier per source square.
        # 0 = no promotion (the usual case), 1..4 = N/B/R/Q.
        self.promo_head = nn.Linear(D, cfg.n_promotions)

    def source_target(self, h_squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the (S, T) intermediate representations.

        Exposed separately so interpretability code can grab them without
        re-running the dot product.

            h_squares: (B, 64, D)
        Returns:
            S: (B, 64, D)
            T: (B, 64, D)
        """
        h = self.act(self.shared(h_squares))
        return self.source_proj(h), self.target_proj(h)

    def forward(self, h_squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_squares: (B, 64, D).

        Returns:
            move_logits: (B, 64, 64)  -- logits[i, j] = move from i to j
            promo_logits: (B, 64, n_promotions) -- conditioned on source sq
        """
        S, T = self.source_target(h_squares)
        # (B, 64, D) @ (B, D, 64) -> (B, 64, 64)
        move_logits = torch.bmm(S, T.transpose(1, 2))
        promo_logits = self.promo_head(S)
        return move_logits, promo_logits


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ChessTransformer(nn.Module):
    """Full model: embed -> transformer body -> policy head.

    Forward returns a dict so callers can pick what they need without
    positional confusion.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = ChessEmbedding(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.policy = PolicyHead(cfg)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        return_residuals: bool = False,
    ) -> dict:
        """tokens: (B, 68) long.

        Args:
            return_residuals: if True, also return the post-block residual
                stream activations. Useful for probes and patching.

        Returns dict with keys:
            move_logits  : (B, 64, 64)
            promo_logits : (B, 64, n_promotions)
            residuals    : list of (B, 68, D), one per layer (if requested)
        """
        x = self.embed(tokens)

        residuals = [] if return_residuals else None
        for block in self.blocks:
            x = block(x)
            if return_residuals:
                residuals.append(x)

        x = self.ln_f(x)

        # Slice out the 64 board-square embeddings for the policy head.
        # The other 4 tokens (castling/ep/halfmove/cls) influenced the
        # squares via attention but aren't read directly here.
        h_squares = x[:, PIECE_TOKENS]   # (B, 64, D)

        move_logits, promo_logits = self.policy(h_squares)

        out = {
            "move_logits":  move_logits,
            "promo_logits": promo_logits,
        }
        if return_residuals:
            out["residuals"] = residuals
        return out

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss helpers -- callers can use these directly or roll their own.
# ---------------------------------------------------------------------------

def policy_loss(
    move_logits:  torch.Tensor,   # (B, 64, 64)
    promo_logits: torch.Tensor,   # (B, 64, n_promo)
    from_sq:      torch.Tensor,   # (B,) long
    to_sq:        torch.Tensor,   # (B,) long
    promotion:    torch.Tensor,   # (B,) long, 0 if not a promotion
) -> torch.Tensor:
    """Cross-entropy over the 4096 (from, to) pairs, plus a promotion term
    that only fires on actual promotion moves.

    No legality mask is applied -- on supervised data the target move is
    always legal, and unmasked CE is both faster and a slightly stronger
    training signal (the model has to learn that, e.g., a rook can't
    teleport diagonally).
    """
    B = move_logits.shape[0]

    # Flatten the 64x64 grid to a single 4096-class CE.
    flat_logits = move_logits.reshape(B, 64 * 64)
    flat_target = from_sq * 64 + to_sq
    move_ce = F.cross_entropy(flat_logits, flat_target)

    # Promotion: gather the promo logits at each example's source square,
    # then CE against the promotion id. This trains promo_head[from] for
    # both promotion and non-promotion moves (target = 0 for the latter),
    # which is fine -- the model learns "this source square does not
    # promote" as the default.
    src_promo = promo_logits[torch.arange(B, device=promo_logits.device), from_sq]
    promo_ce  = F.cross_entropy(src_promo, promotion)

    return move_ce + 0.1 * promo_ce


def load_model(ckpt_path: str, device: torch.device) -> ChessTransformer:
    """Load a checkpoint into a fresh, uncompiled ChessTransformer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt["cfg"]
    model = ChessTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded {ckpt_path}")
    print(f"  trained for {ckpt.get('step', '?')} steps")
    if "val_loss" in ckpt:
        print(f"  saved val_loss: {ckpt['val_loss']:.4f}  "
              f"val_acc: {ckpt.get('val_acc', 0):.1%}")
    print(f"  model: {model.num_params():,} params")
    return model


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig()
    model = ChessTransformer(cfg)
    print(f"params: {model.num_params():,}")

    B = 4
    fake_tokens = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    # Put a couple of pieces on the board so the embedding does something.
    fake_tokens[:, 0]  = 4   # white rook on a1
    fake_tokens[:, 4]  = 6   # white king on e1
    fake_tokens[:, 60] = 12  # black king on e8

    out = model(fake_tokens, return_residuals=True)
    print("move_logits :", out["move_logits"].shape)
    print("promo_logits:", out["promo_logits"].shape)
    print("residuals   :", len(out["residuals"]), "x", out["residuals"][0].shape)

    # Fake a training step.
    from_sq   = torch.tensor([0, 0, 0, 0])
    to_sq     = torch.tensor([1, 2, 3, 4])
    promotion = torch.tensor([0, 0, 0, 0])

    loss = policy_loss(out["move_logits"], out["promo_logits"],
                       from_sq, to_sq, promotion)
    loss.backward()
    print(f"loss: {loss.item():.4f}  (backward ran cleanly)")