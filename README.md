# Chess Transformer — Mechanistic Interpretability

A small transformer trained to predict human chess moves, built from the ground up to be taken apart.

The goal is not playing strength. The goal is a model whose internals can be read: attention maps that correspond directly to square-to-square communication, residual streams that localize move information by design, and tooling that makes probing and patching straightforward.

---

## Architecture

**6.5M parameters** — 8 layers, 8 heads, d_model 256, d_ff 1024, pre-norm, bidirectional attention.

Three design choices made deliberately for interpretability:

**Spatial 68-token board representation.** Tokens 0–63 are the 64 board squares in python-chess order (a1=0, …, h8=63). Tokens 64–66 are metadata (castling rights, en passant target, halfmove clock bucket). Token 67 is a `[CLS]` register kept as a pooling slot for position-level probes. Each square gets its own token so attention maps read directly as square-to-square attention weights — no decoding required.

**Color canonicalization.** Every position the model sees is "white to move." Black-to-move positions are rank-mirrored and piece colors swapped before tokenization. The move target and value label are mirrored in lockstep. The model never sees a side identity; concepts learned are side-invariant.

**Bilinear policy head.** Move logits are computed as:

```
S = source_mlp(h_squares)   # (B, 64, D)
T = target_mlp(h_squares)   # (B, 64, D)
logit[i, j] = <S[i], T[j]>
```

`logit[i, j]` depends *only* on the final residual-stream activations at squares `i` and `j`. If the model has learned to represent a future move, information about it has nowhere to live except the source or target square's residual stream. A small promotion head is layered on top for pawn promotions.

The attention module keeps Q/K/V/O as separate `nn.Linear` layers (not fused), stores attention probabilities on request via a flag, and supports mean-ablation patching at the per-head level. All of these are off by default so training has zero overhead.

---

## Dataset

Lichess games filtered to both players ≥ 1800 Elo and ≥ 3-minute time control. The download script streams and filters the Lichess database in one pass; the current committed dataset has **1,000 games / ~73k positions** (see `datasets/meta.json`). Rerun `data/download_data.py` with a higher `TARGET_GAMES` to scale up.

Train/val split is at the game level (not position level) to prevent positions from the same game leaking across splits.

---

## Repo structure

```
model.py                  Model definition: ChessEmbedding, TransformerBlock,
                          PolicyHead, ChessTransformer, policy_loss, load_model

data/
  tokenizer.py            tokenize(board) and iter_positions(game) — the
                          canonical-frame conversion lives here
  preprocess.py           Two-pass PGN → memmap .npy pipeline (binarize_pgn)
  download_data.py        Stream-filter Lichess PGN by Elo + time control
  dataloader.py           make_loader() for training

train.ipynb               Training loop, checkpointing, W&B logging

interp/
  attention.py            Attention visualization: show_square_attention,
                          plot_single_head_attention, show_multi_square_attention
  ablation.py             Mean-ablation patching: compute_layer_mean_per_head,
                          patched_heads context manager, prob_of_move_under_ablation

evaluation/
  play.py                 get_legal_move_probs, suggest_moves, play_game (interactive)
  evals.py                Full eval suite
  evaluate.py             Entry point: load checkpoint + run evals
  stockfish.py            Stockfish agreement metrics

checkpoints/              Saved .pt files (not tracked by git if large)
datasets/                 Preprocessed .npy arrays + meta.json
```

---

## Workflow

### Training

```bash
pip install -r requirements.txt
```

1. Optionally re-download data: `python data/download_data.py` (writes `filtered_games.pgn`)
2. Optionally re-preprocess: `python data/preprocess.py` (writes `datasets/`)
3. Open and run `train.ipynb` — produces a checkpoint in `checkpoints/`

The prebuilt dataset is already committed; skip steps 1–2 unless you want more data.

### Interpretability work

The intended workflow for exploratory analysis is a clean Colab notebook that imports this repo as a library:

```python
# In a fresh Colab notebook:
!git clone <repo-url>
%cd transformer-chess

from model import load_model
from interp.attention import show_square_attention, plot_single_head_attention
from interp.ablation import compute_layer_mean_per_head, patched_heads, prob_of_move_under_ablation
from evaluation.play import suggest_moves, get_legal_move_probs
import chess, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/5M-longer-run.pt", device)
```

**Attention maps** — `show_square_attention` runs a forward pass with attention hooks enabled and plots all heads across all layers for a given square. `plot_single_head_attention` produces a single polished figure suitable for publication. Both handle the canonical-frame remapping transparently: pass a normal `chess.Board` (either color to move) and the display maps back to the original orientation.

**Activation patching** — `compute_layer_mean_per_head` computes the mean per-head output over a calibration set. `patched_heads` is a context manager that replaces specified heads with their means for the duration of a forward pass. `prob_of_move_under_ablation` measures how a target move's probability changes under a given ablation spec.

**Move predictions** — `suggest_moves(model, board, device, k=5)` returns top-k legal moves with probabilities. `play_game` is an interactive REPL with an SVG board that runs in Jupyter.

The repo is intentionally kept as a library. Exploratory interpretability work should happen in notebooks, not by editing files here.

---

## Notes 
- There is no value head. The `[CLS]` token is kept in the sequence as a pooling slot for future position-level probes but is not currently supervised.
