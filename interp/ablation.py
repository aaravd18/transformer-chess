import torch
from contextlib import contextmanager
from evaluation.play import show_suggestions, get_legal_move_probs
import chess

@torch.no_grad()
def compute_layer_mean_per_head(
    model: torch.nn.Module,
    layer_idx: int,
    calibration_loader,
    device: torch.device,
    n_positions: int,
) -> torch.Tensor:
    """Mean of layer `layer_idx`'s per-head output, averaged over board
    positions but NOT over sequence position.

    Stops once at least `n_positions` examples have been seen.

    Returns:
        (n_heads, L, head_dim) tensor on `device`.
    """
    model.eval()
    attn = model.blocks[layer_idx].attn

    prev_flag = attn._save_per_head
    attn._save_per_head = True

    running_sum = None
    n_seen = 0

    try:
        for batch in calibration_loader:
            tokens = batch["tokens"].to(device)         # (B, 68) long
            _ = model(tokens)
            per_head = attn.last_per_head_out            # (B, H, L, head_dim)
            B = per_head.shape[0]

            batch_sum = per_head.sum(dim=0)              # (H, L, head_dim)
            running_sum = batch_sum if running_sum is None else running_sum + batch_sum
            n_seen += B

            if n_seen >= n_positions:
                break
    finally:
        attn._save_per_head = prev_flag
        attn.last_per_head_out = None

    if n_seen == 0:
        raise ValueError("calibration_loader yielded no batches")

    print(f"  computed mean over {n_seen} positions")
    return running_sum / n_seen


@contextmanager
def patched_heads(model, ablation_spec, layer_means):
    """Install mean-ablation patches for the duration of the with-block.

    ablation_spec: {layer_idx: [head indices]}.
    layer_means:   {layer_idx: (n_heads, L, head_dim) tensor}.
                   Only needs entries for layers that appear in
                   ablation_spec.
    """
    patched_layers = []
    try:
        for layer_idx, head_indices in ablation_spec.items():
            if not head_indices:
                continue
            attn = model.blocks[layer_idx].attn
            mean = layer_means[layer_idx]
            heads = torch.tensor(head_indices, dtype=torch.long, device=mean.device)
            attn._patch_per_head = {
                "heads": heads,
                "values": mean[heads],   # (P, L, head_dim)
            }
            patched_layers.append(layer_idx)
        yield
    finally:
        for layer_idx in patched_layers:
            model.blocks[layer_idx].attn._patch_per_head = None


@torch.no_grad()
def show_suggestions_under_ablation(
    model,
    board: chess.Board,
    ablation_spec: dict,
    layer_means: dict,
    device: torch.device,
    k=5
) -> tuple[float, chess.Move, float]:
    """Returns (prob_of_target_move, top_move, top_move_prob)."""
    model.eval()
    with patched_heads(model, ablation_spec, layer_means):
        show_suggestions(model, board, device, k=k)


@torch.no_grad()
def prob_of_move_under_ablation(
    model,
    board: chess.Board,
    target_move: chess.Move,
    ablation_spec: dict,
    layer_means: dict,
    device: torch.device,
) -> tuple[float, chess.Move, float]:
    """Returns (prob_of_target_move, top_move, top_move_prob)."""
    model.eval()
    with patched_heads(model, ablation_spec, layer_means):
        results = get_legal_move_probs(model, board, device)

    target_prob = 0.0
    for mv, p in results:
        if mv == target_move:
            target_prob = p
            break

    top_move, top_prob = results[0]
    return target_prob, top_move, top_prob


@torch.no_grad()
def run_sanity_checks(model, layer_means, device):
    """Verify the patching infrastructure before trusting real results."""
    print("\n--- Sanity checks ---")

    # Use a single fixed position.
    board = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(mv)
    tokens = torch.from_numpy(tokenize(board)).long().unsqueeze(0).to(device)

    out_unpatched = model(tokens)["move_logits"]

    # 1. Empty spec is a no-op.
    with patched_heads(model, {}, layer_means):
        out_empty = model(tokens)["move_logits"]
    assert torch.allclose(out_unpatched, out_empty), "empty spec changed output"
    print("  [OK] empty ablation_spec is a no-op")

    # 2. Empty head list is a no-op.
    with patched_heads(model, {0: []}, layer_means):
        out_empty_list = model(tokens)["move_logits"]
    assert torch.allclose(out_unpatched, out_empty_list), "empty head list changed output"
    print("  [OK] empty head list is a no-op")

    # 3. Ablating all 8 heads of layer 0 substantially changes the output.
    with patched_heads(model, {0: list(range(8))}, layer_means):
        out_full = model(tokens)["move_logits"]
    diff = (out_unpatched - out_full).abs().max().item()
    assert diff > 0.1, f"ablating all of layer 0 barely changed logits (max diff {diff:.4f})"
    print(f"  [OK] ablating all layer-0 heads changes logits substantially (max diff {diff:.2f})")
