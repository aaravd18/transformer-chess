"""Chess-specific evaluation for the policy model.

Three tiers, ordered by what to look at first:

  Tier 1 -- structural sanity (does the model respect the rules?)
    - legal_move_rate

  Tier 2 -- chess sense (does it pick good moves?)
    - top_k_accuracy
    - stockfish_agreement (optional; requires a Stockfish binary)

  Tier 3 -- emergent skill (specific concepts)
    - mate_in_one_recall

Everything works on the same canonical white-to-move frame the model was
trained on.

Default scale is ~20k positions. Per-metric subsampling lets us spend
compute where it matters: mate-in-1 sees the full 20k (rare event,
needs the sample), legality sees ~5k (cheap, high-rate, plateaus fast),
top-k sees ~20k (the headline number), Stockfish sees ~500 (slow).

Usage:
    results = run_full_eval(model, val_loader, device, n_predict=20000)
    print_eval_results(results)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import chess
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reconstructing a chess.Board from our token tensor
# ---------------------------------------------------------------------------
#
# The model only ever sees canonicalized (white-to-move) positions. To
# check legality we need a real chess.Board for that canonical position.
# We reconstruct it from the tokens directly.
#
# This is the inverse of data.tokenize() for the parts we care about
# (pieces, castling, en passant). We don't reconstruct the halfmove
# clock or move number because they don't affect move legality outside
# of the 50/75-move rules, which essentially never matter at our depth.

# Token indices (must match data.tokenize()).
CASTLING_TOKEN = 64
EP_TOKEN = 65

# ID -> (piece_type, color). Inverse of data._PIECE_TO_ID.
_ID_TO_PIECE = {0: None}
for color, base in [(chess.WHITE, 1), (chess.BLACK, 7)]:
    for offset, ptype in enumerate(
        [chess.PAWN, chess.KNIGHT, chess.BISHOP,
         chess.ROOK, chess.QUEEN, chess.KING]
    ):
        _ID_TO_PIECE[base + offset] = (ptype, color)


def board_from_tokens(tokens: np.ndarray) -> chess.Board:
    """Reconstruct a chess.Board (white to move) from a (68,) token array."""
    board = chess.Board.empty()
    board.turn = chess.WHITE

    for sq in range(64):
        pid = int(tokens[sq])
        if pid == 0:
            continue
        ptype, color = _ID_TO_PIECE[pid]
        board.set_piece_at(sq, chess.Piece(ptype, color))

    cr = int(tokens[CASTLING_TOKEN])
    rights = ""
    if cr & 1: rights += "K"
    if cr & 2: rights += "Q"
    if cr & 4: rights += "k"
    if cr & 8: rights += "q"
    board.set_castling_fen(rights if rights else "-")

    ep = int(tokens[EP_TOKEN])
    board.ep_square = ep if ep < 64 else None

    return board


# ---------------------------------------------------------------------------
# Decoding model output into chess.Move objects
# ---------------------------------------------------------------------------

def decode_move(from_sq: int, to_sq: int, board: chess.Board,
                promo_id: int = 0) -> chess.Move:
    """Build a chess.Move from (from, to) plus an optional promotion id.

    Defaults to queen promotion if a pawn reaches rank 8 with no
    explicit promotion choice -- otherwise we'd flag legal promotions
    as illegal.
    """
    promo_map = {1: chess.KNIGHT, 2: chess.BISHOP,
                 3: chess.ROOK, 4: chess.QUEEN}
    promotion = promo_map.get(promo_id)

    if promotion is None:
        piece = board.piece_at(from_sq)
        if (piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_sq) == 7):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


# ---------------------------------------------------------------------------
# Helper: collect predictions and metadata over a set of positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_predictions(model, val_loader, device,
                         n_positions: int = 20000,
                         top_k: int = 5) -> dict:
    """Run the model on val_loader, return per-position predictions + ground truth.

    Returns a dict of numpy arrays (all length N <= n_positions):
        tokens     : (N, 68) uint8        -- raw inputs, for board reconstruction
        true_from  : (N,) int             -- played from-square
        true_to    : (N,) int             -- played to-square
        true_promo : (N,) int             -- played promotion id
        pred_from  : (N,) int             -- top-1 from-square
        pred_to    : (N,) int             -- top-1 to-square
        pred_promo : (N,) int             -- top-1 promotion id
        topk_flat  : (N, top_k) int       -- top-k flat indices (from*64+to)
    """
    model.eval()

    bufs = defaultdict(list)
    seen = 0

    for batch in val_loader:
        if seen >= n_positions:
            break

        tokens = batch["tokens"].to(device, non_blocking=True)
        out = model(tokens)
        move_logits  = out["move_logits"]                  # (B, 64, 64)
        promo_logits = out["promo_logits"]                 # (B, 64, n_promo)

        B = move_logits.shape[0]
        flat = move_logits.reshape(B, 64 * 64)             # (B, 4096)

        # Top-1 (from, to).
        top1 = flat.argmax(dim=-1)                         # (B,)
        pred_from = (top1 // 64).cpu().numpy()
        pred_to   = (top1 %  64).cpu().numpy()

        # Top-k flat indices.
        topk = flat.topk(top_k, dim=-1).indices.cpu().numpy()

        # Promotion: read the promo head at the predicted from-square.
        idx = torch.arange(B, device=device)
        pred_promo = promo_logits[idx, top1 // 64].argmax(dim=-1).cpu().numpy()

        bufs["tokens"].append(batch["tokens"].numpy())
        bufs["true_from"].append(batch["from_sq"].numpy())
        bufs["true_to"].append(batch["to_sq"].numpy())
        bufs["true_promo"].append(batch["promotion"].numpy())
        bufs["pred_from"].append(pred_from)
        bufs["pred_to"].append(pred_to)
        bufs["pred_promo"].append(pred_promo)
        bufs["topk_flat"].append(topk)

        seen += B

    return {k: np.concatenate(v)[:n_positions] for k, v in bufs.items()}


# ---------------------------------------------------------------------------
# Subsampling and board caching
# ---------------------------------------------------------------------------

def _subsample(preds: dict, n: int, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Return a sub-dict of preds with n rows, deterministically sampled.

    Also returns the indices used, so callers can index into a shared
    board cache without rebuilding boards.
    """
    total = len(preds["true_from"])
    if n >= total:
        idx = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=n, replace=False)
    return {k: v[idx] for k, v in preds.items()}, idx


def _build_board_cache(preds: dict) -> list[chess.Board]:
    """Reconstruct every board once. Shared across metrics that need boards."""
    return [board_from_tokens(preds["tokens"][i])
            for i in range(len(preds["true_from"]))]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% CI for the mean of a binary/numeric array via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot_means[b] = sample.mean()
    return (float(np.quantile(boot_means, alpha / 2)),
            float(np.quantile(boot_means, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Tier 1: legality
# ---------------------------------------------------------------------------

def legal_move_rate(preds: dict, boards: list[chess.Board]) -> dict:
    """Fraction of top-1 predictions that are legal in their position.

    Per-piece is computed over positions where the model's *predicted*
    from-square holds that piece. Positions where the predicted from-sq
    is empty count as illegal but aren't binned into any piece.
    """
    n = len(preds["true_from"])
    legal_flags = np.zeros(n, dtype=bool)
    legal_by_piece = defaultdict(lambda: [0, 0])  # symbol -> [legal, total]

    for i in range(n):
        board = boards[i]
        move = decode_move(int(preds["pred_from"][i]),
                           int(preds["pred_to"][i]),
                           board,
                           int(preds["pred_promo"][i]))
        is_legal = move in board.legal_moves
        legal_flags[i] = is_legal

        piece = board.piece_at(int(preds["pred_from"][i]))
        if piece is not None:
            sym = piece.symbol().upper()
            legal_by_piece[sym][1] += 1
            legal_by_piece[sym][0] += int(is_legal)

    by_piece = {
        sym: legal / total if total > 0 else None
        for sym, (legal, total) in legal_by_piece.items()
    }
    by_piece_n = {sym: total for sym, (_, total) in legal_by_piece.items()}

    ci_lo, ci_hi = _bootstrap_ci(legal_flags.astype(float))
    return {
        "overall":     float(legal_flags.mean()),
        "ci95":        (ci_lo, ci_hi),
        "by_piece":    by_piece,
        "by_piece_n":  by_piece_n,
        "n_positions": n,
    }


# ---------------------------------------------------------------------------
# Tier 2: chess sense
# ---------------------------------------------------------------------------

def top_k_accuracy(preds: dict, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Top-k accuracy against the played move.

    Hit at k iff the played (from, to) pair appears in the top-k logits.
    """
    target = preds["true_from"] * 64 + preds["true_to"]    # (N,)
    topk   = preds["topk_flat"]                            # (N, max_k)

    out = {"n_positions": len(target)}
    for k in ks:
        if k > topk.shape[1]:
            raise ValueError(f"top-{k} requested but only {topk.shape[1]} stored")
        hits = (topk[:, :k] == target[:, None]).any(axis=1).astype(float)
        ci_lo, ci_hi = _bootstrap_ci(hits)
        out[f"top{k}"] = float(hits.mean())
        out[f"top{k}_ci95"] = (ci_lo, ci_hi)
    return out


def stockfish_agreement(preds: dict, boards: list[chess.Board],
                        stockfish_path: str,
                        depth: int = 10,
                        centipawn_threshold: int = 50) -> Optional[dict]:
    """For each position, check whether the model's top-1 move is within
    `centipawn_threshold` of optimal at depth `depth`.

    Slow -- expects a small subsample (a few hundred positions).
    Returns None if Stockfish can't be launched.
    """
    try:
        import chess.engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"[stockfish_agreement] skipping: {e}")
        return None

    n = len(preds["true_from"])
    n_evaluated = 0
    n_within_threshold = 0
    cp_losses = []
    n_illegal_skipped = 0

    try:
        for i in range(n):
            board = boards[i]
            pred_move = decode_move(int(preds["pred_from"][i]),
                                    int(preds["pred_to"][i]),
                                    board,
                                    int(preds["pred_promo"][i]))
            if pred_move not in board.legal_moves:
                # Tracked separately by legality metric; skip here.
                n_illegal_skipped += 1
                continue

            info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_before = info_before["score"].white().score(mate_score=10000)

            board.push(pred_move)
            info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
            score_after = info_after["score"].white().score(mate_score=10000)
            board.pop()

            # Approx cp loss (white's POV, white to move in canonical frame).
            cp_loss = score_before - score_after
            cp_losses.append(cp_loss)
            if cp_loss <= centipawn_threshold:
                n_within_threshold += 1
            n_evaluated += 1
    finally:
        engine.quit()

    if n_evaluated == 0:
        return {"sampled": 0, "agreement_rate": None,
                "n_illegal_skipped": n_illegal_skipped}

    return {
        "sampled":            n_evaluated,
        "n_illegal_skipped":  n_illegal_skipped,
        "agreement_rate":     n_within_threshold / n_evaluated,
        "median_cp_loss":     float(np.median(cp_losses)),
        "mean_cp_loss":       float(np.mean(cp_losses)),
    }


# ---------------------------------------------------------------------------
# Tier 3: emergent skill
# ---------------------------------------------------------------------------

def mate_in_one_recall(preds: dict, boards: list[chess.Board]) -> dict:
    """Of positions where a mate-in-1 exists, did the model predict it?

    Most positions don't have a mate-in-1 -- this is why the metric
    runs on the full prediction set, not a subsample.
    """
    n = len(preds["true_from"])
    n_with_mate = 0
    n_predicted_mate = 0
    hit_flags = []  # 1/0 per mate-in-1 position, for CI

    for i in range(n):
        board = boards[i]

        mate_moves = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                mate_moves.append(move)
            board.pop()

        if not mate_moves:
            continue
        n_with_mate += 1

        pred_move = decode_move(int(preds["pred_from"][i]),
                                int(preds["pred_to"][i]),
                                board,
                                int(preds["pred_promo"][i]))
        hit = pred_move in mate_moves
        hit_flags.append(int(hit))
        if hit:
            n_predicted_mate += 1

    if n_with_mate == 0:
        return {"n_positions_with_mate": 0, "n_found": 0,
                "recall": None, "ci95": None}

    ci_lo, ci_hi = _bootstrap_ci(np.array(hit_flags, dtype=float))
    return {
        "n_positions_with_mate": n_with_mate,
        "n_found":               n_predicted_mate,
        "recall":                n_predicted_mate / n_with_mate,
        "ci95":                  (ci_lo, ci_hi),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

# Default per-metric sample sizes. Mate-in-1 gets the full prediction
# set because mating positions are rare. Legality plateaus fast so a
# smaller sample is fine. Top-k is the headline so we use the full set.
DEFAULT_METRIC_SIZES = {
    "legality":     5_000,
    "topk":         20_000,
    "mate_in_one":  20_000,   # rare event; needs the full sample
    "stockfish":    500,      # slow; just need a directional signal
}


def run_full_eval(model, val_loader, device,
                  n_predict: int = 20_000,
                  metric_sizes: Optional[dict] = None,
                  stockfish_path: Optional[str] = None) -> dict:
    """Run all evals and return a single dict of results.

    Args:
        n_predict: how many positions to run the model on (the expensive step).
        metric_sizes: optional override of per-metric subsample sizes.
        stockfish_path: path to a Stockfish binary; skips that metric if None.

    Each metric gets its own subsample of size `min(metric_sizes[name], n_predict)`,
    drawn deterministically (seed=0). Boards are reconstructed once for
    the full prediction set and shared across metrics.
    """
    sizes = {**DEFAULT_METRIC_SIZES, **(metric_sizes or {})}
    sizes = {k: min(v, n_predict) for k, v in sizes.items()}

    print(f"Collecting predictions on up to {n_predict} positions...")
    preds = _collect_predictions(model, val_loader, device,
                                 n_positions=n_predict, top_k=5)
    actual_n = len(preds["true_from"])
    print(f"  got {actual_n} positions")

    print(f"Reconstructing {actual_n} boards (shared cache)...")
    full_boards = _build_board_cache(preds)

    results = {"n_predicted": actual_n, "metric_sizes": sizes}

    print(f"Tier 1: legality (n={sizes['legality']})...")
    sub, idx = _subsample(preds, sizes["legality"], seed=0)
    sub_boards = [full_boards[i] for i in idx]
    results["legality"] = legal_move_rate(sub, sub_boards)

    print(f"Tier 2: top-k accuracy (n={sizes['topk']})...")
    sub, _ = _subsample(preds, sizes["topk"], seed=1)
    results["topk"] = top_k_accuracy(sub, ks=(1, 3, 5))

    if stockfish_path is not None:
        print(f"Tier 2: stockfish agreement (n={sizes['stockfish']}, slow)...")
        sub, idx = _subsample(preds, sizes["stockfish"], seed=2)
        sub_boards = [full_boards[i] for i in idx]
        results["stockfish"] = stockfish_agreement(
            sub, sub_boards, stockfish_path, depth=10)

    print(f"Tier 3: mate-in-1 recall (n={sizes['mate_in_one']})...")
    sub, idx = _subsample(preds, sizes["mate_in_one"], seed=3)
    sub_boards = [full_boards[i] for i in idx]
    results["mate_in_one"] = mate_in_one_recall(sub, sub_boards)

    return results


def print_eval_results(results: dict) -> None:
    """Pretty-print the dict from run_full_eval."""
    print(f"\n=== Eval results (predicted on {results['n_predicted']} positions) ===\n")

    leg = results["legality"]
    lo, hi = leg["ci95"]
    print(f"Legal move rate:      {leg['overall']:.1%}  "
          f"[{lo:.1%}, {hi:.1%}]   (n={leg['n_positions']})")
    for sym in "PNBRQK":
        rate = leg["by_piece"].get(sym)
        n_sym = leg["by_piece_n"].get(sym, 0)
        if rate is not None:
            print(f"  {sym}: {rate:.1%}  (n={n_sym})")

    topk = results["topk"]
    print(f"\nTop-k accuracy (n={topk['n_positions']}):")
    for k in (1, 3, 5):
        lo, hi = topk[f"top{k}_ci95"]
        print(f"  top-{k}: {topk[f'top{k}']:.1%}  [{lo:.1%}, {hi:.1%}]")

    if "stockfish" in results and results["stockfish"]:
        sf = results["stockfish"]
        if sf["agreement_rate"] is not None:
            print(f"\nStockfish agreement:  {sf['agreement_rate']:.1%} "
                  f"(within 50cp, n={sf['sampled']}, "
                  f"skipped {sf['n_illegal_skipped']} illegal)")
            print(f"Median cp loss:       {sf['median_cp_loss']:.0f}")
            print(f"Mean cp loss:         {sf['mean_cp_loss']:.0f}")

    m = results["mate_in_one"]
    if m["recall"] is not None:
        lo, hi = m["ci95"]
        print(f"\nMate-in-1 recall:     {m['recall']:.1%}  [{lo:.1%}, {hi:.1%}]   "
              f"({m['n_found']}/{m['n_positions_with_mate']})")
    else:
        print(f"\nMate-in-1: no positions in sample contained a mate-in-1")