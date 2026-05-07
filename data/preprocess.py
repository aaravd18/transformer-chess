"""Convert a PGN file into pre-tokenized binary arrays for training.

Output layout (under `out_dir`):
    train/
        tokens.npy     (N_train, 68) uint8   -- model input
        from_sq.npy    (N_train,)    uint8   -- policy target: from-square
        to_sq.npy      (N_train,)    uint8   -- policy target: to-square
        promotion.npy  (N_train,)    uint8   -- 0 if not a promotion, 1..4 otherwise
    val/
        tokens.npy     (N_val, 68) uint8
        from_sq.npy    (N_val,)    uint8
        to_sq.npy      (N_val,)    uint8
        promotion.npy  (N_val,)    uint8
    meta.json                                -- {"n_positions": N, "n_games": G, ...}

We do two passes over the PGN: pass 1 counts positions per game so we
can pre-allocate memmaps for each split, pass 2 fills them. This keeps
memory bounded regardless of dataset size and produces files that the
training loop can mmap directly (no full load into RAM).

The train/val split happens at the GAME level (not position level) to
avoid leaking positions from the same game across splits. The split is
deterministic given `seed`.

Why memmap-friendly .npy instead of a single .npz: random-access
shuffling for training works directly on memmaps, you can inspect any
field without decompressing, and there's no penalty for skipping the
value field (which we don't use here -- the value head was removed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import shutil

from tokenizer import *



# ---------------------------------------------------------------------------
# Pass 1: count positions
# ---------------------------------------------------------------------------

def _count_positions(
    pgn_path: Path,
    log_every_games: int = 1000,
) -> tuple[list[int], int, int]:
    """Walk the PGN once and tally positions per game.

    Returns:
        per_game_counts: list of position counts, one entry per usable game
        n_games_used:    games with a usable result tag
        n_games_skipped: games we'll skip (no decisive/drawn result)
    """
    per_game_counts: list[int] = []
    n_skipped = 0
    n_total = 0  # games seen, including skipped
    t0 = time.time()
    last_log_total = 0

    with pgn_path.open() as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            n_total += 1

            # Match iter_positions's filter: skip games with no Result tag
            # or "*". We can check this without walking the moves.
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                n_skipped += 1
            else:
                # Count plies in the mainline. This is cheap -- python-chess
                # has already parsed the moves into the game tree.
                per_game_counts.append(sum(1 for _ in game.mainline_moves()))

            if n_total - last_log_total >= log_every_games:
                dt = time.time() - t0
                n_used = len(per_game_counts)
                n_pos = sum(per_game_counts)
                games_per_s = n_total / dt
                pos_per_s = n_pos / dt
                print(f"  {n_total:>8,} games scanned  "
                      f"({games_per_s:,.0f} games/s, {pos_per_s:,.0f} pos/s)  "
                      f"[used={n_used:,} skipped={n_skipped:,} positions={n_pos:,}]",
                      file=sys.stderr)
                last_log_total = n_total

    return per_game_counts, len(per_game_counts), n_skipped


# ---------------------------------------------------------------------------
# Pass 2: fill the arrays
# ---------------------------------------------------------------------------

def _fill_arrays(
    pgn_path: Path,
    arrays_by_split: dict,   # {"train": {...}, "val": {...}}
    is_val: np.ndarray,      # bool array, one entry per usable game
    log_every: int = 5000,
) -> dict[str, int]:
    """Walk the PGN a second time and write into the right split.

    For each usable game, `is_val[game_idx]` decides whether its
    positions go into the train or val arrays.

    Returns the actual number of positions written per split. Should
    match the pass-1 totals, but we return them so the caller can
    sanity-check.
    """
    indices = {"train": 0, "val": 0}
    game_idx = 0
    t0 = time.time()
    last_log_total = 0

    with pgn_path.open() as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Same filter as pass 1, so game_idx stays aligned with is_val.
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            split = "val" if is_val[game_idx] else "train"
            arrs = arrays_by_split[split]
            idx = indices[split]

            for rec in iter_positions(game):
                arrs["tokens"][idx]    = rec["tokens"]
                arrs["from_sq"][idx]   = rec["from_sq"]
                arrs["to_sq"][idx]     = rec["to_sq"]
                arrs["promotion"][idx] = rec["promotion"]
                # Note: rec["value"] is intentionally dropped.
                idx += 1

            indices[split] = idx
            game_idx += 1

            total = indices["train"] + indices["val"]
            if total - last_log_total >= log_every:
                rate = total / (time.time() - t0)
                print(f"  {total:>9,} positions  ({rate:,.0f}/s)  "
                      f"[train={indices['train']:,} val={indices['val']:,}]",
                      file=sys.stderr)
                last_log_total = total

    return indices


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def binarize_pgn(
    pgn_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> dict:
    """Tokenize all positions in `pgn_path` and write binary arrays to `out_dir`.

    Args:
        pgn_path:     path to a .pgn file (one or many games concatenated)
        out_dir:      directory to write train/ and val/ subdirectories.
                      Will be created if it doesn't exist. Existing files
                      with the same names will be overwritten.
        val_fraction: fraction of *games* (not positions) to put in val.
                      Defaults to 0.05. The position-level fraction will
                      be approximately equal, with small variance from
                      game-length differences.
        seed:         RNG seed for the train/val split. Fix this for
                      reproducible eval numbers across runs.

    Returns:
        the metadata dict that was written to meta.json.
    """
    pgn_path  = Path(pgn_path)
    out_dir   = Path(out_dir)
    train_dir = out_dir / "train"
    val_dir   = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not pgn_path.exists():
        raise FileNotFoundError(pgn_path)

    # ---- Pass 1: count ----
    print(f"Pass 1: counting positions in {pgn_path}...", file=sys.stderr)
    t0 = time.time()
    per_game_counts, n_used, n_skipped = _count_positions(pgn_path)
    n_positions = sum(per_game_counts)
    dt = time.time() - t0
    print(f"  {n_positions:,} positions across {n_used:,} games "
          f"(skipped {n_skipped:,}) in {dt:.1f}s",
          file=sys.stderr)

    if n_positions == 0:
        raise RuntimeError(
            f"No usable positions found in {pgn_path}. "
            f"Check that games have a Result tag (1-0, 0-1, or 1/2-1/2)."
        )

    # ---- Decide train/val split at the GAME level ----
    # Splitting by game (not position) prevents positions from the same
    # game leaking across splits, which would let the model memorize
    # game-specific patterns and inflate val accuracy.
    rng = np.random.default_rng(seed)
    is_val = rng.random(n_used) < val_fraction
    counts = np.asarray(per_game_counts)
    n_val   = int(counts[is_val].sum())
    n_train = int(counts[~is_val].sum())
    print(f"  split: {n_train:,} train / {n_val:,} val "
          f"({n_val / n_positions:.1%} val)", file=sys.stderr)

    # ---- Allocate output arrays as memmaps ----
    # We write directly into the .npy files via np.lib.format. The
    # simpler path -- np.memmap with raw .bin files -- loses the dtype
    # and shape headers, so np.load can't read them back without extra
    # plumbing. Using open_memmap gives us proper .npy files.
    def _alloc(split_dir: Path, n: int) -> dict:
        return {
            "tokens": np.lib.format.open_memmap(
                split_dir / "tokens.npy", mode="w+",
                dtype=np.uint8, shape=(n, SEQ_LEN)),
            "from_sq": np.lib.format.open_memmap(
                split_dir / "from_sq.npy", mode="w+",
                dtype=np.uint8, shape=(n,)),
            "to_sq": np.lib.format.open_memmap(
                split_dir / "to_sq.npy", mode="w+",
                dtype=np.uint8, shape=(n,)),
            "promotion": np.lib.format.open_memmap(
                split_dir / "promotion.npy", mode="w+",
                dtype=np.uint8, shape=(n,)),
        }

    arrays_by_split = {
        "train": _alloc(train_dir, n_train),
        "val":   _alloc(val_dir,   n_val),
    }

    # ---- Pass 2: fill ----
    print(f"Pass 2: tokenizing into {out_dir}...", file=sys.stderr)
    t0 = time.time()
    written = _fill_arrays(pgn_path, arrays_by_split, is_val)
    dt = time.time() - t0
    total_written = written["train"] + written["val"]
    print(f"  wrote {written['train']:,} train + {written['val']:,} val "
          f"in {dt:.1f}s ({total_written / dt:,.0f}/s)", file=sys.stderr)

    if written["train"] != n_train or written["val"] != n_val:
        # Should never happen, but better to fail loudly than to ship
        # silently truncated arrays.
        raise RuntimeError(
            f"Position count mismatch: pass 1 said train={n_train} val={n_val}, "
            f"pass 2 wrote train={written['train']} val={written['val']}"
        )

    # Flush memmaps to disk before we exit.
    for arrs in arrays_by_split.values():
        for arr in arrs.values():
            arr.flush()

    # ---- Metadata ----
    def _meta_for(n: int) -> dict:
        return {
            "n_positions": n,
            "fields": {
                "tokens":    {"shape": [n, SEQ_LEN], "dtype": "uint8"},
                "from_sq":   {"shape": [n],          "dtype": "uint8"},
                "to_sq":     {"shape": [n],          "dtype": "uint8"},
                "promotion": {"shape": [n],          "dtype": "uint8"},
            },
        }

    meta = {
        "n_positions":     int(n_positions),
        "n_games":         int(n_used),
        "n_games_skipped": int(n_skipped),
        "seq_len":         int(SEQ_LEN),
        "pgn_source":      str(pgn_path),
        "val_fraction":    float(val_fraction),
        "seed":            int(seed),
        "train":           _meta_for(n_train),
        "val":             _meta_for(n_val),
    }
    (out_dir   / "meta.json").write_text(json.dumps(meta,          indent=2))
    # Per-split meta files so each split directory is self-describing.
    (train_dir / "meta.json").write_text(json.dumps(meta["train"], indent=2))
    (val_dir   / "meta.json").write_text(json.dumps(meta["val"],   indent=2))

    total_bytes = 0
    for split_dir in (train_dir, val_dir):
        for name in ("tokens.npy", "from_sq.npy", "to_sq.npy", "promotion.npy"):
            total_bytes += (split_dir / name).stat().st_size
    print(f"Done. {total_bytes / 1e6:.1f} MB written to {out_dir}",
          file=sys.stderr)

    return meta


def build_dataset(local_out, drive_out=None):
    binarize_pgn("filtered_games.pgn", local_out)

    if drive_out:
        print(f"Copying to Drive: {drive_out}")
        shutil.copytree(local_out, drive_out, dirs_exist_ok=True)
        print("Done.")


def copy_drive_to_local(drive_path, local_path):
    if not Path(local_path).exists():
        print(f"Copying {drive_path} -> {local_path}...")
        shutil.copytree(drive_path, local_path)
        print("Done.")
    else:
        print(f"{local_path} already exists, skipping copy.")


if __name__ == "__main__":
    # local_out = "/content/data"
    # drive_out = "data"
    build_dataset("datasets")