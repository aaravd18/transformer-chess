"""
Downloads stockfish evaluated positions that have a clear best move (i.e best move has eval 100cp above next best move)
Groups by move type
Saves filtered positions to evals.csv
"""

"""
Downloads Stockfish evaluated positions that have a clear best move
(i.e. best move has eval 100cp above next best move).

Adds simple move labels using python-chess:
- best_move_san
- moving_piece
- is_capture
- is_check

Saves filtered positions to evals.csv.
"""

import csv
import json
import os

import chess
import requests
import zstandard as zstd


URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUTFILE = "evaluation/positions/evals.csv"

TARGET_ROWS = 100_000
MIN_DEPTH = 25
MIN_GAP_CP = 100
CHUNK_SIZE = 1024 * 1024  # 1MB decompressed chunks


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

def get_capture_info(move: chess.Move, moving_piece: str, is_capture: bool):
    """
    For bishop, rook, and queen captures, return:
    - capture_range
    - capture_direction

    capture_direction is one of:
    - diagonal
    - vertical
    - horizontal

    Returns (None, None) for unsupported moves.
    """
    if not is_capture or moving_piece not in {"bishop", "rook", "queen"}:
        return None, None

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)

    file_dist = abs(to_file - from_file)
    rank_dist = abs(to_rank - from_rank)

    capture_range = max(file_dist, rank_dist)

    if file_dist == rank_dist:
        capture_direction = "diagonal"
    elif file_dist == 0:
        capture_direction = "vertical"
    elif rank_dist == 0:
        capture_direction = "horizontal"
    else:
        capture_direction = None

    return capture_range, capture_direction


def first_move(pv_line: str):
    return pv_line.split()[0] if pv_line else None


def get_cp(pv):
    # Skip mate positions for clean centipawn comparison
    return pv.get("cp")


def select_eval(evals):
    """
    Pick the deepest eval with:
    - depth >= MIN_DEPTH
    - at least 2 PVs, so we can compare best vs next-best
    """
    candidates = [
        e for e in evals
        if e.get("depth", 0) >= MIN_DEPTH and len(e.get("pvs", [])) >= 2
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))


def label_best_move(fen: str, best_move_uci: str):
    """
    Use python-chess to label the best move.

    Returns:
    - best_move_san
    - moving_piece
    - is_capture
    - is_check
    - capture_range

    Returns None if the FEN/move is invalid.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_move_uci)
    except ValueError:
        return None

    if move not in board.legal_moves:
        return None

    piece = board.piece_at(move.from_square)
    if piece is None:
        return None

    moving_piece = PIECE_NAMES[piece.piece_type]

    best_move_san = board.san(move)
    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    capture_range, capture_direction = get_capture_info(
        move,
        moving_piece,
        is_capture
    )

    return {
        "best_move_san": best_move_san,
        "moving_piece": moving_piece,
        "is_capture": is_capture,
        "is_check": is_check,
        "capture_range": capture_range,
        "capture_direction": capture_direction,
    }


def process_row(row):
    fen = row.get("fen")
    eval_obj = select_eval(row.get("evals", []))

    if not fen or eval_obj is None:
        return None

    pvs = eval_obj["pvs"]

    best = pvs[0]
    second = pvs[1]

    best_cp = get_cp(best)
    second_cp = get_cp(second)

    if best_cp is None or second_cp is None:
        return None

    # pvs[0] is already the best line and pvs[1] is the next-best line.
    gap = abs(best_cp - second_cp)

    if gap < MIN_GAP_CP:
        return None

    best_move = first_move(best.get("line", ""))
    next_best = first_move(second.get("line", ""))

    if best_move is None or next_best is None:
        return None

    move_labels = label_best_move(fen, best_move)

    if move_labels is None:
        return None

    return {
        "fen": fen,
        "best_move": best_move,
        "best_move_san": move_labels["best_move_san"],
        "best_move_eval": best_cp,
        "next_best": next_best,
        "next_best_eval": second_cp,
        "moving_piece": move_labels["moving_piece"],
        "is_capture": move_labels["is_capture"],
        "is_check": move_labels["is_check"],
        "capture_range": move_labels["capture_range"],
        "capture_direction": move_labels["capture_direction"],
    }


def main():
    saved = 0
    seen = 0
    buffer = ""

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    with requests.get(URL, stream=True) as response:
        response.raise_for_status()

        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(response.raw)

        with open(OUTFILE, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fen",
                    "best_move",
                    "best_move_san",
                    "best_move_eval",
                    "next_best",
                    "next_best_eval",
                    "moving_piece",
                    "is_capture",
                    "is_check",
                    "capture_range",
                    "capture_direction"
                ]
            )
            writer.writeheader()

            while saved < TARGET_ROWS:
                chunk = stream_reader.read(CHUNK_SIZE)

                if not chunk:
                    break

                buffer += chunk.decode("utf-8")

                lines = buffer.split("\n")
                buffer = lines.pop()  # keep incomplete trailing line

                for line in lines:
                    seen += 1

                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    output_row = process_row(row)

                    if output_row is None:
                        continue

                    writer.writerow(output_row)
                    saved += 1

                    if saved % 5000 == 0:
                        print(f"Saved {saved:,} positions after scanning {seen:,} rows")

                    if saved >= TARGET_ROWS:
                        break

    print(f"Done. Saved {saved:,} positions to {OUTFILE}")


if __name__ == "__main__":
    main()