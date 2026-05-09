"""Play with a trained model: get move predictions, see top-k options,
or play a full game against it.

Three things you'll likely use:

    predict_move(model, board, device)
        Returns the model's top legal move for the given position.

    suggest_moves(model, board, device, k=5)
        Returns the top-k legal moves with their probabilities.

    play_game(model, device)
        Interactive REPL where you play a game against the model.

Everything handles the canonical-frame conversion automatically: the
caller passes a normal chess.Board (white or black to move), and we
mirror under the hood as needed before/after calling the model.
"""
from __future__ import annotations

import torch
import chess
from model import *


# ---------------------------------------------------------------------------
# The core: take a board, return a probability distribution over legal moves
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_legal_move_probs(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    temperature: float = 1.0,
) -> list[tuple[chess.Move, float]]:
    """Return [(move, probability), ...] over all legal moves, sorted high->low.

    How this works end-to-end:
      1. tokenize() canonicalizes (mirrors black-to-move boards) and emits
         the 68-token input.
      2. Forward pass gives us a 64x64 logit grid + a per-source-square
         promotion head.
      3. We mask: build a tensor that's -inf everywhere except at legal
         (from, to) entries, then add. Softmax then gives a clean
         distribution over only legal moves.
      4. We map predictions back to the user's frame -- if the original
         board was black-to-move, we mirror move squares back so the
         returned moves are sensible to push() on the original board.

    temperature scales logits before softmax. 1.0 = model's native
    distribution; <1 sharpens (more confident); >1 flattens (more random).
    """
    was_black_to_move = (board.turn == chess.BLACK)

    # ---- Tokenize (mirrors the board internally if black to move) ----
    tokens_np = tokenize(board)
    tokens = torch.from_numpy(tokens_np).long().unsqueeze(0).to(device)  # (1, 68)

    # ---- Forward ----
    model.eval()
    out = model(tokens)
    move_logits  = out["move_logits"][0]    # (64, 64)
    promo_logits = out["promo_logits"][0]   # (64, 5)

    # ---- Build the legality mask in the canonical frame ----
    # tokenize() may have mirrored the board, so we need legal moves in
    # the *canonical* (white-to-move) frame to match the logit indices.
    canonical_board = board.mirror() if was_black_to_move else board

    # legal_grid[i, j] = True iff there's a legal move from i to j.
    legal_grid = torch.full((64, 64), float("-inf"), device=device)
    canonical_legal_moves = list(canonical_board.legal_moves)

    # Track promotions per (from, to) so we can pick the right promotion
    # piece using the model's promo head, not arbitrarily default to queen.
    # Map: (from, to) -> list of legal promotion piece types (None means non-promo).
    from collections import defaultdict
    legal_promos = defaultdict(list)

    for m in canonical_legal_moves:
        legal_grid[m.from_square, m.to_square] = 0.0   # 0 means "allowed"
        legal_promos[(m.from_square, m.to_square)].append(m.promotion)

    # ---- Mask + temperature + softmax ----
    masked = (move_logits + legal_grid) / max(temperature, 1e-6)
    flat = masked.reshape(-1)                       # (4096,)
    probs = torch.softmax(flat, dim=-1).reshape(64, 64)

    # ---- Decode each legal move's probability and map back to user frame ----
    promo_id_to_piece = {1: chess.KNIGHT, 2: chess.BISHOP,
                         3: chess.ROOK, 4: chess.QUEEN}

    results = []
    seen_user_moves = set()  # dedupe -- multiple promotions share (from, to)

    for m in canonical_legal_moves:
        prob = probs[m.from_square, m.to_square].item()

        # If this (from, to) has multiple promotion options, split the
        # probability among them according to the promo head.
        promos_here = legal_promos[(m.from_square, m.to_square)]
        if len(promos_here) > 1:
            promo_dist = torch.softmax(promo_logits[m.from_square], dim=-1)
            # m.promotion is one of {KNIGHT, BISHOP, ROOK, QUEEN}; map to id 1..4.
            promo_id = next(pid for pid, pt in promo_id_to_piece.items()
                            if pt == m.promotion)
            move_prob = prob * promo_dist[promo_id].item()
        else:
            move_prob = prob

        # Map move back to user's frame.
        if was_black_to_move:
            user_move = chess.Move(
                chess.square_mirror(m.from_square),
                chess.square_mirror(m.to_square),
                promotion=m.promotion,
            )
        else:
            user_move = m

        # Dedupe on (from, to, promo) — covers the rare case where mirroring
        # collapses two distinct canonical moves to the same user move.
        key = (user_move.from_square, user_move.to_square, user_move.promotion)
        if key in seen_user_moves:
            continue
        seen_user_moves.add(key)

        results.append((user_move, move_prob))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def predict_move(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
) -> chess.Move:
    """Return the model's single best legal move."""
    moves = get_legal_move_probs(model, board, device)
    if not moves:
        raise ValueError("No legal moves -- is the position terminal?")
    return moves[0][0]


def suggest_moves(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    k: int = 5,
    temperature: float = 1.0,
) -> list[tuple[chess.Move, float]]:
    """Return the top-k legal moves with probabilities."""
    moves = get_legal_move_probs(model, board, device, temperature)
    return moves[:k]


def show_suggestions(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    k: int = 5,
) -> None:
    """Pretty-print the top-k suggestions in SAN."""
    print(board)
    print(f"\n{'White' if board.turn == chess.WHITE else 'Black'} to move.")
    print(f"Top {k} model suggestions:")
    for move, prob in suggest_moves(model, board, device, k=k):
        san = board.san(move)
        print(f"  {san:<8}  {prob:>6.1%}")


# ---------------------------------------------------------------------------
# Sampling: pick a move stochastically (useful for varied play)
# ---------------------------------------------------------------------------

def sample_move(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    temperature: float = 1.0,
) -> chess.Move:
    """Sample a legal move from the model's distribution.

    With temperature=1.0 this gives "natural" play -- the model picks
    each move proportional to its predicted probability. Lowering temp
    -> more deterministic. Raising it -> more random/exploratory.
    """
    moves_with_probs = get_legal_move_probs(model, board, device, temperature)
    if not moves_with_probs:
        raise ValueError("No legal moves")

    moves, probs = zip(*moves_with_probs)
    probs_t = torch.tensor(probs)
    # Renormalize -- get_legal_move_probs returns probs that sum to ~1
    # already, but float drift can leave them slightly off.
    probs_t = probs_t / probs_t.sum()
    idx = torch.multinomial(probs_t, num_samples=1).item()
    return moves[idx]


# ---------------------------------------------------------------------------
# Interactive play loop
# ---------------------------------------------------------------------------

def _make_renderer(size: int = 400):
    """Returns a render(board, last_move) function that updates a single
    SVG display cell in place.

    Why a closure over a display handle instead of clear_output(): on
    Colab, clear_output() doesn't cooperate with input() -- the input
    prompt ends up hidden. The display_id mechanism updates the SVG
    in place without clearing surrounding outputs, which means input()
    works normally below it.
    """
    try:
        from IPython.display import display, SVG
        import chess.svg

        # Create the display cell once with a placeholder; capture its handle.
        handle = display(SVG('<svg xmlns="http://www.w3.org/2000/svg"/>'),
                         display_id=True)

        def render(board: chess.Board, last_move: chess.Move | None = None) -> None:
            svg = chess.svg.board(
                board,
                size=size,
                lastmove=last_move,
                check=board.king(board.turn) if board.is_check() else None,
            )
            handle.update(SVG(svg))

        return render
    except ImportError:
        # Not in a notebook -- fall back to plain text printing.
        def render(board, last_move=None):
            print(board)
        return render


def play_game(
    model: torch.nn.Module,
    device: torch.device,
    human_color: chess.Color = chess.WHITE,
    temperature: float = 0.3,
    show_suggestions_each_turn: bool = False,
    board_size: int = 400,
) -> chess.Board:
    """Play a full game against the model. Renders an SVG board in Jupyter.

    Type moves in SAN ('e4', 'Nf3', 'O-O') or UCI ('e2e4'). Special
    commands:
        'quit' / 'q'  -> abort
        'undo'        -> take back the last full move (yours and model's)
        'help'        -> show top model suggestions for the current position

    Lower temperature (e.g. 0.3) makes the model play its top choices
    more often. Default 0.5 gives reasonable variety without too many
    weak moves.

    Note on Jupyter: the SVG board updates in place above the input via
    a persistent display handle (display_id pattern). This works on Colab,
    classic Jupyter, and JupyterLab without the clear_output()+input()
    interaction problems.
    """
    board = chess.Board()
    history = []
    last_move: chess.Move | None = None

    # One persistent SVG cell that gets updated in place each turn.
    render = _make_renderer(size=board_size)
    render(board, last_move)

    while not board.is_game_over():
        if board.turn == human_color:
            # ---- Human turn ----
            move_num = len(board.move_stack) // 2 + 1
            color_str = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n{color_str} to move ({move_num}). Commands: quit | undo | help")

            while True:
                try:
                    user_input = input("Your move: ").strip()
                except EOFError:
                    print("Aborted.")
                    return board

                if user_input in ("quit", "q"):
                    print("Aborted.")
                    return board
                if user_input == "help":
                    print("Model's top suggestions for *your* position:")
                    for m, p in suggest_moves(model, board, device, k=5):
                        print(f"  {board.san(m):<8}  {p:>6.1%}")
                    continue
                if user_input == "undo":
                    if len(history) >= 2:
                        board.pop(); board.pop()
                        history.pop(); history.pop()
                        last_move = history[-1] if history else None
                        render(board, last_move)
                        print("Undid one full move.")
                        continue
                    else:
                        print("Nothing to undo.")
                        continue

                # Try to parse the move.
                try:
                    move = board.parse_san(user_input)
                except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError,
                        chess.AmbiguousMoveError):
                    try:
                        move = chess.Move.from_uci(user_input)
                        if move not in board.legal_moves:
                            raise ValueError("not legal")
                    except (ValueError, chess.InvalidMoveError):
                        print(f"Couldn't parse '{user_input}'. Try SAN ('e4', 'Nf3') or UCI ('e2e4').")
                        continue
                break

            history.append(move)
            board.push(move)
            last_move = move
            render(board, last_move)

        else:
            # ---- Model turn ----
            move = sample_move(model, board, device, temperature=temperature)
            san = board.san(move)

            if show_suggestions_each_turn:
                print("\nModel's top 3 candidates were:")
                for m, p in suggest_moves(model, board, device, k=3):
                    marker = "*" if m == move else " "
                    print(f"  {marker} {board.san(m):<8}  {p:>6.1%}")

            history.append(move)
            board.push(move)
            last_move = move
            render(board, last_move)
            print(f"Model played: {san}")

    # ---- Game over ----
    outcome = board.outcome()
    print(f"\nGame over: {outcome.result()}  ({outcome.termination.name})")
    return board


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load a checkpoint and show predictions on a few positions.
    ckpt_path = "checkpoints/5M-longer-run.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(ckpt_path, device)

    # Test position 1: starting position
    print("\n=== Starting position ===")
    show_suggestions(model, chess.Board(), device, k=5)

    # Test position 2: classic Italian-game-ish middlegame
    print("\n=== After 1.e4 e5 2.Nf3 Nc6 ===")
    b = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        b.push_san(san)
    show_suggestions(model, b, device, k=5)

    # Test position 3: a tactical position (mate in 1)
    # Famous "Scholar's mate" position one move before the mate.
    print("\n=== Scholar's mate setup (Qxf7# next?) ===")
    b = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]:
        b.push_san(san)
    show_suggestions(model, b, device, k=5)