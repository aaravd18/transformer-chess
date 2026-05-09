import chess
import chess.engine
import torch
from evaluation.play import *

# ---------------------------------------------------------------------------
# Stockfish move function
# ---------------------------------------------------------------------------

def stockfish_move_fn(
    board: chess.Board,
    elo: int,
    engine: chess.engine.SimpleEngine,
    time_limit: float = 0.1,
) -> chess.Move:
    """Get Stockfish's move at a target Elo.

    We pass in a persistent `engine` handle rather than spawning a new
    process per move — Stockfish startup is slow (~100ms) and we'll be
    making thousands of move calls.

    Stockfish's UCI_Elo is typically clamped to [1320, 3190]. For weaker
    play, you'd want `Skill Level` (0-20) instead; if your model is very
    weak you may need to swap in that approach.
    """
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": elo,
    })
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_stockfish(
    model: torch.nn.Module,
    model_color: chess.Color,
    stockfish_elo: int,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    max_plies: int = 400,
) -> float:
    """Play one game; return result from model's perspective: 1, 0.5, or 0.

    max_plies guards against pathological non-terminating games — if the
    model produces low-quality moves that avoid checkmate but neither
    side captures or pushes pawns, we'd otherwise loop forever (well,
    until the 75-move rule, but that's still 150 plies of nothing).
    """
    board = chess.Board()
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == model_color:
            move = predict_move(model, board, device)
        else:
            move = stockfish_move_fn(board, stockfish_elo, engine)
        board.push(move)
        ply += 1

    # Score from model's perspective.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Hit max_plies without a natural termination — call it a draw.
        # This is a pragmatic choice; alternatives would be to score it
        # as a loss for the model (since it presumably caused the loop)
        # or to exclude the game entirely.
        return 0.5
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == model_color else 0.0


# ---------------------------------------------------------------------------
# Match at one Elo level
# ---------------------------------------------------------------------------
def run_match(
    model: torch.nn.Module,
    stockfish_elo: int,
    num_games: int,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
) -> dict:
    """Play `num_games` against Stockfish at a fixed Elo, alternating colors.

    Returns aggregate stats. We alternate strictly (W,B,W,B,...) rather
    than randomizing — with small num_games this gives lower-variance
    estimates than random color assignment.
    """
    wins = draws = losses = 0
    total_score = 0.0

    for i in range(num_games):
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        score = play_stockfish(model, model_color, stockfish_elo, engine, device)
        total_score += score
        if score == 1.0:
            wins += 1
        elif score == 0.5:
            draws += 1
        else:
            losses += 1

    score_rate = total_score / num_games
    return {
        "stockfish_elo": stockfish_elo,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": score_rate,
    }