import chess
import chess.engine
import torch
from evaluation.play import *

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
    max_plies guards against non-terminating games 
    """

    board = chess.Board()
    ply = 0
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": stockfish_elo,
    })

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == model_color:
            move = predict_move(model, board, device)
        else:
            move = engine.play(board, chess.engine.Limit(time=0.1)).move
        board.push(move)
        ply += 1

    # Score from model's perspective.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Hit max_plies without a natural termination — call it a draw.
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
    """Play `num_games` against Stockfish at a fixed Elo, alternating colors."""
    
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