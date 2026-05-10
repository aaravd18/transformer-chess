import chess
import chess.engine
import chess.pgn
import torch
from pathlib import Path
from evaluation.play import *


def play_stockfish(
    model: torch.nn.Module,
    model_color: chess.Color,
    stockfish_elo: int,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    max_plies: int = 400,
    game_id: int | None = None,
) -> tuple[float, chess.pgn.Game]:
    """Play one game and return (score from model perspective, PGN game)."""

    board = chess.Board()
    moves = []
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

        moves.append(move)
        board.push(move)
        ply += 1

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        result = "1/2-1/2"
        score = 0.5
        termination = "Max plies reached"
    else:
        result = board.result(claim_draw=True)
        termination = outcome.termination.name

        if outcome.winner is None:
            score = 0.5
        else:
            score = 1.0 if outcome.winner == model_color else 0.0

    # Build PGN
    game = chess.pgn.Game()
    game.headers["Event"] = f"Model vs Stockfish {stockfish_elo}"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_id) if game_id is not None else "?"
    game.headers["White"] = "Model" if model_color == chess.WHITE else f"Stockfish {stockfish_elo}"
    game.headers["Black"] = f"Stockfish {stockfish_elo}" if model_color == chess.WHITE else "Model"
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["ModelColor"] = "White" if model_color == chess.WHITE else "Black"
    game.headers["StockfishElo"] = str(stockfish_elo)

    node = game
    replay_board = chess.Board()

    for move in moves:
        node = node.add_variation(move)
        replay_board.push(move)

    return score, game


def run_match(
    model: torch.nn.Module,
    stockfish_elo: int,
    num_games: int,
    engine: chess.engine.SimpleEngine,
    device: torch.device,
    pgn_path: str | Path = "model_vs_stockfish.pgn",
) -> dict:
    """Play num_games against Stockfish and save all games to PGN."""

    wins = draws = losses = 0
    total_score = 0.0
    games = []

    for i in range(num_games):
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK

        score, game = play_stockfish(
            model=model,
            model_color=model_color,
            stockfish_elo=stockfish_elo,
            engine=engine,
            device=device,
            game_id=i + 1,
        )

        games.append(game)
        total_score += score

        if score == 1.0:
            wins += 1
        elif score == 0.5:
            draws += 1
        else:
            losses += 1

    pgn_path = Path(pgn_path)
    with open(pgn_path, "w", encoding="utf-8") as f:
        for game in games:
            print(game, file=f, end="\n\n")

    score_rate = total_score / num_games

    return {
        "stockfish_elo": stockfish_elo,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": score_rate,
        "pgn_path": str(pgn_path),
    }