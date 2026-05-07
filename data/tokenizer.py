"""Data pipeline for the chess transformer.

This module turns chess positions into the model's input format. Two
public entry points:

    tokenize(board) -> np.ndarray
        Convert a chess.Board into a 68-element uint8 array.

    iter_positions(game) -> generator of dicts
        Walk a PGN game and yield one training record per ply.

The 68-token layout:
       0 = a1, 1 = b1 ... 63=h8
       0-63  : piece on each square         (vocab 13, 0=empty, 1-6=white P/N/B/R/Q/K, 7-12=black p/n/b/r/q/k)
       64    : castling rights              (vocab 16)
       65    : en passant target square     (vocab 65, 64='none')
       66    : halfmove clock bucket        (vocab 8)
       67    : [CLS] register, always 0     (vocab 1)

The board is canonicalized so the model always sees 'white to move'.
For black-to-move positions we mirror the board (flipping ranks and
swapping piece colors). When walking a game, we mirror the move and
flip the value-target sign in lockstep so all three -- tokens, move,
value -- are expressed in the same canonical frame.
"""
import numpy as np
import chess
import chess.pgn

# Vocab definitions. Exported so model.py can build matching embedding tables.
VOCAB_PIECE        = 13
VOCAB_CASTLING     = 16
VOCAB_EP           = 65   # 0..63 + 64 for "none"
VOCAB_HALFMOVE     = 8
VOCAB_CLS          = 1

SEQ_LEN = 68

# Token-position constants
PIECE_TOKENS    = slice(0, 64)
CASTLING_TOKEN  = 64
EP_TOKEN        = 65
HALFMOVE_TOKEN  = 66
CLS_TOKEN       = 67


# Piece -> integer 0..12. Empty is 0; white pieces 1-6 (P,N,B,R,Q,K);
# black pieces 7-12 (p,n,b,r,q,k).
_PIECE_TO_ID = {None: 0}
for color, base in [(chess.WHITE, 1), (chess.BLACK, 7)]:
    for offset, ptype in enumerate(
        [chess.PAWN, chess.KNIGHT, chess.BISHOP,
         chess.ROOK, chess.QUEEN, chess.KING]
    ):
        _PIECE_TO_ID[chess.Piece(ptype, color)] = base + offset


# Promotion encoding for the policy target. 0 = no promotion.
_PROMO_TO_ID = {
    None:           0,
    chess.KNIGHT:   1,
    chess.BISHOP:   2,
    chess.ROOK:     3,
    chess.QUEEN:    4,
}


# ---------------------------------------------------------------------------
# Bucketing helpers for the small-cardinality continuous fields.
# ---------------------------------------------------------------------------

def _bucket_halfmove(hm: int) -> int:
    """50-move-rule clock has flavor near certain thresholds."""
    if hm == 0:    return 0
    if hm < 10:    return 1
    if hm < 30:    return 2
    if hm < 50:    return 3
    if hm < 70:    return 4
    if hm < 80:    return 5
    if hm < 90:    return 6
    return 7


# ---------------------------------------------------------------------------
# Public API: tokenize one board.
# ---------------------------------------------------------------------------

def tokenize(board: chess.Board) -> np.ndarray:
    """Convert a chess.Board into a (68,) uint8 array of token IDs.

    Args:
        board: a python-chess Board (will not be mutated)

    Returns:
        np.ndarray of dtype uint8, shape (68,).
    """
    if board.turn == chess.BLACK:
        board = board.mirror()

    out = np.empty(SEQ_LEN, dtype=np.uint8)

    # Pieces on each square (tokens 0..63)
    for sq in range(64):
        out[sq] = _PIECE_TO_ID[board.piece_at(sq)]

    # Castling rights (token 64), 4 bits packed: WK | WQ<<1 | BK<<2 | BQ<<3
    # Same as 8*BQ + 4*BK + 2*WQ + WK
    cr = (
        (int(board.has_kingside_castling_rights(chess.WHITE))  << 0) |
        (int(board.has_queenside_castling_rights(chess.WHITE)) << 1) |
        (int(board.has_kingside_castling_rights(chess.BLACK))  << 2) |
        (int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
    )
    out[CASTLING_TOKEN] = cr

    # En passant target (token 65): 0..63 if available, else 64.
    out[EP_TOKEN] = 64 if board.ep_square is None else board.ep_square

    # Halfmove buckets.
    out[HALFMOVE_TOKEN] = _bucket_halfmove(board.halfmove_clock)

    # CLS register, always 0; gives the value head a fixed pooling slot.
    out[CLS_TOKEN] = 0

    return out


# ---------------------------------------------------------------------------
# Public API: walk a game and yield training records.
# ---------------------------------------------------------------------------

def _result_to_value(result_str: str):
    """PGN '1-0' / '0-1' / '1/2-1/2' -> {-1, 0, +1} from White's POV.

    Returns None for '*' or missing tags -- caller should skip those games.
    """
    if result_str == "1-0":     return  1
    if result_str == "0-1":     return -1
    if result_str == "1/2-1/2": return  0
    return None


def iter_positions(game: chess.pgn.Game):
    """Yield one training record per ply in `game`.

    Each record is a dict:
        tokens     : np.ndarray (68,) uint8 -- input to the transformer
        from_sq    : int 0..63             -- policy target: from-square
        to_sq      : int 0..63             -- policy target: to-square
        promotion  : int 0..4              -- 0 if not a promotion
        value      : int -1, 0, or +1      -- value-head target, in the
                                              side-to-move's POV in the
                                              canonical frame.

    Tokens, target move, and target value are all expressed in the same
    (canonical) frame. When the original position was black-to-move we:
        - mirror the board (handled inside tokenize())
        - mirror the played move (square_mirror flips rank, keeps file)
        - flip the value sign (game result is stored from white's POV)

    Args:
        game: a parsed chess.pgn.Game

    Yields nothing if the game has no decisive result tag.
    """
    result_white_pov = _result_to_value(game.headers.get("Result", "*"))
    if result_white_pov is None:
        return

    board = game.board()
    for move in game.mainline_moves():
        # Tokenize the position BEFORE pushing the move -- the model
        # predicts the move from the position.
        tokens  = tokenize(board)
        if board.turn == chess.BLACK:
            from_sq = chess.square_mirror(move.from_square)
            to_sq   = chess.square_mirror(move.to_square)
            value   = -result_white_pov
        else:
            from_sq = move.from_square
            to_sq   = move.to_square
            value   = result_white_pov

        yield {
            "tokens":    tokens,
            "from_sq":   from_sq,
            "to_sq":     to_sq,
            "promotion": _PROMO_TO_ID[move.promotion],
            "value":     value,
        }

        board.push(move)