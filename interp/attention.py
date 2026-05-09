"""Attention visualization for the chess transformer.

Two public entries:
    plot_attention_map(attn_row, board, square, ...) -> matplotlib Figure
        Render a single (68,) attention vector as an 8x8 board heatmap
        plus a small sidebar for the 4 non-square tokens.

    show_square_attention(model, board, square, ...) -> None
        Run the model on `board`, extract attention for `square` from
        every (layer, head), and plot them all.
"""
from __future__ import annotations
from pathlib import Path

import chess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import torch
import numpy as np
from data.tokenizer import *


_PIECE_GLYPHS = {
    chess.KING:   "\u265A",
    chess.QUEEN:  "\u265B",
    chess.ROOK:   "\u265C",
    chess.BISHOP: "\u265D",
    chess.KNIGHT: "\u265E",
    chess.PAWN:   "\u265F",
}


def plot_attention_map(
    attn_vector: np.ndarray,
    board: chess.Board,
    focus_square: int,
    ax: plt.Axes | None = None,
    title: str = "",
    cmap: str = "viridis",
    show_pieces: bool = True,
) -> plt.Axes:
    """Plot a single (68,) attention vector as an 8x8 board.

    The 4 non-square tokens (castling/ep/halfmove/CLS) are drawn as a
    small column to the right of the board so you can see how much
    attention is going to board-metadata vs squares.

    Args:
        attn_vector: shape (68,), an attention row OR column. Values
            don't have to sum to 1; we display them on their own scale.
        board: the *original*, un-canonicalized board the user passed in.
            Used purely to overlay piece glyphs and rank/file labels.
        focus_square: the square (in `board`'s coordinates, 0..63) that
            this attention vector is "about". Outlined in red.
        ax: optional matplotlib Axes; one is created if not given.
        title: subplot title.
        cmap: colormap.
        show_pieces: overlay piece glyphs from `board` on the heatmap.

    Returns the Axes.
    """
    assert attn_vector.shape == (SEQ_LEN,), \
        f"expected ({SEQ_LEN},), got {attn_vector.shape}"

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # The attention vector is in canonical (white-to-move) frame. If the
    # user's board is black-to-move, we need to un-mirror for display so
    # squares line up with what they typed.
    square_attn = attn_vector[:64].copy()
    if board.turn == chess.BLACK:
        # Mirroring in tokenize() flips ranks (square_mirror in python-chess
        # = sq ^ 56). Apply the same permutation to undo it for display.
        square_attn = square_attn[[sq ^ 56 for sq in range(64)]]

    # python-chess square index = file + 8*rank. Reshaping (64,) -> (8, 8)
    # gives grid[rank, file] directly. With origin="lower" below, row 0
    # (rank 0 = white's back rank) is drawn at the bottom of the figure,
    # which matches standard chess board orientation.
    grid = square_attn.reshape(8, 8)

    vmax = float(square_attn.max()) if square_attn.max() > 0 else 1.0
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=vmax,
                   extent=(-0.5, 7.5, -0.5, 7.5), origin="lower")

    # Overlay pieces. White pieces are drawn in white with a black outline,
    # black pieces in black with a white outline -- so both are legible
    # regardless of cell brightness.
    if show_pieces:
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                continue
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            glyph = _PIECE_GLYPHS[piece.piece_type]
            if piece.color == chess.WHITE:
                fill_color = "white"
                stroke_color = "black"
            else:
                fill_color = "black"
                stroke_color = "white"

            ax.text(
                file, rank, glyph,
                ha="center", va="center",
                fontsize=20, color=fill_color,
                path_effects=[pe.withStroke(linewidth=2,
                                            foreground=stroke_color)],
            )

    # Outline the focus square.
    f = chess.square_file(focus_square)
    r = chess.square_rank(focus_square)
    ax.add_patch(Rectangle((f - 0.5, r - 0.5), 1, 1,
                           fill=False, edgecolor="red", linewidth=2.5))

    # File/rank labels.
    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(1, 9))
    ax.set_title(title, fontsize=10)

    # Sidebar for the 4 non-square tokens.
    meta_labels = ["cast", "ep", "hm", "cls"]
    meta_vals = attn_vector[64:68]
    # Draw as a thin column to the right of the board.
    for i, (lab, val) in enumerate(zip(meta_labels, meta_vals)):
        # Map to same color scale as the board.
        color = plt.get_cmap(cmap)(val / vmax if vmax > 0 else 0.0)
        ax.add_patch(Rectangle((8.2, 7 - 2 * i - 1), 0.8, 1.6,
                               facecolor=color, edgecolor="gray",
                               clip_on=False))
        ax.text(8.6, 7 - 2 * i - 0.2, lab, ha="center", va="center",
                fontsize=8, clip_on=False)
        ax.text(8.6, 7 - 2 * i - 0.7, f"{val:.2f}",
                ha="center", va="center", fontsize=7, clip_on=False)

    ax.set_xlim(-0.5, 9.2)
    ax.set_aspect("equal")

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return ax

def show_square_attention(
    model,                       # ChessTransformer
    board: chess.Board,
    square: int,
    direction: str = "from",     # "from" = what `square` attends to,
                                 # "to"   = what attends to `square`
    layers: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> None:
    """Run the model on `board` and plot attention for `square` across
    every head of every requested layer.

    Args:
        model: a ChessTransformer in eval mode.
        board: position to analyze. Will not be mutated.
        square: square index 0..63 in the user's board frame.
        direction: "from" plots the row of A (where `square` is looking),
                   "to"   plots the column (who is looking at `square`).
        layers: which layers to plot. None = all.
        device: where to run the model.
    """
    assert 0 <= square < 64
    assert direction in ("from", "to")

    model = model.to(device).eval()

    # Tokenize -- this canonicalizes to white-to-move internally.
    tokens = torch.from_numpy(tokenize(board)).long().unsqueeze(0).to(device)

    # Translate the user's square to the canonical frame so we index
    # the attention matrix correctly. tokenize() mirrors via board.mirror(),
    # which is a rank flip (sq ^ 56).
    canon_sq = square ^ 56 if board.turn == chess.BLACK else square

    # Turn on the existing hook on every attention module.
    attn_modules = [blk.attn for blk in model.blocks]
    for m in attn_modules:
        m._save_attn_probs = True

    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for m in attn_modules:
            m._save_attn_probs = False

    # Collect (layer, head, 68) vectors.
    if layers is None:
        layers = list(range(len(attn_modules)))

    n_heads = model.cfg.n_heads
    n_layers = len(layers)

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.2 * n_heads, 3.2 * n_layers),
                             squeeze=False)

    sq_name = chess.square_name(square)
    fig.suptitle(
        f"Attention {'from' if direction == 'from' else 'to'} {sq_name} "
        f"(turn: {'white' if board.turn else 'black'})",
        fontsize=12,
    )

    for row, layer_idx in enumerate(layers):
        # last_attn_probs is (1, H, L, L); drop batch.
        probs = attn_modules[layer_idx].last_attn_probs[0].cpu().numpy()
        for head in range(n_heads):
            if direction == "from":
                vec = probs[head, canon_sq, :]      # row: query=square
            else:
                vec = probs[head, :, canon_sq]      # col: key=square
            plot_attention_map(
                vec, board, square,
                ax=axes[row, head],
                title=f"L{layer_idx} H{head}",
            )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


# Visual constants. Tweak these to taste.
_CHESS_FONT_FAMILY = "DejaVu Sans"
_GLYPH_Y_OFFSET = -0.05

# rcParams applied while drawing. Used inside plt.rc_context so they
# don't leak into the rest of the notebook.
_RC_PARAMS = {
    "font.family":       "serif",
    "font.serif":        ["Charter", "Georgia", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "cm",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_get_attention(
    model,
    board: chess.Board,
    layer: int,
    device: str | torch.device,
) -> np.ndarray:
    """Run a forward pass and return attention probs from one layer.

    Returns: (n_heads, seq_len, seq_len) numpy array.
    """
    tokens = torch.from_numpy(tokenize(board)).long().unsqueeze(0).to(device)
    target_attn = model.blocks[layer].attn
    target_attn._save_attn_probs = True
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        target_attn._save_attn_probs = False
    return target_attn.last_attn_probs[0].cpu().numpy()


def _draw_pieces(ax: plt.Axes, board: chess.Board, fontsize: float) -> None:
    """Draw all pieces from `board` onto the heatmap `ax`."""
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        glyph = _PIECE_GLYPHS[piece.piece_type]

        if piece.color == chess.WHITE:
            fill, stroke = "white", "black"
        else:
            fill, stroke = "black", "white"

        ax.text(
            file, rank + _GLYPH_Y_OFFSET, glyph,
            ha="center", va="center",
            fontsize=fontsize, color=fill,
            family=_CHESS_FONT_FAMILY,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=stroke)],
        )


def _draw_metadata_sidebar(
    ax: plt.Axes,
    meta_vals: np.ndarray,
    vmax: float,
    cmap_obj,
) -> None:
    """Draw the 4-cell castling/ep/hm/cls sidebar."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    labels = ["castle", "ep", "hm", "cls"]
    cell_h = 1.6
    top_y = 7.3
    for i, (lab, val) in enumerate(zip(labels, meta_vals)):
        y = top_y - i * (cell_h + 0.15) - cell_h
        norm_val = float(val) / vmax if vmax > 0 else 0.0
        face = cmap_obj(norm_val)

        ax.add_patch(Rectangle(
            (0.05, y), 0.9, cell_h,
            facecolor=face, edgecolor="#cccccc", linewidth=0.8,
        ))
        text_color = "white" if norm_val < 0.55 else "black"
        ax.text(0.5, y + cell_h * 0.62, lab,
                ha="center", va="center",
                fontsize=9, color=text_color)
        ax.text(0.5, y + cell_h * 0.30, f"{float(val):.3f}",
                ha="center", va="center",
                fontsize=8, color=text_color, family="monospace")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_single_head_attention(
    model,                       # ChessTransformer
    board: chess.Board,
    square: int,
    layer: int,
    head: int,
    direction: str = "from",     # "from" = what `square` attends to,
                                 # "to"   = what attends to `square`
    save_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    figsize: tuple[float, float] = (7.0, 7.5),
) -> None:
    """Plot the attention map of a single (layer, head) for `square` in
    a polished, blog-ready style.

    The figure is rendered inline in the notebook and, if `save_path` is
    given, also saved as SVG.

    Args:
        model: a ChessTransformer.
        board: position to analyze. Will not be mutated.
        square: square index 0..63 in the user's board frame.
        layer: which transformer block to read attention from.
        head: which head within that block.
        direction: "from" -> what `square` is attending to (query=square),
                   "to"   -> what is attending to `square` (key=square).
        save_path: if given, save the figure as SVG. ".svg" extension is
            appended if missing.
        device: where to run the model.
        figsize: matplotlib figure size in inches.
    """
    # ----- input validation -----
    assert 0 <= square < 64, f"square {square} out of range"
    assert direction in ("from", "to"), \
        f"direction must be 'from' or 'to', got {direction!r}"
    assert 0 <= layer < len(model.blocks), \
        f"layer {layer} out of range (model has {len(model.blocks)} blocks)"
    assert 0 <= head < model.cfg.n_heads, \
        f"head {head} out of range (model has {model.cfg.n_heads} heads)"

    # ----- run model & extract attention -----
    model = model.to(device).eval()
    canon_sq = square ^ 56 if board.turn == chess.BLACK else square
    probs = _run_and_get_attention(model, board, layer, device)  # (H, L, L)

    if direction == "from":
        attn_vec = probs[head, canon_sq, :]
    else:
        attn_vec = probs[head, :, canon_sq]

    # Un-mirror for display if black to move.
    square_attn = attn_vec[:64].copy()
    if board.turn == chess.BLACK:
        square_attn = square_attn[[sq ^ 56 for sq in range(64)]]
    grid = square_attn.reshape(8, 8)
    vmax = float(square_attn.max()) if square_attn.max() > 0 else 1.0

    # ----- draw -----
    with plt.rc_context(_RC_PARAMS):
        fig = plt.figure(figsize=figsize, facecolor="white")
        gs = fig.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=[0.10, 1.0, 0.06],
            width_ratios=[1.0, 0.10],
            left=0.08, right=0.94, top=0.96, bottom=0.06,
            wspace=0.05, hspace=0.05,
        )

        ax_title = fig.add_subplot(gs[0, :])
        ax_board = fig.add_subplot(gs[1, 0])
        ax_meta  = fig.add_subplot(gs[1, 1])
        ax_title.axis("off")

        # Title block.
        sq_name = chess.square_name(square)
        turn_str = "white to move" if board.turn == chess.WHITE else "black to move"
        direction_str = (f"Attention from {sq_name}" if direction == "from"
                         else f"Attention to {sq_name}")

        ax_title.text(
            0.0, 0.7, direction_str,
            fontsize=18, fontweight="bold",
            ha="left", va="center", transform=ax_title.transAxes,
        )
        ax_title.text(
            0.0, 0.15,
            f"Layer {layer}  \u00b7  Head {head}  \u00b7  {turn_str}",
            fontsize=11, color="#555555",
            ha="left", va="center", transform=ax_title.transAxes,
        )

        # Board heatmap.
        im = ax_board.imshow(
            grid, cmap="viridis", vmin=0.0, vmax=vmax,
            extent=(-0.5, 7.5, -0.5, 7.5), origin="lower",
            interpolation="nearest",
        )

        # Subtle grid lines.
        for i in range(9):
            ax_board.axhline(i - 0.5, color="white", linewidth=0.6, alpha=0.3)
            ax_board.axvline(i - 0.5, color="white", linewidth=0.6, alpha=0.3)

        _draw_pieces(ax_board, board, fontsize=38)

        # Focus-square highlight with soft halo.
        f = chess.square_file(square)
        r = chess.square_rank(square)
        ax_board.add_patch(Rectangle(
            (f - 0.5, r - 0.5), 1, 1,
            fill=False, edgecolor="#d62728", linewidth=2.8,
            path_effects=[pe.withStroke(linewidth=5, foreground="#d6272833")],
        ))

        # File/rank labels.
        ax_board.set_xticks(range(8))
        ax_board.set_xticklabels(list("abcdefgh"), fontsize=10, color="#333333")
        ax_board.set_yticks(range(8))
        ax_board.set_yticklabels(range(1, 9), fontsize=10, color="#333333")
        ax_board.tick_params(length=0, pad=4)
        for spine in ax_board.spines.values():
            spine.set_visible(False)
        ax_board.set_aspect("equal")
        ax_board.set_xlim(-0.5, 7.5)
        ax_board.set_ylim(-0.5, 7.5)

        # Metadata sidebar.
        _draw_metadata_sidebar(
            ax_meta, attn_vec[64:68], vmax, plt.get_cmap("viridis"),
        )

        # Colorbar.
        cbar_ax = fig.add_axes((0.08, 0.04, 0.60, 0.018))
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=8, colors="#555555", pad=3)
        cbar.set_ticks([0.0, vmax])
        cbar.set_ticklabels(["0", f"{vmax:.2f}"])

        fig.text(
            0.72, 0.048, "attention weight",
            fontsize=9, color="#555555",
            ha="left", va="center",
        )

        # ----- save before display -----
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix.lower() != ".svg":
                save_path = save_path.with_suffix(".svg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path, format="svg",
                bbox_inches="tight", facecolor="white",
                transparent=False,
            )
            print(f"saved -> {save_path}")

        plt.close(fig)
        return fig
    
def _plot_board_attention(
    square_attn: np.ndarray,
    board: chess.Board,
    focus_square: int,
    ax: plt.Axes,
    title: str = "",
    cmap: str = "viridis",
    show_pieces: bool = True,
) -> plt.Axes:
    """Like plot_attention_map but board-only (no meta sidebar).

    Expects `square_attn` to already be in the user's board frame
    (i.e. un-mirrored if the board was black-to-move) and length 64.
    """
    assert square_attn.shape == (64,), \
        f"expected (64,), got {square_attn.shape}"

    grid = square_attn.reshape(8, 8)
    vmax = float(square_attn.max()) if square_attn.max() > 0 else 1.0
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=vmax,
                   extent=(-0.5, 7.5, -0.5, 7.5), origin="lower")

    if show_pieces:
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                continue
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            glyph = _PIECE_GLYPHS[piece.piece_type]
            if piece.color == chess.WHITE:
                fill_color, stroke_color = "white", "black"
            else:
                fill_color, stroke_color = "black", "white"
            ax.text(
                file, rank, glyph,
                ha="center", va="center",
                fontsize=16, color=fill_color,
                path_effects=[pe.withStroke(linewidth=2,
                                            foreground=stroke_color)],
            )

    f = chess.square_file(focus_square)
    r = chess.square_rank(focus_square)
    ax.add_patch(Rectangle((f - 0.5, r - 0.5), 1, 1,
                           fill=False, edgecolor="red", linewidth=2.0))

    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"), fontsize=7)
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(1, 9), fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    return ax


def show_multi_square_attention(
    model,                                # ChessTransformer
    board: chess.Board,
    square_to_layer: dict[int, int],
    direction: str = "from",              # "from" or "to"
    device: str | torch.device = "cpu",
    cmap: str = "viridis",
) -> None:
    """Plot attention for a set of (square, layer) pairs.

    Each column of the output is one square at its specified layer; each
    of the 8 rows is one head. Non-board tokens are excluded so the
    plotted vector is the (64,) board-only slice (not renormalized).

    Args:
        model: a ChessTransformer in eval mode.
        board: position to analyze; not mutated.
        square_to_layer: {square_index: layer_index}. Square indices are
            in the user's board frame (0..63).
        direction: "from" = row of A (square is the query),
                   "to"   = column of A (square is the key).
        device: where to run the model.
        cmap: matplotlib colormap.
    """
    assert direction in ("from", "to")
    for sq, layer in square_to_layer.items():
        assert 0 <= sq < 64, f"bad square {sq}"
        assert 0 <= layer < len(model.blocks), \
            f"bad layer {layer} for model with {len(model.blocks)} layers"

    model = model.to(device).eval()
    tokens = torch.from_numpy(tokenize(board)).long().unsqueeze(0).to(device)

    needed_layers = sorted(set(square_to_layer.values()))
    attn_modules = [model.blocks[i].attn for i in needed_layers]
    for m in attn_modules:
        m._save_attn_probs = True
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for m in attn_modules:
            m._save_attn_probs = False

    layer_probs = {
        i: model.blocks[i].attn.last_attn_probs[0].cpu().numpy()
        for i in needed_layers
    }

    n_heads = model.cfg.n_heads
    items = list(square_to_layer.items())
    n_cols = len(items)

    fig, axes = plt.subplots(
        n_heads, n_cols,
        figsize=(3.6 * n_cols, 3.6 * n_heads),
        squeeze=False,
    )
    fig.suptitle(
        f"Attention {direction} target squares "
        f"(turn: {'white' if board.turn else 'black'})",
        fontsize=13,
    )

    unmirror = [sq ^ 56 for sq in range(64)] if board.turn == chess.BLACK \
               else list(range(64))

    # Column headers: one per square, on the top row only.
    for col, (square, layer_idx) in enumerate(items):
        sq_name = chess.square_name(square)
        axes[0, col].set_title(
            f"{sq_name}  (L{layer_idx})", fontsize=12, pad=10,
        )

    # Row headers: head index, on the leftmost column only.
    for head in range(n_heads):
        axes[head, 0].set_ylabel(
            f"H{head}", fontsize=12, rotation=0,
            ha="right", va="center", labelpad=20,
        )

    for col, (square, layer_idx) in enumerate(items):
        canon_sq = square ^ 56 if board.turn == chess.BLACK else square
        probs = layer_probs[layer_idx]   # (H, 68, 68)

        for head in range(n_heads):
            if direction == "from":
                vec = probs[head, canon_sq, :64]
            else:
                vec = probs[head, :64, canon_sq]
            vec = vec[unmirror]

            _plot_board_attention(
                vec, board, square,
                ax=axes[head, col],
                title="",        # titles handled by column headers above
                cmap=cmap,
            )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()