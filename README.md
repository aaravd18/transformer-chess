# Mechanistic Interpretability of a Chess Transformer

Training a transformer to play chess, then taking it apart to see how it works.

---

## Architecture

**6.5M parameters** — 8 layers, 8 heads, d_model 256, d_ff 1024, pre-norm, bidirectional attention.

Design choices for interpretability:

**68-token board representation.** Tokens 0–63 are the 64 board squares in python-chess order (a1=0, …, h8=63). Tokens 64–66 are metadata (castling rights, en passant target, halfmove clock bucket). Token 67 is a `[CLS]` register kept as a pooling slot for position-level probes (currently not supervised in training though). Each square gets its own token so attention maps read directly as square-to-square attention weights 

**Color canonicalization.** Every position is canonicalized to "white to move." in order to halve the state space. Black-to-move positions and move targets are mirrored. 

**Bilinear policy head.** Move logits are computed as:

```
S = source_mlp(h_squares)   # (B, 64, D)
T = target_mlp(h_squares)   # (B, 64, D)
logit[i, j] = <S[i], T[j]>
```

`logit[i, j]` depends *only* on the final residual-stream activations at squares `i` and `j`. If the model has learned to represent a future move, information about it has nowhere to live except the source or target square's residual stream. A small promotion head is layered on top for pawn promotions.

The attention module keeps Q/K/V/O as separate `nn.Linear` layers (not fused), stores attention probabilities on request via a flag, and supports activation patching at the per-head level.

---

## Dataset

Lichess games filtered to both players ≥ 1800 Elo and ≥ 3-minute time control. My dataset used 50,000 games = 3.75M unique positions (3.5M train 0.25M validation). To recreate, run `data/download_data.py` and adjust the parameters accordingly.

---

## Repo structure

```
model.py                  Model definition and policy loss.

data/                     Data pipeline: Lichess PGN download/filtering, 
                          tokenizer functions + tokenizing dataset,
                          training dataloader.

train.ipynb               Training loop, checkpointing, W&B logging.

interp/                   Interpretability helpers: attention visualization, activation patching, etc.

evaluation/               Eval suite: legal-move probabilities, interactive
                          play, game simulation vs stockfish

datasets/                 Preprocessed .npy arrays + meta.json. Committed repo contains a small 1000 game dataset
                          for quick testing
```

---

## Workflow

### Training

```bash
pip install -r requirements.txt
```

1. Download data: `python data/download_data.py` (writes `filtered_games.pgn`)
2. Preprocess: `python data/preprocess.py` (writes `datasets/`)
3. Open and run `train.ipynb` - adjust path to wherever you want to save your model (I saved on google drive)



### Interpretability work

The intended workflow for exploratory analysis is a clean Colab notebook that imports this repo as a library:

## Exploratory Analysis Workflow

The intended workflow for exploratory analysis is a clean Colab notebook that imports this repo as a library.

```python
# in a fresh Colab notebook:
!git clone <repo-url>
%cd transformer-chess
!pip install -r requirements.txt

from model import load_model
import interp.attention as attn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("MODEL_PATH", device)
```

---

## Example Usage

### Visualize attention from a square

```python
import chess

board = chess.Board()

moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6", "O-O", "d6"]
for move in moves:
    board.push_san(move)

attn.show_square_attention(
    model,
    board,
    chess.F3,
    direction="from",
)
```

### Plot a single attention head

```python
attn.plot_single_head_attention(
    model,
    board,
    chess.B5,
    layer=0,
    head=0,
    direction="from",
)
```

### Compare attention across multiple pieces

```python
attn.show_multi_square_attention(
    model,
    board,
    square_to_layer={
        chess.B5: 0,
        chess.F6: 0,
        chess.E5: 0,
        chess.F3: 0,
    },
    direction="from",
)
```

### Show model move suggestions

```python
from evaluation.play import show_suggestions

board = chess.Board()

moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d6"]
for move in moves:
    board.push_san(move)

show_suggestions(model, board, device, k=5)
```

### Play against the model

```python
from evaluation.play import play_game

play_game(
    model,
    device,
    human_color=chess.WHITE,
    temperature=1e-5,
)
```

### Run Stockfish evaluation matches

```python
import chess.engine
import evaluation.stockfish as sfe

engine = chess.engine.SimpleEngine.popen_uci("STOCKFISH_PATH")

results = sfe.run_match(
    model=model,
    stockfish_elo=1400,
    num_games=25,
    engine=engine,
    device=device,
    pgn_path="matches.pgn",
)
```

---
