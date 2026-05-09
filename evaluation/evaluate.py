"""Load a trained checkpoint and run the full chess eval suite.

Hardcode CKPT_PATH and DATA_DIR for Colab and run:
    python run_eval.py
"""
from evaluation.evals import *
from model import *
from data.dataloader import make_loader

# ---------------------------------------------------------------------------
# EDIT FOR COLAB
# ---------------------------------------------------------------------------
CKPT_PATH      = "checkpoints/5M-longer-run.pt"
DATA_DIR       = "datasets/"
N_PREDICT      = 20000             # how many positions to run the model on
STOCKFISH_PATH = None               # e.g. "/usr/games/stockfish"; None to skip
# ---------------------------------------------------------------------------


def evaluate_model(ckpt_path, data_dir, n_predict, stockfish_path=None):
    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    model = load_model(ckpt_path, device)

    # ---- Load val loader ----
    val_loader = make_loader(
        data_dir=data_dir,
        split="val",
        batch_size=128,
        num_workers=2,
    )
    print(f"Val batches: {len(val_loader):,}")

    # ---- Run evals ----
    results = run_full_eval(
        model, val_loader, device,
        n_predict=n_predict,
        stockfish_path=stockfish_path,
    )
    print_eval_results(results)


if __name__ == "__main__":
    evaluate_model(CKPT_PATH, DATA_DIR, N_PREDICT, STOCKFISH_PATH)