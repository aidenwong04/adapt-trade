import asyncio
import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path

from adapt_trade.ingestion.ws_client import stream_ohlcv
from adapt_trade.features.pipeline import FeaturePipeline
from adapt_trade.model.learner import OnlineClassifier

# --- Paths ---
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pkl"
LOG_PATH = LOG_DIR / "predictions.csv"
CHECKPOINT_EVERY = 100

CSV_HEADERS = [
    "timestamp", "symbol", "close",
    "hat_prob_up", "hat_predicted_label",
    "lr_prob_up", "lr_predicted_label",
    "actual_label",
]

def load_state():
    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        return state["pipeline"], state["learner"]
    print("No checkpoint found. Starting fresh.")
    return FeaturePipeline(), OnlineClassifier()


def save_state(pipeline, learner):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "learner": learner}, f)


def init_csv():
    LOG_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def append_row(row: dict):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)


async def main():
    pipeline, learner = load_state()
    init_csv()

    q = asyncio.Queue()
    asyncio.create_task(stream_ohlcv(["BTCUSDT"], q))

    first_kline = await q.get()
    prev_features = pipeline.update(first_kline)
    prev_kline = first_kline
    prev_row = None

    try:
        while True:
            current_kline = await q.get()
            print(current_kline)

            prev_close = prev_kline["close"]
            curr_close = current_kline["close"]
            label = 1 if curr_close > prev_close else 0

            # Write previous buffered row now that actual_label is known
            if prev_row is not None:
                prev_row["actual_label"] = label
                append_row(prev_row)

            # Learn from previous bar
            if prev_features is not None:
                metrics = learner.learn(prev_features, label)
                if learner.n_trained % 500 == 0:
                    print(
                        f"n={learner.n_trained} | "
                        f"HAT acc={metrics['hatc']['accuracy']:.3f} "
                        f"auc={metrics['hatc']['rocauc']:.3f} | "
                        f"LR acc={metrics['lr']['accuracy']:.3f} "
                        f"auc={metrics['lr']['rocauc']:.3f}"
                    )

            # Predict on current bar
            curr_features = pipeline.update(current_kline)
            if curr_features is not None:
                result = learner.predict(curr_features)

                hat_prob_up = result["hatc"]["proba"].get(1, 0.5)
                hat_predicted_label = result["hatc"]["prediction"]

                lr_prob_up = result["lr"]["proba"].get(1, 0.5)
                lr_predicted_label = result["lr"]["prediction"]

                ts = datetime.fromtimestamp(
                    current_kline["ts"] / 1000, tz=timezone.utc
                ).isoformat()

                prev_row = {
                    "timestamp": ts,
                    "symbol": current_kline["symbol"],
                    "close": curr_close,
                    "hat_prob_up": round(hat_prob_up, 4),
                    "hat_predicted_label": hat_predicted_label,
                    "lr_prob_up": round(lr_prob_up, 4),
                    "lr_predicted_label": lr_predicted_label,
                    "actual_label": None,
                }
            else:
                prev_row = None

            # Checkpoint
            if learner.n_trained > 0 and learner.n_trained % CHECKPOINT_EVERY == 0:
                save_state(pipeline, learner)
                print(f"Checkpoint saved at n={learner.n_trained}")

            prev_kline = current_kline
            prev_features = curr_features

    except Exception as e:
        print(f"Learning loop crashed: {e}")
        save_state(pipeline, learner)
        raise


if __name__ == "__main__":
    asyncio.run(main())
