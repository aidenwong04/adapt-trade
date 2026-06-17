import pandas as pd
from pathlib import Path
import mlflow
from adapt_trade.features.pipeline import FeaturePipeline
from adapt_trade.model.learner import OnlineClassifier

CONFIG = {
    "run_name": "hyperparam_tuning",
    "features": "ret_1,cum_ret_10,volatility_10,bar_range,vol_ratio,close_vs_ewm,bar_direction,rsi,macd",
    "rsi": "fadingfactor 0.4",
    "macd": "ewm_fast:0.154, ewm_slow:0.074",
    "grace_period": "450",
    "delta": "1e-7",
    "symbol": "BTCUSDT",
    "interval": "15m",
    "hat_seed": 42,
}

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "btcusdt_15m.csv"
df = pd.read_csv(DATA_PATH)
df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

with mlflow.start_run(run_name=CONFIG['run_name']):
    mlflow.log_params(CONFIG)

    pipeline = FeaturePipeline()
    model = OnlineClassifier()
    prev_features = None
    prev_label = None

    for _, row in df.iterrows():
        features = pipeline.update(row)
        if features is None:
            continue

        if prev_features is not None:
            metrics = model.learn(prev_features, prev_label)
            if model.n_trained % 500 == 0:
                print(f"n={model.n_trained} | "
                      f"HAT acc={metrics['hatc']['accuracy']:.3f} auc={metrics['hatc']['rocauc']:.3f} | "
                      f"LR acc={metrics['lr']['accuracy']:.3f} auc={metrics['lr']['rocauc']:.3f}")

        predictions = model.predict(features)
        prev_features = features
        prev_label = row['label']

    # log final metrics to MLflow
    mlflow.log_metric("hat_accuracy", metrics['hatc']['accuracy'])
    mlflow.log_metric("hat_rocauc", metrics['hatc']['rocauc'])
    mlflow.log_metric("lr_accuracy", metrics['lr']['accuracy'])
    mlflow.log_metric("lr_rocauc", metrics['lr']['rocauc'])

    print(f"\n--- Final Results ---")
    print(f"Total bars trained: {model.n_trained}")
    print(f"HAT  → Accuracy: {metrics['hatc']['accuracy']:.3f} | ROCAUC: {metrics['hatc']['rocauc']:.3f}")
    print(f"LR   → Accuracy: {metrics['lr']['accuracy']:.3f} | ROCAUC: {metrics['lr']['rocauc']:.3f}")