import pandas as pd
from adapt_trade.features.pipeline import FeaturePipeline
from adapt_trade.model.learner import OnlineClassifier

# 1. Load
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "btcusdt_15m.csv"
df = pd.read_csv(DATA_PATH)

print("Successfully loaded csv.")

# 2. Label: price went up next bar?
df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()  # drop last row (no next bar)

# 3. Instantiate
pipeline = FeaturePipeline()
model = OnlineClassifier()

prev_features = None
prev_label = None

for _,row in df.iterrows():
    # update 
    features = pipeline.update(row)
    if features is None:
        continue
    
    if prev_features is not None:
        # want to learn from the previous bar 
        metrics = model.learn(prev_features, prev_label)
        if model.n_trained % 500 == 0 and model.n_trained > 0:
            print(f"n={model.n_trained} | "
                f"HAT acc={metrics['hatc']['accuracy']:.3f} auc={metrics['hatc']['rocauc']:.3f} | "
                f"LR acc={metrics['lr']['accuracy']:.3f} auc={metrics['lr']['rocauc']:.3f}")
        
    predictions = model.predict(features) #predict using what u learned from prevoius bar

    prev_features = features
    prev_label = row['label']

print("\n--- Final Results ---")
print(f"Total bars trained: {model.n_trained}")
print(f"HAT  → Accuracy: {metrics['hatc']['accuracy']:.3f} | ROCAUC: {metrics['hatc']['rocauc']:.3f}")
print(f"LR   → Accuracy: {metrics['lr']['accuracy']:.3f} | ROCAUC: {metrics['lr']['rocauc']:.3f}")