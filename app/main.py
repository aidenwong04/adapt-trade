from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
import pickle

from adapt_trade.features.pipeline import FeaturePipeline
from adapt_trade.model.learner import OnlineClassifier

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "model.pkl"

def load_state():
    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        return state["pipeline"], state["learner"]
    print("No checkpoint found.")
    return FeaturePipeline(), OnlineClassifier()

pipeline, learner = load_state()

app = FastAPI()

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

@app.get("/")
def read_root():
    return {"message":"welcome :)"}

@app.get("/livez")
def liveness_check():
    return {"status":"ok"}

# want to make an endpoint that takes the current .pkl file (local for now, bt will change to google cloud storage)
# takes the checkpoint and then returns a prediction for the next interval.

@app.post("/predict")
def predict(candle: Candle): # uses pydantic's basemodel to ensure the input is of type Candle
    features = pipeline.update(candle.model_dump()) # converts the candle to a dict
    if features is None:
        raise HTTPException(status_code=400, detail="Pipeline still warming up: not enough bars seen yet.")
    result = learner.predict(features)
    return {
        "hat": {
            "prob_up": round(result["hatc"]["proba"].get(1, 0.5), 4),
            "prediction": result["hatc"]["prediction"],
        },
        "lr": {
            "prob_up": round(result["lr"]["proba"].get(1, 0.5), 4),
            "prediction": result["lr"]["prediction"],
        }
    }



