# adapt-trade

A production-grade online learning system for 15-minute BTC/ETH price direction
prediction. The model continuously adapts to market regime changes using
[River](https://github.com/online-ml/river)'s Hoeffding Adaptive Tree classifier,
with real-time drift detection (ADWIN/KSWIN), a FastAPI serving layer, MLflow
checkpointing, and Prometheus observability — deployed on Google Cloud Run.

## Architecture

┌─────────────────────────────────────────────────────┐
│ Docker Container (Cloud Run) │
│ │
│ Background Learning Loop │
│ Binance WebSocket → Feature Pipeline │
│ → Hoeffding Adaptive Tree │
│ → ADWIN/KSWIN Drift Detection │
│ → MLflow Checkpoint │
│ │ │
│ Shared Model State │
│ │ │
│ FastAPI Serving Loop │
│ POST /predict GET /metrics │
│ GET /health GET /model-info │
└─────────────────────────────────────────────────────┘

## Setup

```bash
git clone https://github.com/aidenwong04/adapt-trade
cd adapt-trade
pip install -e ".[dev]"
```

## Usage

Run the ingestion smoke test (waits for a closed 15-minute candle):

```bash
python test/test_ingestion.py
```

Start the full system (once serving layer is implemented):

```bash
uvicorn src.serving.app:app --reload
```

## Project Status

| Week | Component                            | Status  |
| ---- | ------------------------------------ | ------- |
| 1    | Binance WebSocket ingestion          | ✅ Done |
| 2    | Feature pipeline + prequential loop  | 🔲      |
| 3    | FastAPI serving layer + Docker       | 🔲      |
| 4    | Drift detection + Prometheus metrics | 🔲      |
| 5    | MLflow checkpointing + model-info    | 🔲      |
| 6    | Cloud Run deployment + CI/CD         | 🔲      |
| 7    | Monitoring dashboard                 | 🔲      |

## Tech Stack

- **Online model:** [River](https://github.com/online-ml/river) — `HoeffdingAdaptiveTreeClassifier`
- **Drift detection:** ADWIN (error rate), KSWIN (feature distributions)
- **Serving:** FastAPI + Uvicorn
- **Experiment tracking:** MLflow
- **Observability:** Prometheus client
- **Deployment:** Docker → Google Cloud Run

## Evaluation

The model is evaluated using prequential (test-then-train) methodology — each
example is predicted before the model learns from it, giving an unbiased estimate
of online generalization error. Primary metric: rolling 500-example accuracy
(target: >52%).
