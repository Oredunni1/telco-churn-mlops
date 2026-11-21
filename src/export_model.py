"""Export the winning MLflow model to a portable joblib pickle for HF Spaces.

Design Decision — joblib pickle vs. `.skops`:
`.skops` is safer on deserialization and is used natively by MLflow 3.x for
trusted serialization, whereas pickle/joblib can execute arbitrary code if
loading an untrusted file. For this self-contained portfolio deployment, we
export a `joblib` pickle (`models/logreg_balanced.pkl`) because it is
concise, fast, and loads a model artifact we generated ourselves—allowing the
Hugging Face Space runtime to depend only on `scikit-learn`, `pandas`, and
`gradio` without pulling in MLflow or SQLite.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from src import data

RUN_ID = os.getenv("MODEL_RUN_ID", "b1d20639f6fd4e6c83da0053f9a538ac")
MLFLOW_DB_URI = f"sqlite:///{data.BASE_DIR}/mlflow.db"
OUTPUT_PATH = Path(data.BASE_DIR) / "models" / "logreg_balanced.pkl"


def export_model(
    run_id: str = RUN_ID,
    tracking_uri: str = MLFLOW_DB_URI,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Load the trained model from MLflow and save it via joblib."""
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, protocol=4)
    return output_path


if __name__ == "__main__":
    saved_path = export_model()
    print(f"Exported model to {saved_path}")