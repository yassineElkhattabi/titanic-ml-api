import joblib
from pathlib import Path


def save_model(model, file_path="models/random_forest_pipeline.joblib"):

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, path)

    print(f"\nModel saved successfully: {path}") 