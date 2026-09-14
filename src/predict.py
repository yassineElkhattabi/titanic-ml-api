import joblib
import pandas as pd


def load_model(file_path="models/random_forest_pipeline.joblib"):
    model = joblib.load(file_path)
    return model


def predict_passenger(model, passenger_data: dict):
    df = pd.DataFrame([passenger_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = {
        "prediction": int(prediction),
        "survival_probability": float(probability[1]),
        "not_survived_probability": float(probability[0])
    }

    return result