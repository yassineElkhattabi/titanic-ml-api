from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src.predict import load_model, predict_passenger


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Machine Learning API for predicting Titanic passenger survival.",
    version="1.0.0"
)


try:
    model = load_model()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


class PassengerInput(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]

    Age: float = Field(
        ge=0,
        le=120,
        description="Passenger age"
    )

    SibSp: int = Field(
        ge=0,
        description="Number of siblings/spouses aboard"
    )

    Parch: int = Field(
        ge=0,
        description="Number of parents/children aboard"
    )

    Fare: float = Field(
        ge=0,
        description="Passenger fare"
    )

    Embarked: Literal["C", "Q", "S"]


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    survival_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API",
        "docs": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse
)
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest",
        "task": "Binary Classification",
        "target": "Survived",
        "features": [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: PassengerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not available."
        )

    try:
        passenger = {
            "Pclass": data.Pclass,
            "Sex": data.Sex,
            "Age": data.Age,
            "SibSp": data.SibSp,
            "Parch": data.Parch,
            "Fare": data.Fare,
            "Embarked": data.Embarked
        }

        result = predict_passenger(
            model,
            passenger
        )

        prediction = result["prediction"]

        return {
            "prediction": prediction,
            "status": (
                "Survived"
                if prediction == 1
                else "Not Survived"
            ),
            "survival_probability": round(
                result["survival_probability"],
                4
            )
        }

    except Exception as e:
        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )