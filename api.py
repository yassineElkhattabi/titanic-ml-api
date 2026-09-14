from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from src.predict import load_model, predict_passenger


app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0"
)

model = load_model()


class PassengerInput(BaseModel):
    Pclass: int = Field(ge=1, le=3)

    Sex: Literal["male", "female"]

    Age: float = Field(ge=0, le=120)

    SibSp: int = Field(ge=0)

    Parch: int = Field(ge=0)

    Fare: float = Field(ge=0)

    Embarked: Literal["C", "Q", "S"]


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    survival_probability: float


@app.get("/")
def home():
    return {
        "message": "Titanic ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: PassengerInput):

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