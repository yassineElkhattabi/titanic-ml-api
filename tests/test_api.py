from fastapi.testclient import TestClient
from api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Titanic Survival Prediction API"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest"
    assert data["task"] == "Binary Classification"
    assert data["target"] == "Survived"


def test_predict():
    passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80,
        "Embarked": "S"
    }

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "status" in data
    assert "survival_probability" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["survival_probability"] <= 1