# 🚢 Titanic ML API

A production-oriented Machine Learning API that predicts whether a Titanic passenger would survive based on passenger information.

The project demonstrates an end-to-end Machine Learning workflow:

**Data → Preprocessing → Model Training → Evaluation → Model Persistence → FastAPI → Testing → CI/CD → Cloud Deployment**

## 🌐 Live API

The API is deployed on FastAPI Cloud.

- API: https://titanic-ml-api.fastapicloud.dev
- Swagger Documentation: https://titanic-ml-api.fastapicloud.dev/docs

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** trained on the Titanic dataset.

The preprocessing and model are stored together in a Scikit-learn `Pipeline`.

### Input Features

| Feature | Description |
|---|---|
| `Pclass` | Passenger class (1, 2, or 3) |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation (C, Q, S) |

### Preprocessing

Numerical features:

- Missing values are filled using the median.
- Features are standardized using `StandardScaler`.

Categorical features:

- Missing values are filled using the most frequent value.
- Categories are encoded using `OneHotEncoder`.

## 📊 Model Evaluation

Several classification algorithms were evaluated.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 |
| Decision Tree | 0.8045 | 0.7656 | 0.7101 | 0.7368 |
| Random Forest | **0.8156** | **0.8000** | 0.6957 | **0.7442** |

5-fold stratified cross-validation:

| Model | Mean F1 | Std F1 |
|---|---:|---:|
| Logistic Regression | 0.7258 | 0.0209 |
| Decision Tree | 0.7244 | 0.0360 |
| Random Forest | **0.7549** | 0.0230 |

Random Forest was selected as the final model.

## ⚙️ API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Information

```http
GET /model-info
```

Returns information about the deployed Machine Learning model.

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 25,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 80,
  "Embarked": "S"
}
```

Example response:

```json
{
  "prediction": 1,
  "status": "Survived",
  "survival_probability": 0.97
}
```

## 🏗️ Project Architecture

```text
titanic-ml-api/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── data/
│   └── raw/
│       └── titanic.csv
│
├── models/
│   └── random_forest_pipeline.joblib
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   ├── save_model.py
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── api.py
├── main.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🧪 Automated Tests

The API is tested using `pytest`.

Run locally:

```bash
pytest -v
```

The tests verify:

- API availability
- Model health
- Model information endpoint
- Prediction endpoint

## 🔄 Continuous Integration

GitHub Actions automatically runs the test suite whenever code is pushed to the `main` branch or a pull request targets `main`.

CI workflow:

```text
Git Push
   ↓
GitHub Actions
   ↓
Python 3.13
   ↓
Install Dependencies
   ↓
Run pytest
   ↓
Tests Pass ✅
```

## ☁️ Deployment

The application is deployed on **FastAPI Cloud**.

Every deployment uses the FastAPI entrypoint:

```toml
[tool.fastapi]
entrypoint = "api:app"
```

The project uses Python 3.13 and Scikit-learn 1.6.1 to maintain compatibility with the serialized ML pipeline.

## 🐳 Docker

Build the image:

```bash
docker build -t titanic-ml-api .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name titanic-api-container titanic-ml-api
```

Then open:

```text
http://localhost:8000/docs
```

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- FastAPI
- Pydantic
- Pytest
- Docker
- Git & GitHub
- GitHub Actions
- FastAPI Cloud

## 🎯 Project Goals

This project was created to practice an end-to-end AI Engineering workflow, including:

- Data preprocessing
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model persistence
- REST API development
- Input validation
- Automated testing
- Containerization
- Continuous Integration
- Cloud deployment

## 👤 Author

**Yassine Elkhattabi**

GitHub: `yassineElkhattabi`