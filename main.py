import pandas as pd

from src.data_loader import load_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.train import train_models, cross_validate_models 
from src.tune import tune_random_forest 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from src.save_model import save_model 
from src.predict import load_model, predict_passenger 
def main():

    df = load_data("data/raw/titanic.csv")

    models, X_train, X_test, y_train, y_test = train_models(df)

    results = evaluate_models(
        models,
        X_test,
        y_test
    )

    results_df = pd.DataFrame(results)

    print("\n--- MODEL COMPARISON ---")
    print(results_df.round(4).to_string(index=False))
    cv_results = cross_validate_models(df)

    cv_df = pd.DataFrame(cv_results)

    print("\n--- CROSS-VALIDATION RESULTS ---")
    print(cv_df.round(4).to_string(index=False))
    best_rf = tune_random_forest(X_train, y_train) 
    y_pred = best_rf.predict(X_test)
    print("\n--- TUNED RANDOM FOREST TEST RESULTS ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    final_model = models["Random Forest"]
    save_model(final_model)
    loaded_model = load_model()

    passenger = {
        "PassengerId": 1000,
        "Pclass": 1,
        "Name": "Test Passenger",
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST123",
        "Fare": 80,
        "Cabin": None,
        "Embarked": "S"
    }

    result = predict_passenger(
        loaded_model,
        passenger
    )

    print("\n--- NEW PASSENGER PREDICTION ---")
    print(result)
if __name__ == "__main__":
    main()