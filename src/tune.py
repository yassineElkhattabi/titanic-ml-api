from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.preprocessing import build_preprocessor


def tune_random_forest(X_train, y_train):

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),

        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n--- HYPERPARAMETER TUNING ---")

    print("Best parameters:")
    print(grid_search.best_params_)

    print(f"\nBest CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_