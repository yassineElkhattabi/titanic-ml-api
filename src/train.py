from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from src.preprocessing import build_preprocessor


def train_models(df):
    # Features / Target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # نفس Train/Test لجميع النماذج
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f"{name} trained successfully.")

    return trained_models, X_train, X_test, y_train, y_test 

def cross_validate_models(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1"
        )

        results.append({
            "Model": name,
            "Mean F1": scores.mean(),
            "Std F1": scores.std()
        })

    return results