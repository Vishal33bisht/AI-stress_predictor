import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "schedule_days.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "stress_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Features and label
    feature_cols = [
        "total_busy_hours",
        "num_meetings",
        "deep_work_hours",
        "commute_hours",
        "sleep_hours",
        "tasks_due",
        "context_switches",
        "meeting_load_ratio",
        "deep_work_ratio",
    ]
    X = df[feature_cols]
    y = df["stress_level"]  # 0 = low, 1 = medium, 2 = high

    return X, y


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
