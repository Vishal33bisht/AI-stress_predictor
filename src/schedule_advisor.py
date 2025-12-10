import pandas as pd
from pathlib import Path
from joblib import load


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "stress_model.joblib"


def load_model():
    model = load(MODEL_PATH)
    return model


def build_feature_row(
    total_busy_hours,
    num_meetings,
    deep_work_hours,
    commute_hours,
    sleep_hours,
    tasks_due,
    context_switches,
):
    # Derived ratios, same as in data generation
    meeting_load_ratio = num_meetings / (total_busy_hours + 0.5)
    deep_work_ratio = deep_work_hours / (total_busy_hours + 0.5)

    data = {
        "total_busy_hours": [total_busy_hours],
        "num_meetings": [num_meetings],
        "deep_work_hours": [deep_work_hours],
        "commute_hours": [commute_hours],
        "sleep_hours": [sleep_hours],
        "tasks_due": [tasks_due],
        "context_switches": [context_switches],
        "meeting_load_ratio": [meeting_load_ratio],
        "deep_work_ratio": [deep_work_ratio],
    }
    return pd.DataFrame(data)


def explain_recommendations(features, stress_level):
    """
    Very simple rules to give textual suggestions.
    You can improve these later.
    """
    total_busy = features["total_busy_hours"].iloc[0]
    meetings = features["num_meetings"].iloc[0]
    deep_work = features["deep_work_hours"].iloc[0]
    sleep = features["sleep_hours"].iloc[0]
    tasks = features["tasks_due"].iloc[0]
    switches = features["context_switches"].iloc[0]

    suggestions = []

    if stress_level == 0:
        suggestions.append("Day looks manageable. Keep a similar structure.")
        if deep_work < 2:
            suggestions.append("Consider adding at least 1–2 hours of deep work.")
    elif stress_level == 1:
        suggestions.append("Medium stress day. Can be okay, but adjust a bit.")
        if meetings > 4:
            suggestions.append("Try to decline or reschedule 1–2 meetings.")
        if switches > 8:
            suggestions.append("Group similar tasks together to reduce context switching.")
        if sleep < 7:
            suggestions.append("Aim for at least 7 hours of sleep before this day.")
    else:  # high stress
        suggestions.append("High stress day detected. Strongly consider rebalancing.")
        if total_busy > 9:
            suggestions.append("Reduce total busy hours below 8–9 by moving tasks to another day.")
        if tasks > 3:
            suggestions.append("Move non-urgent tasks to another day or delegate.")
        if meetings > 5:
            suggestions.append("Block a no-meeting slot for deep work and recovery.")
        if sleep < 7:
            suggestions.append("Prioritize sleep and do not stack heavy tasks early morning.")
        if switches > 10:
            suggestions.append("Batch tasks into fewer blocks to reduce context switches.")

    return suggestions


def demo():
    model = load_model()

    # Example: Define one sample day manually
    sample_features = build_feature_row(
        total_busy_hours=10,
        num_meetings=5,
        deep_work_hours=1.5,
        commute_hours=1,
        sleep_hours=6,
        tasks_due=4,
        context_switches=12,
    )

    # Predict
    pred = model.predict(sample_features)[0]

    stress_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    print("Predicted stress level:", stress_map.get(pred, pred))
    print("\nRecommendations:")
    for s in explain_recommendations(sample_features, pred):
        print("-", s)


if __name__ == "__main__":
    demo()
