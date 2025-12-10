import numpy as np
import pandas as pd
from pathlib import Path

# Where to save the generated dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "schedule_days.csv"


def generate_day():
    """
    Generate one synthetic 'day' with schedule-related features.
    """

    # Total time in hours
    total_busy_hours = np.random.uniform(0, 12)     # 0 to 12 hours of "busy"
    num_meetings = np.random.randint(0, 8)          # 0–7 meetings
    deep_work_hours = np.random.uniform(0, 6)       # focus time
    commute_hours = np.random.uniform(0, 3)         # 0–3 hours commute
    sleep_hours = np.random.uniform(4, 9)           # 4–9 hours of sleep
    tasks_due = np.random.randint(0, 6)             # 0–5 tasks with deadlines
    context_switches = np.random.randint(0, 15)     # number of times switched between tasks

    # Derived features
    meeting_load_ratio = num_meetings / (total_busy_hours + 0.5)  # avoid divide by 0
    deep_work_ratio = deep_work_hours / (total_busy_hours + 0.5)

    # Simple "hidden" stress score (this is our rule to create labels)
    score = 0
    score += total_busy_hours * 0.4
    score += num_meetings * 0.5
    score += commute_hours * 0.6
    score += tasks_due * 0.8
    score += context_switches * 0.15

    score -= sleep_hours * 0.7
    score -= deep_work_hours * 0.4

    # Convert continuous score into 0/1 stress label
    # You can tune thresholds later
    if score < 3:
        stress_level = 0  # low
    elif score < 7:
        stress_level = 1  # medium (we can merge later if needed)
    else:
        stress_level = 2  # high

    return {
        "total_busy_hours": round(total_busy_hours, 2),
        "num_meetings": num_meetings,
        "deep_work_hours": round(deep_work_hours, 2),
        "commute_hours": round(commute_hours, 2),
        "sleep_hours": round(sleep_hours, 2),
        "tasks_due": tasks_due,
        "context_switches": context_switches,
        "meeting_load_ratio": round(meeting_load_ratio, 3),
        "deep_work_ratio": round(deep_work_ratio, 3),
        "stress_level": stress_level,
    }


def generate_dataset(n_days: int = 400):
    rows = [generate_day() for _ in range(n_days)]
    df = pd.DataFrame(rows)
    return df


def main():
    df = generate_dataset(400)
    print("Sample of generated data:")
    print(df.head())

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
