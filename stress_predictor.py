import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a small synthetic dataset
data = {
    "study_hours":      [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 6, 7, 5, 4, 9, 10, 8, 1, 2, 9],
    "assignments_due":  [0, 1, 0, 1, 2, 2, 3, 3, 0, 1, 2, 3, 1, 1, 3, 4, 2, 0, 0, 4],
    "sleep_hours":      [9, 8, 8, 7, 7, 6, 6, 5, 8, 7, 6, 5, 7, 7, 5, 4, 6, 9, 8, 4],
    "breaks_taken":     [5, 4, 4, 3, 3, 2, 2, 1, 4, 3, 2, 1, 3, 3, 1, 1, 2, 5, 4, 1],
}

df = pd.DataFrame(data)

def label_stress(row):
    score = 0
    score += row["study_hours"] * 0.4
    score += row["assignments_due"] * 1.0
    score -= row["sleep_hours"] * 0.3
    score -= row["breaks_taken"] * 0.2
    return 1 if score > 2.5 else 0

df["stress_level"] = df.apply(label_stress, axis=1)

df.to_csv("study_stress_data.csv", index=False)

loaded_df = pd.read_csv("study_stress_data.csv")

feature_cols = ["study_hours", "assignments_due", "sleep_hours", "breaks_taken"]
X = loaded_df[feature_cols]
y = loaded_df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_day = pd.DataFrame([{
    "study_hours": 7,
    "assignments_due": 2,
    "sleep_hours": 6,
    "breaks_taken": 2
}])

pred = model.predict(new_day)[0]
print("\nPredicted stress level for new day:", pred)
