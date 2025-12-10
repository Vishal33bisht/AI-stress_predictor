import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stress_model.joblib"
USER_DATA_PATH = BASE_DIR / "data" / "user_days.csv"


# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    model = load(MODEL_PATH)
    return model


model = load_model()


# ---------- User Data Helpers ----------
@st.cache_data
def load_user_days():
    if USER_DATA_PATH.exists():
        return pd.read_csv(USER_DATA_PATH)
    else:
        return pd.DataFrame()


def append_user_day(row: dict):
    df_existing = load_user_days()
    df_new = pd.DataFrame([row])
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    USER_DATA_PATH.parent.mkdir(exist_ok=True)
    df_all.to_csv(USER_DATA_PATH, index=False)
    return df_all


# ---------- Feature Builder ----------
def build_feature_row(
    total_busy_hours,
    num_meetings,
    deep_work_hours,
    commute_hours,
    sleep_hours,
    tasks_due,
    context_switches,
):
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


def stress_int_to_label(pred: int) -> str:
    stress_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return stress_map.get(pred, str(pred))


# ---------- Recommendation Logic (Improved) ----------
def explain_recommendations(features, stress_level):
    total_busy = features["total_busy_hours"].iloc[0]
    meetings = features["num_meetings"].iloc[0]
    deep_work = features["deep_work_hours"].iloc[0]
    sleep = features["sleep_hours"].iloc[0]
    tasks = features["tasks_due"].iloc[0]
    switches = features["context_switches"].iloc[0]
    commute = features["commute_hours"].iloc[0]

    suggestions = []

    # Generic hygiene tips
    if sleep < 6:
        suggestions.append("Your sleep is very low. Try to get at least 6–7 hours tonight.")
    elif sleep < 7:
        suggestions.append("Sleep is slightly low. Aim for 7–8 hours for better recovery.")

    if switches > 8:
        suggestions.append("You have high context switching. Try batching similar tasks together.")

    if stress_level == 0:
        suggestions.append("Day looks manageable. Keep a similar structure.")
        if deep_work < 2:
            suggestions.append("Consider adding at least 1–2 hours of deep work for important tasks.")
        if total_busy > 9:
            suggestions.append("Even on low-stress days, try not to exceed 9 busy hours regularly.")

    elif stress_level == 1:
        suggestions.append("Medium stress day. Can be okay, but some tweaks will help.")
        if meetings > 4:
            suggestions.append("Try to decline, shorten, or reschedule 1–2 non-critical meetings/classes.")
        if tasks > 3:
            suggestions.append("Too many tasks with deadlines. Push what is not urgent to another day.")
        if commute > 2:
            suggestions.append("Commute is long. Consider WFH options or combining errands to reduce trips.")

    else:  # high stress
        suggestions.append("High stress day detected. Strongly consider rebalancing.")
        if total_busy > 9:
            suggestions.append("Reduce total busy hours below ~8–9 by moving tasks to another day.")
        if tasks > 3:
            suggestions.append("Move non-urgent tasks to another day or delegate if possible.")
        if meetings > 5:
            suggestions.append("Limit total meetings/classes and block some focus time.")
        if commute > 2:
            suggestions.append("Try to avoid scheduling heavy work immediately after a long commute.")
        if switches > 10:
            suggestions.append("Batch tasks into fewer, longer blocks to reduce mental load.")

    return suggestions


# ---------- What-If Analyzer ----------
def compute_what_if_scenarios(
    total_busy_hours,
    num_meetings,
    deep_work_hours,
    commute_hours,
    sleep_hours,
    tasks_due,
    context_switches,
    model,
):
    scenarios = []

    def predict_variant(tb, nm, dw, ch, sh, td, cs, label: str):
        df = build_feature_row(tb, nm, dw, ch, sh, td, cs)
        pred = model.predict(df)[0]
        return {
            "Scenario": label,
            "total_busy_hours": tb,
            "num_meetings": nm,
            "deep_work_hours": dw,
            "commute_hours": ch,
            "sleep_hours": sh,
            "tasks_due": td,
            "context_switches": cs,
            "Predicted Stress": stress_int_to_label(pred),
        }

    # Baseline
    scenarios.append(
        predict_variant(
            total_busy_hours,
            num_meetings,
            deep_work_hours,
            commute_hours,
            sleep_hours,
            tasks_due,
            context_switches,
            "Current day",
        )
    )

    # Sleep +1 hour (cap at 10)
    sh_plus = min(sleep_hours + 1.0, 10.0)
    scenarios.append(
        predict_variant(
            total_busy_hours,
            num_meetings,
            deep_work_hours,
            commute_hours,
            sh_plus,
            tasks_due,
            context_switches,
            "Sleep +1 hour",
        )
    )

    # Meetings -2 (min 0)
    nm_minus = max(num_meetings - 2, 0)
    scenarios.append(
        predict_variant(
            total_busy_hours,
            nm_minus,
            deep_work_hours,
            commute_hours,
            sleep_hours,
            tasks_due,
            context_switches,
            "Meetings/classes -2",
        )
    )


    # Deep work +1 hour (cap at 6, cannot exceed total busy)
    dw_plus = min(deep_work_hours + 1.0, 6.0, total_busy_hours)
    scenarios.append(
        predict_variant(
            total_busy_hours,
            num_meetings,
            dw_plus,
            commute_hours,
            sleep_hours,
            tasks_due,
            context_switches,
            "Deep work/self-study +1 hour",
        )
    )

    # Context switches -4 (min 0)
    cs_minus = max(context_switches - 4, 0)
    scenarios.append(
        predict_variant(
            total_busy_hours,
            num_meetings,
            deep_work_hours,
            commute_hours,
            sleep_hours,
            tasks_due,
            cs_minus,
            "Context switches -4",
        )
    )

    # Tasks_due -2 (min 0)
    td_minus = max(tasks_due - 2, 0)
    scenarios.append(
        predict_variant(
            total_busy_hours,
            num_meetings,
            deep_work_hours,
            commute_hours,
            sleep_hours,
            td_minus,
            context_switches,
            "Deadlined tasks -2",
        )
    )

    return pd.DataFrame(scenarios)


def suggest_best_change(scenarios_df: pd.DataFrame):
    """
    From the what-if scenarios, pick the best one (lowest stress)
    and generate human-readable suggestions compared to current day.
    """
    if scenarios_df is None or scenarios_df.empty:
        return None

    # Map stress labels to numeric ranking
    stress_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df = scenarios_df.copy()
    df["rank"] = df["Predicted Stress"].map(stress_rank)

    # Get current day row
    current_row = df[df["Scenario"] == "Current day"].iloc[0]

    # Pick the scenario with the best (lowest) stress
    best_row = df.sort_values("rank").iloc[0]

    # If best scenario is already the current one, no improvement
    if best_row["Scenario"] == "Current day":
        return {
            "summary": "Your current plan is already the best among tested scenarios.",
            "changes": [],
            "from_stress": current_row["Predicted Stress"],
            "to_stress": current_row["Predicted Stress"],
        }

    changes = []
    # Check key fields for differences
    fields = [
        ("sleep_hours", "Sleep hours"),
        ("num_meetings", "Number of meetings/classes"),
        ("tasks_due", "Tasks with deadlines"),
        ("deep_work_hours", "Deep work / focus hours"),
        ("context_switches", "Context switches"),
        ("total_busy_hours", "Total busy hours"),
    ]

    for col, label in fields:
        if best_row[col] != current_row[col]:
            changes.append(f"{label}: {current_row[col]} → {best_row[col]}")

    summary = (
        f"To move stress from {current_row['Predicted Stress']} "
        f"to {best_row['Predicted Stress']}, change the following:"
    )

    return {
        "summary": summary,
        "changes": changes,
        "from_stress": current_row["Predicted Stress"],
        "to_stress": best_row["Predicted Stress"],
    }

def auto_reschedule_day(
    total_busy_hours,
    num_meetings,
    deep_work_hours,
    commute_hours,
    sleep_hours,
    tasks_due,
    context_switches,
    model,
):
    """
    Search around the current configuration for a 'better' day
    (lower predicted stress) with reasonably small changes.
    Returns the best alternative configuration and its stress.
    """

    # Map stress labels to numeric ranking (for optimization)
    stress_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    # Helper to evaluate a config
    def eval_config(tb, nm, dw, ch, sh, td, cs):
        df = build_feature_row(tb, nm, dw, ch, sh, td, cs)
        pred_int = model.predict(df)[0]
        pred_label = stress_int_to_label(pred_int)

        # Cost: primarily based on stress level, then how big the changes are
        rank = stress_rank[pred_label]

        change_magnitude = (
            abs(tb - total_busy_hours) * 1.0
            + abs(nm - num_meetings) * 1.0
            + abs(dw - deep_work_hours) * 0.8
            + abs(ch - commute_hours) * 0.5
            + abs(sh - sleep_hours) * 0.8
            + abs(td - tasks_due) * 1.0
            + abs(cs - context_switches) * 0.5
        )

        cost = rank * 100 + change_magnitude
        return {
            "config": {
                "total_busy_hours": tb,
                "num_meetings": nm,
                "deep_work_hours": dw,
                "commute_hours": ch,
                "sleep_hours": sh,
                "tasks_due": td,
                "context_switches": cs,
            },
            "stress_int": pred_int,
            "stress_label": pred_label,
            "cost": cost,
        }

    # Candidate values around current day
    sleep_candidates = sorted(
        {
            max(3.0, sleep_hours),
            min(7.0, 10.0),
            min(8.0, 10.0),
        }
    )

    meetings_candidates = sorted(
        {max(num_meetings - 3, 0), max(num_meetings - 2, 0), num_meetings}
    )

    tasks_candidates = sorted(
        {max(tasks_due - 3, 0), max(tasks_due - 2, 0), tasks_due}
    )

    busy_candidates = sorted(
        {
            min(total_busy_hours, 9.0),
            min(total_busy_hours, 8.0),
            total_busy_hours,
        }
    )

    deep_candidates = sorted(
        {
            min(max(deep_work_hours, 2.0), total_busy_hours),
            deep_work_hours,
        }
    )

    switches_candidates = sorted(
        {
            max(context_switches - 8, 0),
            max(context_switches - 4, 0),
            context_switches,
        }
    )

    best = None

    for tb in busy_candidates:
        for nm in meetings_candidates:
            for td in tasks_candidates:
                for dw in deep_candidates:
                    # Ensure deep work cannot exceed total busy hours
                    dw_eff = min(dw, tb)
                    for sh in sleep_candidates:
                        for cs in switches_candidates:
                            res = eval_config(tb, nm, dw_eff, commute_hours, sh, td, cs)
                            if best is None or res["cost"] < best["cost"]:
                                best = res

    return best


def describe_reschedule_changes(current_cfg: dict, proposed_cfg: dict):
    """
    Produce human-readable change description between current and proposed config.
    """
    changes = []

    def add_change(label, key, unit=""):
        cur = current_cfg[key]
        new = proposed_cfg[key]
        if cur != new:
            changes.append(f"{label}: {cur}{unit} → {new}{unit}")

    add_change("Sleep hours", "sleep_hours", " h")
    add_change("Total busy hours", "total_busy_hours", " h")
    add_change("Meetings / classes / calls", "num_meetings")
    add_change("Deep work / study hours", "deep_work_hours", " h")
    add_change("Tasks with deadlines", "tasks_due")
    add_change("Context switches", "context_switches")

    return changes

# ---------- Profile-Based Templates ----------
PROFILE_TEMPLATES = {
    "Office worker": {
        "None (manual)": None,
        "Meeting-heavy day": {
            "total_busy_hours": 10.0,
            "num_meetings": 7,
            "deep_work_hours": 1.0,
            "commute_hours": 1.0,
            "sleep_hours": 6.0,
            "tasks_due": 3,
            "context_switches": 14,
        },
        "Deep work day": {
            "total_busy_hours": 8.0,
            "num_meetings": 2,
            "deep_work_hours": 4.0,
            "commute_hours": 0.5,
            "sleep_hours": 7.5,
            "tasks_due": 2,
            "context_switches": 6,
        },
        "Weekend errands day": {
            "total_busy_hours": 6.0,
            "num_meetings": 1,
            "deep_work_hours": 1.5,
            "commute_hours": 2.0,
            "sleep_hours": 8.0,
            "tasks_due": 3,
            "context_switches": 5,
        },
    },
    "Student": {
        "None (manual)": None,
        "Regular college day": {
            "total_busy_hours": 7.0,
            "num_meetings": 5,   # lectures/classes
            "deep_work_hours": 2.0,  # self-study
            "commute_hours": 1.0,
            "sleep_hours": 7.5,
            "tasks_due": 2,      # assignments/quizzes
            "context_switches": 8,
        },
        "Exam preparation day": {
            "total_busy_hours": 9.0,
            "num_meetings": 1,   # maybe 1 class/coaching
            "deep_work_hours": 5.0,
            "commute_hours": 1.0,
            "sleep_hours": 6.0,
            "tasks_due": 3,
            "context_switches": 7,
        },
        "Exam day": {
            "total_busy_hours": 8.0,
            "num_meetings": 2,   # exams + maybe class
            "deep_work_hours": 3.0,
            "commute_hours": 1.5,
            "sleep_hours": 5.5,
            "tasks_due": 4,
            "context_switches": 6,
        },
    },
    "Other": {
        "None (manual)": None,
        "Freelance client day": {
            "total_busy_hours": 9.0,
            "num_meetings": 3,   # client calls
            "deep_work_hours": 4.0,
            "commute_hours": 0.5,
            "sleep_hours": 7.0,
            "tasks_due": 3,
            "context_switches": 10,
        },
        "Light admin/chores day": {
            "total_busy_hours": 5.0,
            "num_meetings": 1,
            "deep_work_hours": 1.5,
            "commute_hours": 1.0,
            "sleep_hours": 8.0,
            "tasks_due": 2,
            "context_switches": 5,
        },
    },
}


def init_default_state():
    defaults = {
        "total_busy_hours": 8.0,
        "num_meetings": 4,
        "deep_work_hours": 2.0,
        "commute_hours": 1.0,
        "sleep_hours": 7.0,
        "tasks_due": 3,
        "context_switches": 8,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def apply_template(templates: dict, name: str):
    template = templates.get(name)
    if template is None:
        return
    for k, v in template.items():
        st.session_state[k] = v


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Personal Schedule Load Balancer", layout="wide")

st.title("🧠 Personal Schedule Load Balancer")
st.write(
    "Adjust your day’s schedule below. The model will estimate your stress level "
    "and suggest how to rebalance the day. You can also test 'what-if' scenarios, "
    "use templates, and visualize your time."
)

# Initialize state for sliders
init_default_state()

# ----- Profile Selection -----
st.sidebar.header("Profile")
profile = st.sidebar.selectbox(
    "I am a...", ["Office worker", "Student", "Other"]
)

# Dynamic labels based on profile
if profile == "Student":
    meetings_label = "Number of classes / lectures"
    tasks_label = "Assignments / exams / projects today"
elif profile == "Other":
    meetings_label = "Number of calls / appointments"
    tasks_label = "Important tasks with deadlines today"
else:  # Office worker
    meetings_label = "Number of meetings"
    tasks_label = "Number of tasks with deadlines today"

# ----- Templates -----
st.sidebar.header("Day Configuration")

current_templates = PROFILE_TEMPLATES.get(profile, {"None (manual)": None})

template_choice = st.sidebar.selectbox(
    "Day template", list(current_templates.keys())
)

if st.sidebar.button("Apply template"):
    apply_template(current_templates, template_choice)
    st.sidebar.success(f"Applied template: {template_choice}")

# ----- Sliders (bound to session_state) -----
total_busy_hours = st.sidebar.slider(
    "Total busy hours (meetings/classes + tasks)",
    min_value=0.0,
    max_value=12.0,
    value=float(st.session_state["total_busy_hours"]),
    step=0.5,
    key="total_busy_hours",
)
num_meetings = st.sidebar.slider(
    meetings_label,
    min_value=0,
    max_value=10,
    value=int(st.session_state["num_meetings"]),
    step=1,
    key="num_meetings",
)
deep_work_hours = st.sidebar.slider(
    "Deep work / focus hours (self-study / focused work)",
    min_value=0.0,
    max_value=6.0,
    value=float(st.session_state["deep_work_hours"]),
    step=0.5,
    key="deep_work_hours",
)
commute_hours = st.sidebar.slider(
    "Commute hours",
    min_value=0.0,
    max_value=3.0,
    value=float(st.session_state["commute_hours"]),
    step=0.5,
    key="commute_hours",
)
sleep_hours = st.sidebar.slider(
    "Sleep hours (previous night)",
    min_value=3.0,
    max_value=10.0,
    value=float(st.session_state["sleep_hours"]),
    step=0.5,
    key="sleep_hours",
)
tasks_due = st.sidebar.slider(
    tasks_label,
    min_value=0,
    max_value=10,
    value=int(st.session_state["tasks_due"]),
    step=1,
    key="tasks_due",
)
context_switches = st.sidebar.slider(
    "Approx. context switches (changing tasks)",
    min_value=0,
    max_value=20,
    value=int(st.session_state["context_switches"]),
    step=1,
    key="context_switches",
)

analyze_clicked = st.button("Analyze My Day")

# Initialize variables so they exist even if button not clicked
scenarios_df = None
best_plan = None
stress_label = None

if analyze_clicked:
    # Build features and predict
    features = build_feature_row(
        total_busy_hours=total_busy_hours,
        num_meetings=num_meetings,
        deep_work_hours=deep_work_hours,
        commute_hours=commute_hours,
        sleep_hours=sleep_hours,
        tasks_due=tasks_due,
        context_switches=context_switches,
    )

    pred_int = model.predict(features)[0]
    stress_label = stress_int_to_label(pred_int)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predicted Stress Level")
        if pred_int == 0:
            st.success(f"Stress Level: {stress_label}")
        elif pred_int == 1:
            st.warning(f"Stress Level: {stress_label}")
        else:
            st.error(f"Stress Level: {stress_label}")

        st.subheader("Recommendations to Rebalance Your Day")
        recs = explain_recommendations(features, pred_int)
        for r in recs:
            st.markdown(f"- {r}")

    # ----- Visual Charts -----
    with col2:
        st.subheader("Time Allocation (Hours)")
        time_df = pd.DataFrame(
            {
                "Category": ["Busy", "Deep work", "Commute", "Sleep"],
                "Hours": [
                    total_busy_hours,
                    deep_work_hours,
                    commute_hours,
                    sleep_hours,
                ],
            }
        )
        st.bar_chart(time_df.set_index("Category"))

        st.subheader("Raw Feature Snapshot")
        st.write(features)

    # ----- What-If Analyzer -----
    st.markdown("---")
    st.subheader("🔍 What-If Analyzer")
    st.write("See how small changes could affect your stress level:")

    scenarios_df = compute_what_if_scenarios(
        total_busy_hours,
        num_meetings,
        deep_work_hours,
        commute_hours,
        sleep_hours,
        tasks_due,
        context_switches,
        model,
    )
    st.table(scenarios_df[["Scenario", "Predicted Stress"]])

    # Compute best plan only when scenarios_df exists
    best_plan = suggest_best_change(scenarios_df)

if best_plan is not None:
    st.markdown("### Recommended Adjustment Plan")
    if best_plan["from_stress"] == best_plan["to_stress"]:
        st.info(best_plan["summary"])
    else:
        st.write(best_plan["summary"])
        if best_plan["changes"]:
            for c in best_plan["changes"]:
                st.markdown(f"- {c}")
        else:
            st.write("No specific parameter changes found.")

    # ----- Save Day -----
    st.markdown("---")
    st.subheader("Save this day to history")

    day_label = st.text_input("Give this day a label (e.g., 'Monday sprint', 'Exam day')")
    if st.button("Save Day"):
        if day_label.strip() == "":
            st.warning("Please enter a label before saving.")
        else:
            row_to_save = {
                "label": day_label,
                "profile": profile,
                "total_busy_hours": total_busy_hours,
                "num_meetings": num_meetings,
                "deep_work_hours": deep_work_hours,
                "commute_hours": commute_hours,
                "sleep_hours": sleep_hours,
                "tasks_due": tasks_due,
                "context_switches": context_switches,
                "predicted_stress_level": stress_label,
            }
            df_all = append_user_day(row_to_save)
            st.success(f"Saved day '{day_label}'.")
            st.write("Current saved days:")
            st.dataframe(df_all)

else:
    st.info("Set your schedule parameters in the sidebar, then click 'Analyze My Day'.")

# ----- Auto-Rescheduler -----
st.markdown("---")
st.subheader("🤖 AI Auto-Rescheduler (Experimental)")

current_cfg = {
    "total_busy_hours": total_busy_hours,
    "num_meetings": num_meetings,
    "deep_work_hours": deep_work_hours,
    "commute_hours": commute_hours,
    "sleep_hours": sleep_hours,
    "tasks_due": tasks_due,
    "context_switches": context_switches,
}

auto_plan = auto_reschedule_day(
    total_busy_hours,
    num_meetings,
    deep_work_hours,
    commute_hours,
    sleep_hours,
    tasks_due,
    context_switches,
    model,
)

# Auto-reschedule should only run after "Analyze My Day" is clicked
if analyze_clicked:
    # Use the current configuration as input to the auto-rescheduler
    auto_plan = auto_reschedule_day(
        total_busy_hours,
        num_meetings,
        deep_work_hours,
        commute_hours,
        sleep_hours,
        tasks_due,
        context_switches,
        model,
    )

    if auto_plan is not None:
        # pred_int is defined in the earlier analyze_clicked block
        current_stress_label = stress_int_to_label(pred_int)
        new_stress_label = auto_plan["stress_label"]

        st.write(
            f"Current predicted stress: **{current_stress_label}**  \n"
            f"Recommended new plan stress: **{new_stress_label}**"
        )

        if new_stress_label == current_stress_label:
            st.info(
                "The auto-rescheduler could not find a clearly better plan with small changes."
            )
        else:
            st.write("Based on the model, here is a suggested adjusted configuration:")

            proposed_cfg = auto_plan["config"]
            changes = describe_reschedule_changes(current_cfg, proposed_cfg)

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Current plan**")
                st.json(current_cfg)

            with col_b:
                st.markdown("**Proposed plan**")
                st.json(proposed_cfg)

            if changes:
                st.markdown("**Key changes to reduce stress:**")
                for c in changes:
                    st.markdown(f"- {c}")
            else:
                st.write("No significant parameter changes found.")

# ---------- History Section ----------
st.markdown("---")
st.header("📚 Saved Days History")

history = load_user_days()
if history.empty:
    st.write("No days saved yet.")
else:
    st.write("Previously saved days with predicted stress levels:")
    st.dataframe(history)

    # Numeric trend chart
    stress_num_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    hist_plot = history.copy()
    hist_plot["stress_num"] = history["predicted_stress_level"].map(stress_num_map)
    hist_plot["index"] = range(1, len(hist_plot) + 1)

    st.subheader("Stress Trend Across Saved Days")
    st.line_chart(hist_plot.set_index("index")["stress_num"])


# ---------- Batch CSV Analysis ----------
st.markdown("---")
st.header("📂 Batch Analysis from CSV")

st.write(
    "Upload a CSV file with columns: "
    "`total_busy_hours`, `num_meetings`, `deep_work_hours`, `commute_hours`, "
    "`sleep_hours`, `tasks_due`, `context_switches`."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)

        required_cols = [
            "total_busy_hours",
            "num_meetings",
            "deep_work_hours",
            "commute_hours",
            "sleep_hours",
            "tasks_due",
            "context_switches",
        ]

        missing = [c for c in required_cols if c not in df_upload.columns]
        if missing:
            st.error(f"Missing required columns in uploaded CSV: {missing}")
        else:
            # Derived columns
            df_upload["meeting_load_ratio"] = df_upload["num_meetings"] / (
                df_upload["total_busy_hours"] + 0.5
            )
            df_upload["deep_work_ratio"] = df_upload["deep_work_hours"] / (
                df_upload["total_busy_hours"] + 0.5
            )

            feature_cols = required_cols + ["meeting_load_ratio", "deep_work_ratio"]
            X_batch = df_upload[feature_cols]

            preds = model.predict(X_batch)
            df_upload["predicted_stress_level"] = [stress_int_to_label(p) for p in preds]

            st.subheader("Batch Analysis Result")
            st.dataframe(df_upload)

    except Exception as e:
        st.error(f"Error reading or processing CSV: {e}")
