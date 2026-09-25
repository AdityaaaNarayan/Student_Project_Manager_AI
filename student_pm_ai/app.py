

import re

import pandas as pd
import streamlit as st

from agents.coordinator import Coordinator
from agents.github_monitor import fetch_github_signals
from utils.llm_client import active_provider


def _sorted_task_df(tasks):
    """
    Sort tasks by the numeric part of task_id (T1, T2, ... T9, T10, ...)
    so dependency chains read top-to-bottom in the table, regardless of
    what order the LLM happened to output them in.
    """
    df = pd.DataFrame(tasks)
    if "task_id" in df.columns:
        df["_sort_key"] = df["task_id"].apply(
            lambda t: int(re.sub(r"\D", "", str(t)) or 0)
        )
        df = df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    return df

st.set_page_config(page_title="Student Project AI Manager", page_icon="🧭", layout="wide")

if "state" not in st.session_state:
    st.session_state.state = None
if "coordinator" not in st.session_state:
    st.session_state.coordinator = Coordinator()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("Adaptive Multi-Agent AI Framework")
st.caption(
    "Intelligent Planning & Risk-Aware Management of Student Software Projects — prototype"
)

provider = active_provider()
if provider == "gemini":
    mode = "🟢 Live LLM — using Google Gemini"
elif provider == "anthropic":
    mode = "🟢 Live LLM — using Anthropic Claude"
else:
    mode = "🟡 Offline mock mode (no API key set — using the deterministic planner)"
st.info(mode)

# ---------------------------------------------------------------------------
# STEP 1: Project setup + initial plan
# ---------------------------------------------------------------------------
st.header("1. Project Setup")

with st.form("setup_form"):
    brief = st.text_area(
        "Project brief",
        value="Build a campus event management web app with student login, "
              "event listing, and RSVP.",
        height=90,
    )

    st.markdown("**Team members** (one per line: `Name, skill1;skill2, days_available`)")
    team_raw = st.text_area(
        "Team",
        value="Aditya, backend;databases, 6\nAbhishek, frontend;design, 5\nRitik, testing;docs, 4",
        height=90,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Generate Initial Plan", type="primary")

if submitted:
    team = []
    for line in team_raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 1 and parts[0]:
            name = parts[0]
            skills = parts[1].split(";") if len(parts) > 1 else []
            days = float(parts[2]) if len(parts) > 2 else 5
            team.append({"name": name, "skills": skills, "days_available": days})

    with st.spinner("Planning Agent is generating the task plan..."):
        st.session_state.state = st.session_state.coordinator.create_initial_plan(brief, team)
    st.session_state.last_result = None
    st.success(f"Plan created with {len(st.session_state.state.tasks)} tasks.")

# ---------------------------------------------------------------------------
# STEP 2: Show current plan
# ---------------------------------------------------------------------------
if st.session_state.state:
    st.header("2. Current Task Plan")
    df = _sorted_task_df(st.session_state.state.tasks)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------
    # STEP 3: Live project signals (auto-fetch from GitHub, or simulate)
    # ---------------------------------------------------------------------
    st.header("3. Monitor & Adapt")
    st.caption(
        "Pull live signals straight from a GitHub repo, or move the sliders "
        "manually to simulate them - either way you can review/adjust "
        "before running the Risk-Assessment Agent."
    )

    # Defaults used to pre-fill the sliders below. A GitHub fetch overwrites
    # these in session_state; otherwise they keep whatever the sliders
    # were last set to.
    for key, default in [
        ("sig_days_since_commit", 1),
        ("sig_pct_overdue", 0.0),
        ("sig_missed_checkins", 0),
        ("sig_workload_ratio", 1.0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.expander("🔗 Auto-fetch signals from GitHub", expanded=False):
        st.caption(
            "Works on public repos with no setup. For private repos or "
            "higher rate limits, add GITHUB_TOKEN to your .env file."
        )
        gh_col1, gh_col2 = st.columns([2, 1])
        with gh_col1:
            gh_repo = st.text_input(
                "Repository (owner/repo)", placeholder="e.g. octocat/Hello-World"
            )
        with gh_col2:
            gh_team_size = st.number_input(
                "Team size (optional)", min_value=0, value=0, step=1,
                help="Leave 0 to infer team size from GitHub issue assignees.",
            )

        if st.button("Fetch from GitHub"):
            if "/" not in gh_repo:
                st.error("Enter the repo as `owner/repo`, e.g. `octocat/Hello-World`.")
            else:
                owner, repo_name = gh_repo.split("/", 1)
                with st.spinner("Monitoring Agent pulling live signals from GitHub..."):
                    try:
                        features, notes = fetch_github_signals(
                            owner.strip(), repo_name.strip(),
                            team_size=gh_team_size or None,
                        )
                        st.session_state["sig_days_since_commit"] = min(
                            features["days_since_last_commit"], 14
                        )
                        st.session_state["sig_pct_overdue"] = min(features["pct_tasks_overdue"], 1.0)
                        st.session_state["sig_workload_ratio"] = min(
                            max(features["workload_ratio"], 0.3), 2.5
                        )
                        for n in notes:
                            st.write(f"- {n}")
                        st.success("Sliders below updated from GitHub. Review, then run the risk check.")
                    except Exception as e:
                        st.error(f"Couldn't fetch from GitHub: {e}")

    col1, col2 = st.columns(2)
    with col1:
        days_since_commit = st.slider(
            "Days since last commit", 0, 14, key="sig_days_since_commit"
        )
        pct_overdue = st.slider(
            "% of tasks overdue", 0.0, 1.0, step=0.05, key="sig_pct_overdue"
        )
    with col2:
        missed_checkins = st.slider(
            "Missed team check-ins", 0, 5, key="sig_missed_checkins",
            help="GitHub has no concept of check-ins - set this one manually.",
        )
        workload_ratio = st.slider(
            "Workload ratio (assigned / available capacity)", 0.3, 2.5, step=0.1,
            key="sig_workload_ratio",
        )

    task_ids = [t["task_id"] for t in st.session_state.state.tasks]
    overdue_ids = st.multiselect("Mark specific tasks as overdue", task_ids)

    use_ml = st.checkbox("Use trained ML risk model (uncheck for transparent rule-based scoring)", value=True)
    st.session_state.coordinator.use_ml_risk = use_ml

    if st.button("Run Risk Check", type="primary"):
        with st.spinner("Risk-Assessment Agent scoring project state..."):
            result = st.session_state.coordinator.monitor_and_adapt(
                st.session_state.state,
                features={
                    "days_since_last_commit": days_since_commit,
                    "pct_tasks_overdue": pct_overdue,
                    "missed_checkins": missed_checkins,
                    "workload_ratio": workload_ratio,
                },
                overdue_task_ids=overdue_ids,
            )
        st.session_state.last_result = result

    if st.session_state.last_result:
        result = st.session_state.last_result
        risk = result["risk"]

        color = {"low": "green", "medium": "orange", "high": "red"}[risk.level]
        st.markdown(f"### Risk Level: :{color}[{risk.level.upper()}]  ({risk.score:.1f} / 100)")
        for r in risk.reasons:
            st.write(f"- {r}")

        if result["replanned"]:
            st.warning("⚠️ Risk threshold exceeded — the Coordinator triggered an automatic REPLAN.")
            st.subheader("Revised Plan")
            st.dataframe(_sorted_task_df(result["new_tasks"]), use_container_width=True, hide_index=True)
        else:
            st.success("Risk is within acceptable range — no replan needed.")

    # ---------------------------------------------------------------------
    # STEP 4: Audit trail
    # ---------------------------------------------------------------------
    with st.expander("📜 Full agent activity log"):
        for event in st.session_state.state.history:
            st.json(event)
else:
    st.warning("Generate an initial plan above to begin.")
