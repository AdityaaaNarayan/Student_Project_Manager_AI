# Adaptive Multi-Agent AI Framework — Prototype

Intelligent Planning and Risk-Aware Management of Student Software Projects.

This is a **working prototype**, not a mockup: the planning loop, the risk
scoring (rule-based *and* trained ML model), and the adaptive replanning
cycle all run end-to-end, with or without an LLM API key.

## What's actually implemented

| Component | Status |
|---|---|
| Planning Agent (generates task plans) | ✅ Working (LLM + offline mock mode) |
| Risk-Assessment Agent (rule-based) | ✅ Working, tested |
| Risk-Assessment Agent (ML / RandomForest) | ✅ Trained on synthetic data (72% accuracy, 0.79 ROC-AUC) |
| Coordinator (adaptive feedback loop) | ✅ Working, tested — verified it correctly reassigns overdue tasks under high risk |
| Streamlit UI | ✅ Written, ready to run locally |
| Monitoring Agent (live GitHub signals) | ✅ Working, tested — auto-fetches commit activity, overdue issues, workload from a real repo |
| Scheduling Agent | ⏳ Not yet built — see "Next steps" |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) generate the synthetic risk dataset and train the model
#    — a trained model is already included at models/risk_model.pkl,
#    so this step is optional unless you want to regenerate it.
python data/generate_sample_data.py
python models/train_risk_model.py

# 3. (Optional) enable a real LLM for the Planning Agent
cp .env.example .env
# then edit .env and add EITHER:
#   GEMINI_API_KEY=...     (recommended — free tier, get one at https://aistudio.google.com)
#   ANTHROPIC_API_KEY=...  (pay-as-you-go, get one at https://console.anthropic.com)
# If both are set, Gemini is used by default. Force a specific one with
# LLM_PROVIDER=gemini or LLM_PROVIDER=anthropic in .env.

# 4. Run the tests
python tests/test_core.py

# 5. Launch the app
streamlit run app.py
```

Without step 3, the app still works fully — the Planning Agent falls back
to a deterministic offline planner so you can demo and develop without
any API costs.

**Provider auto-detection:** `utils/llm_client.py` checks `GEMINI_API_KEY`
first, then `ANTHROPIC_API_KEY`, then falls back to mock mode. The app's
status banner tells you which one is active. If a key is set but its SDK
package isn't installed (`google-genai` or `anthropic`), it also falls
back to mock mode gracefully rather than crashing.

## Architecture

```
                    ┌───────────────────────────┐
                    │   Orchestrator/Coordinator │
                    │        (coordinator.py)    │
                    └─────────────┬─────────────┘
                     ┌────────────┼────────────┐
                     ▼                          ▼
          ┌─────────────────────┐   ┌─────────────────────────┐
          │   Planning Agent     │   │  Risk-Assessment Agent   │
          │  (planning_agent.py) │   │     (risk_agent.py)      │
          │  - generate_plan()   │   │  - rule_based_score()    │
          │  - replan()          │   │  - MLRiskModel (trained) │
          └─────────────────────┘   └─────────────────────────┘

Loop: create plan → monitor project signals → score risk →
      if risk ≥ threshold → replan → update plan → repeat
```

## Project structure

```
student_pm_ai/
├── app.py                       # Streamlit UI
├── requirements.txt
├── .env.example
├── agents/
│   ├── planning_agent.py        # task decomposition + adaptive replanning
│   ├── risk_agent.py            # rule-based + ML risk scoring
│   └── coordinator.py           # the feedback loop tying agents together
├── utils/
│   └── llm_client.py            # Anthropic API wrapper w/ mock fallback
├── data/
│   ├── generate_sample_data.py  # synthetic risk dataset generator
│   └── sample_risk_data.csv     # 400-row synthetic dataset (already generated)
├── models/
│   ├── train_risk_model.py      # trains + saves the RandomForest model
│   └── risk_model.pkl           # already-trained model (ready to use)
└── tests/
    └── test_core.py             # 5 passing sanity tests, no API key required
```

## How the risk signals map to the literature

The four features used by the Risk-Assessment Agent were chosen to mirror
the recurring risk factors identified in student-capstone risk-management
studies:

- `days_since_last_commit` → team inactivity / stalled progress
- `pct_tasks_overdue` → schedule slippage
- `missed_checkins` → communication breakdown
- `workload_ratio` → uneven/overloaded task distribution

Replace `data/sample_risk_data.csv` with real, anonymized data from actual
student teams (task-board exports, Git logs, attendance records) to retrain
a model that reflects your own institution's projects.

## Auto-fetching signals from GitHub

In the app's "3. Monitor & Adapt" section, expand **🔗 Auto-fetch signals
from GitHub**, enter a repo as `owner/repo` (e.g. `octocat/Hello-World`),
and click **Fetch from GitHub**. This pulls real data via `agents/github_monitor.py`:

| Signal | How it's computed |
|---|---|
| `days_since_last_commit` | Time since the most recent commit |
| `pct_tasks_overdue` | % of open issues whose GitHub **Milestone** due date has passed (requires the repo to use Milestones with due dates - otherwise this stays 0 with a note explaining why) |
| `workload_ratio` | Open assigned issues ÷ (team size × an assumed capacity of 3 issues/person) — team size is inferred from distinct assignees if you don't provide one |
| `missed_checkins` | **Not derivable from GitHub** — there's no native "check-in" concept, so this stays manual. Wire it up to attendance/standup data from another tool if you have one. |

Works on public repos with zero setup (rate-limited to ~60 requests/hour).
For private repos or a higher rate limit, create a read-only token at
https://github.com/settings/tokens and set `GITHUB_TOKEN` in `.env`.

The fetched values pre-fill the sliders, but you can still review and
adjust them by hand before running the risk check — the fetch is a
starting point, not a black box.

## Next steps (not yet built)

1. **Scheduling Agent** — currently the Planning Agent does reassignment
   directly during `replan()`. Splitting this into its own agent (as in the
   architecture diagram from the proposal) would let you swap in a proper
   optimization/RL-based allocator later.
2. **Task-board integration** — the Monitoring Agent currently reads commit
   activity and GitHub Issues/Milestones. If your team tracks tasks in
   Trello, Jira, or GitHub Projects (v2) instead of plain Issues, extend
   `agents/github_monitor.py` (or add a sibling module) to pull from that
   API too.
3. **Persistent storage** — add a database (SQLite is enough) so plans and
   history survive app restarts, and so you can log real pilot-study data.
4. **Model retraining** — once you collect real risk-labeled data from a
   pilot team, retrain `models/train_risk_model.py` on it instead of the
   synthetic dataset.
5. **Evaluation** — run the pilot study described in the methodology
   document: one team using this tool vs. one using manual tracking,
   comparing missed deadlines and risk-detection lead time.

## Notes

- The ML model was trained on **synthetic data** (see `generate_sample_data.py`)
  built to reflect realistic risk-factor relationships, not real student
  projects. Its 72%/0.79 ROC-AUC numbers are a reasonable starting point for
  a prototype, but you should retrain on real data before citing performance
  numbers in a report.
- `MODEL_NAME` in `utils/llm_client.py` defaults to `claude-sonnet-4-6` —
  change it via the `.env` file if you want a different model.
