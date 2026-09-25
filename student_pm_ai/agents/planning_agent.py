"""
planning_agent.py
------------------
Turns a project brief + team roster into a structured task plan, and can
REVISE an existing plan when the Risk-Assessment Agent flags a problem.

Plan schema (a plain list of dicts, kept deliberately simple so it is
easy to render in a table, save as CSV/JSON, or hand to another agent):

{
    "task_id": "T1",
    "name": "Design database schema",
    "owner": "Aditya",
    "duration_days": 3,
    "depends_on": [],
    "status": "not_started"   # not_started | in_progress | done | overdue
}
"""

from typing import Any, Dict, List, Optional

from utils.llm_client import generate_json

SYSTEM_PROMPT = """You are the Planning Agent inside a student software-project
management system. Given a project brief and a list of team members (with their
skills and number of days available), produce a realistic task breakdown.

Rules:
- Split the project into 6-12 concrete tasks.
- Assign each task to the team member whose skills best match it.
- Balance workload roughly evenly across the team based on days available.
- Add realistic dependencies between tasks (e.g., backend before integration).
- Respond with ONLY a JSON object, no prose, in this exact shape:

{
  "tasks": [
    {"task_id": "T1", "name": "...", "owner": "...", "duration_days": 3,
     "depends_on": [], "status": "not_started"}
  ]
}
"""


def _mock_plan(brief: str, team: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic offline fallback so the system runs without an API key."""
    generic_tasks = [
        ("Requirements & scope document", 2, []),
        ("System design / architecture", 3, ["T1"]),
        ("Database / data model setup", 2, ["T2"]),
        ("Core backend implementation", 5, ["T3"]),
        ("Frontend / UI implementation", 5, ["T2"]),
        ("Integration of frontend & backend", 3, ["T4", "T5"]),
        ("Testing & bug fixing", 3, ["T6"]),
        ("Documentation & final report", 2, ["T6"]),
        ("Presentation preparation", 1, ["T7", "T8"]),
    ]
    names = [m["name"] for m in team] if team else ["Member A", "Member B", "Member C"]
    tasks = []
    for i, (name, duration, deps) in enumerate(generic_tasks, start=1):
        owner = names[(i - 1) % len(names)]
        tasks.append({
            "task_id": f"T{i}",
            "name": name,
            "owner": owner,
            "duration_days": duration,
            "depends_on": deps,
            "status": "not_started",
        })
    return {"tasks": tasks}


def generate_plan(brief: str, team: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an initial project plan from a brief + team roster."""
    team_desc = "\n".join(
        f"- {m['name']}: skills={m.get('skills', [])}, "
        f"days_available={m.get('days_available', 5)}"
        for m in team
    ) or "- (no team members supplied, assume a generic team of 3)"

    user_prompt = f"""Project brief:
{brief}

Team:
{team_desc}

Produce the task plan JSON now."""

    return generate_json(
        SYSTEM_PROMPT,
        user_prompt,
        mock_fn=lambda: _mock_plan(brief, team),
    )


def replan(
    brief: str,
    team: List[Dict[str, Any]],
    current_tasks: List[Dict[str, Any]],
    risk_notes: str,
) -> Dict[str, Any]:
    """
    Ask the Planning Agent to revise an existing plan in light of a risk
    signal (e.g., "Task T4 is 3 days overdue, owner overloaded").
    """
    team_desc = "\n".join(
        f"- {m['name']}: skills={m.get('skills', [])}, "
        f"days_available={m.get('days_available', 5)}"
        for m in team
    )

    user_prompt = f"""Project brief:
{brief}

Team:
{team_desc}

Current task plan (JSON):
{current_tasks}

The Risk-Assessment Agent has flagged the following problem(s):
{risk_notes}

Revise the plan to mitigate this risk - you may reassign owners, split a
task, adjust durations, or reorder dependencies. Keep task_ids stable
where the task is unchanged. Respond with the same JSON shape as before."""

    def _mock_replan():
        # Simple offline heuristic: push overdue/at-risk tasks to whichever
        # teammate currently has the lightest load, and stretch its duration.
        tasks = [dict(t) for t in current_tasks]
        load = {t["owner"]: 0 for t in tasks}
        for t in tasks:
            load[t["owner"]] = load.get(t["owner"], 0) + t["duration_days"]

        for t in tasks:
            if t.get("status") == "overdue":
                lightest = min(load, key=load.get)
                if lightest != t["owner"]:
                    load[t["owner"]] -= t["duration_days"]
                    t["owner"] = lightest
                    load[lightest] = load.get(lightest, 0) + t["duration_days"]
                t["duration_days"] = t["duration_days"] + 1
        return {"tasks": tasks}

    return generate_json(SYSTEM_PROMPT, user_prompt, mock_fn=_mock_replan)
