"""
coordinator.py
--------------
The Orchestrator/Coordinator Agent. This is the piece that makes the
system "adaptive" rather than a one-shot planner: it watches the current
project state, asks the Risk-Assessment Agent for a score, and - if risk
crosses a threshold - asks the Planning Agent to revise the plan.

This module has no LLM/ML dependency of its own; it just wires the other
two agents together, so it's easy to read, test, and explain in a report.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents import planning_agent, risk_agent

HIGH_RISK_THRESHOLD = 55  # score (0-100) at which we trigger an automatic replan


@dataclass
class ProjectState:
    brief: str
    team: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)  # audit trail


class Coordinator:
    def __init__(self, use_ml_risk: bool = True, risk_threshold: float = HIGH_RISK_THRESHOLD):
        self.use_ml_risk = use_ml_risk
        self.risk_threshold = risk_threshold

    # ---------- Step 1: initial plan ----------
    def create_initial_plan(self, brief: str, team: List[Dict[str, Any]]) -> ProjectState:
        plan = planning_agent.generate_plan(brief, team)
        state = ProjectState(brief=brief, team=team, tasks=plan["tasks"])
        state.history.append({"event": "plan_created", "num_tasks": len(state.tasks)})
        return state

    # ---------- Step 2: risk check ----------
    def check_risk(self, state: ProjectState, features: Dict[str, float]) -> risk_agent.RiskResult:
        result = risk_agent.assess(features, use_ml=self.use_ml_risk)
        state.history.append({
            "event": "risk_checked",
            "score": result.score,
            "level": result.level,
            "reasons": result.reasons,
        })
        return result

    # ---------- Step 3: the adaptive loop ----------
    def monitor_and_adapt(
        self,
        state: ProjectState,
        features: Dict[str, float],
        overdue_task_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run one full cycle: score risk, and if it's high, trigger a replan.

        Returns a dict summarizing what happened, so the UI (or a CLI)
        can display it directly:
            {
                "risk": RiskResult,
                "replanned": bool,
                "new_tasks": [...] | None,
            }
        """
        overdue_task_ids = overdue_task_ids or []

        # Mark overdue tasks so the Planning Agent's mock/LLM prompt can see them
        for t in state.tasks:
            if t["task_id"] in overdue_task_ids and t["status"] != "done":
                t["status"] = "overdue"

        risk_result = self.check_risk(state, features)

        replanned = False
        new_tasks = None

        if risk_result.score >= self.risk_threshold:
            risk_notes = "; ".join(risk_result.reasons)
            revised = planning_agent.replan(
                brief=state.brief,
                team=state.team,
                current_tasks=state.tasks,
                risk_notes=risk_notes,
            )
            new_tasks = revised["tasks"]
            state.tasks = new_tasks
            state.history.append({
                "event": "replanned",
                "trigger_score": risk_result.score,
                "reasons": risk_result.reasons,
            })
            replanned = True

        return {
            "risk": risk_result,
            "replanned": replanned,
            "new_tasks": new_tasks,
        }
