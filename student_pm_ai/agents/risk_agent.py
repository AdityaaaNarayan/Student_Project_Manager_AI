"""
risk_agent.py
-------------
Scores how risky a project's current state is, using signals inspired by
the capstone-risk literature (time management issues, tool/skill gaps,
communication breakdown are the most common realized risks in student
teams).

Two modes are provided:

1. rule_based_score()  - no dependencies, always available, easy to reason
   about and explain in a report. Good default / fallback.

2. MLRiskModel         - a small RandomForestClassifier trained on
   data/sample_risk_data.csv (synthetic, but structured after the
   real risk factors reported in the literature). Swap in real
   student-team telemetry once you start collecting it.

Both return a risk score in [0, 100] plus a short human-readable reason,
so the Coordinator can decide whether to trigger a replan and can show
students WHY the system is worried.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")

FEATURE_COLUMNS = [
    "days_since_last_commit",
    "pct_tasks_overdue",
    "missed_checkins",
    "workload_ratio",
]


@dataclass
class RiskResult:
    score: float          # 0-100, higher = riskier
    level: str            # "low" | "medium" | "high"
    reasons: List[str]


def _level_from_score(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def rule_based_score(features: Dict[str, float]) -> RiskResult:
    """
    Transparent, explainable scoring - no ML required. Weights are simple
    and easy to defend/tune in a report; this is also what runs if no
    trained model file is present yet.
    """
    days_since_commit = features.get("days_since_last_commit", 0)
    pct_overdue = features.get("pct_tasks_overdue", 0)       # 0-1
    missed_checkins = features.get("missed_checkins", 0)
    workload_ratio = features.get("workload_ratio", 1.0)     # assigned/available

    score = 0.0
    reasons = []

    if days_since_commit >= 5:
        score += 30
        reasons.append(f"No commits/activity for {days_since_commit} days")
    elif days_since_commit >= 3:
        score += 15
        reasons.append(f"Low activity: {days_since_commit} days since last commit")

    if pct_overdue >= 0.4:
        score += 35
        reasons.append(f"{pct_overdue*100:.0f}% of tasks are overdue")
    elif pct_overdue >= 0.2:
        score += 18
        reasons.append(f"{pct_overdue*100:.0f}% of tasks are overdue")

    if missed_checkins >= 2:
        score += 20
        reasons.append(f"{missed_checkins} missed team check-ins")
    elif missed_checkins == 1:
        score += 8
        reasons.append("1 missed team check-in")

    if workload_ratio >= 1.5:
        score += 15
        reasons.append(f"Team overloaded ({workload_ratio:.1f}x available capacity)")

    score = min(score, 100)
    if not reasons:
        reasons.append("No significant risk signals detected")

    return RiskResult(score=score, level=_level_from_score(score), reasons=reasons)


class MLRiskModel:
    """Loads a trained RandomForest model (see models/train_risk_model.py)."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        if os.path.exists(model_path):
            self._model = joblib.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def score(self, features: Dict[str, float]) -> RiskResult:
        if not self.is_trained:
            # Graceful fallback so the app never crashes if training hasn't
            # been run yet.
            return rule_based_score(features)

        row = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLUMNS}])
        proba = self._model.predict_proba(row)[0][1]  # P(high risk)
        score = round(proba * 100, 1)

        importances = dict(zip(FEATURE_COLUMNS, self._model.feature_importances_))
        top_features = sorted(importances, key=importances.get, reverse=True)[:2]
        reasons = [f"Model flagged '{f}' as a top risk driver" for f in top_features]
        reasons.append(f"Predicted risk probability: {proba*100:.0f}%")

        return RiskResult(score=score, level=_level_from_score(score), reasons=reasons)


def assess(features: Dict[str, float], use_ml: bool = True) -> RiskResult:
    """Convenience entry point used by the Coordinator."""
    if use_ml:
        model = MLRiskModel()
        if model.is_trained:
            return model.score(features)
    return rule_based_score(features)
