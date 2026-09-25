"""
test_core.py
------------
Minimal sanity tests you can run with:  pytest tests/
(or just: python tests/test_core.py)

These don't require any API key - they exercise the offline mock
planner and the rule-based / ML risk agent directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.coordinator import Coordinator
from agents import risk_agent


def test_rule_based_low_risk():
    result = risk_agent.rule_based_score({
        "days_since_last_commit": 0,
        "pct_tasks_overdue": 0.0,
        "missed_checkins": 0,
        "workload_ratio": 1.0,
    })
    assert result.level == "low"
    assert result.score < 40


def test_rule_based_high_risk():
    result = risk_agent.rule_based_score({
        "days_since_last_commit": 7,
        "pct_tasks_overdue": 0.6,
        "missed_checkins": 3,
        "workload_ratio": 1.8,
    })
    assert result.level == "high"
    assert result.score >= 70


def test_initial_plan_has_tasks():
    c = Coordinator()
    team = [{"name": "A", "skills": ["backend"], "days_available": 2}]
    state = c.create_initial_plan("Build a to-do list app", team)
    assert len(state.tasks) > 0
    assert all("task_id" in t for t in state.tasks)


def test_replan_triggers_on_high_risk():
    c = Coordinator(risk_threshold=1)  # force trigger for this test
    team = [{"name": "A", "skills": ["backend"], "days_available": 2}]
    state = c.create_initial_plan("Build a to-do list app", team)
    result = c.monitor_and_adapt(
        state,
        features={"days_since_last_commit": 5, "pct_tasks_overdue": 0.5,
                   "missed_checkins": 2, "workload_ratio": 1.5},
    )
    assert result["replanned"] is True
    assert result["new_tasks"] is not None


def test_replan_not_triggered_on_low_risk():
    c = Coordinator(risk_threshold=99999)  # effectively never trigger
    team = [{"name": "A", "skills": ["backend"], "days_available": 2}]
    state = c.create_initial_plan("Build a to-do list app", team)
    result = c.monitor_and_adapt(
        state,
        features={"days_since_last_commit": 0, "pct_tasks_overdue": 0.0,
                   "missed_checkins": 0, "workload_ratio": 1.0},
    )
    assert result["replanned"] is False


def test_github_monitor_computes_signals_from_mocked_api():
    """
    Verifies the GitHub Monitoring Agent's math without hitting the real
    GitHub API - we monkeypatch the low-level _get() call with realistic
    fake responses and check the computed features.
    """
    import datetime
    from agents import github_monitor as gm

    now = datetime.datetime.now(datetime.timezone.utc)

    def fmt(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    fake_commits = [{"commit": {"committer": {"date": fmt(now - datetime.timedelta(days=4))}}}]
    fake_issues = [
        {"number": 1, "assignees": [{"login": "a"}],
         "milestone": {"due_on": fmt(now - datetime.timedelta(days=2))}},
        {"number": 2, "assignees": [{"login": "a"}, {"login": "b"}],
         "milestone": {"due_on": fmt(now + datetime.timedelta(days=5))}},
        {"number": 3, "pull_request": {}, "assignees": [{"login": "a"}]},  # excluded
    ]

    original_get = gm._get
    try:
        gm._get = lambda url, params=None: fake_commits if "commits" in url else fake_issues
        features, notes = gm.fetch_github_signals("org", "repo")
        assert 3.5 <= features["days_since_last_commit"] <= 4.5
        assert features["pct_tasks_overdue"] == 0.5  # 1 of 2 milestone-tracked issues overdue
        assert len(notes) > 0
    finally:
        gm._get = original_get  # don't leak the monkeypatch to other tests


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASSED: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed.")
