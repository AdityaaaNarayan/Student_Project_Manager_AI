"""
github_monitor.py
------------------
The Monitoring Agent: pulls live signals from a real GitHub repo instead
of requiring someone to move sliders by hand.

Uses GitHub's public REST API directly via `requests` - no extra SDK
needed. Works without a token for public repos (rate-limited to ~60
requests/hour); set GITHUB_TOKEN in .env for higher limits and to read
private repos your account has access to.

What this CAN measure from GitHub directly:
    - days_since_last_commit   -> from the commits endpoint
    - pct_tasks_overdue        -> % of open issues whose milestone due
                                   date has passed (requires the repo to
                                   use GitHub Milestones with due dates)
    - workload_ratio           -> open assigned issues per contributor,
                                   compared to an assumed baseline capacity

What GitHub has NO native concept of, so we can't auto-derive it:
    - missed_checkins          -> there's no "check-in" object in GitHub.
                                   Left for the user to enter manually, or
                                   you could wire this up to attendance
                                   data from a separate tool later.
"""

import datetime
import os
from typing import Dict, List, Optional, Tuple

import requests

GITHUB_API = "https://api.github.com"
BASELINE_CAPACITY_PER_PERSON = 3  # assumed "comfortable" # of open issues/person


def _headers() -> Dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, params: Optional[dict] = None) -> list:
    resp = requests.get(url, headers=_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_gh_date(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc
    )


def days_since_last_commit(owner: str, repo: str, branch: Optional[str] = None) -> Optional[float]:
    params = {"per_page": 1}
    if branch:
        params["sha"] = branch
    commits = _get(f"{GITHUB_API}/repos/{owner}/{repo}/commits", params)
    if not commits:
        return None
    last_date = _parse_gh_date(commits[0]["commit"]["committer"]["date"])
    delta = datetime.datetime.now(datetime.timezone.utc) - last_date
    return round(delta.total_seconds() / 86400, 1)


def _open_issues(owner: str, repo: str) -> List[dict]:
    """Open issues, excluding pull requests (GitHub's issues endpoint
    returns both; PRs carry a 'pull_request' key)."""
    issues = _get(f"{GITHUB_API}/repos/{owner}/{repo}/issues", {"state": "open", "per_page": 100})
    return [i for i in issues if "pull_request" not in i]


def pct_open_issues_overdue(owner: str, repo: str) -> Tuple[Optional[float], str]:
    """
    % of open issues whose GitHub Milestone due date has passed.
    Returns (value, note). value is None if no issues have a milestone
    with a due date set, since there's nothing to compute from.
    """
    issues = _open_issues(owner, repo)
    if not issues:
        return 0.0, "No open issues found."

    now = datetime.datetime.now(datetime.timezone.utc)
    considered = 0
    overdue = 0
    for issue in issues:
        milestone = issue.get("milestone")
        if milestone and milestone.get("due_on"):
            considered += 1
            if _parse_gh_date(milestone["due_on"]) < now:
                overdue += 1

    if considered == 0:
        return None, ("No open issues have a Milestone due-date set, so overdue % "
                       "can't be computed. Add due dates to GitHub Milestones to enable this.")
    return round(overdue / considered, 2), f"{overdue}/{considered} milestone-tracked issues are overdue."


def workload_ratio(owner: str, repo: str, team_size: Optional[int] = None) -> Tuple[float, str]:
    """
    Rough proxy for team workload: total assignments on open issues,
    divided by (team size x an assumed comfortable capacity per person).
    Not a precise measure - a simple heuristic to feed the Risk Agent.
    """
    issues = _open_issues(owner, repo)
    assigned_count = sum(len(i.get("assignees", [])) for i in issues)

    if not team_size:
        assignees = {a["login"] for i in issues for a in i.get("assignees", [])}
        team_size = len(assignees) or 1
        note = f"Team size not provided - inferred {team_size} contributor(s) from issue assignees."
    else:
        note = f"Using provided team size: {team_size}."

    ratio = round(assigned_count / (team_size * BASELINE_CAPACITY_PER_PERSON), 2)
    return ratio, f"{assigned_count} open assigned issues across {team_size} people. {note}"


def fetch_github_signals(
    owner: str, repo: str, team_size: Optional[int] = None, branch: Optional[str] = None
) -> Tuple[Dict[str, float], List[str]]:
    """
    Main entry point used by the UI. Returns (features, notes) where
    `features` matches risk_agent's FEATURE_COLUMNS shape and `notes` is
    a list of human-readable strings explaining what was found / any
    fields that couldn't be computed (shown to the user so they know to
    double check or fill those in manually).
    """
    features = {
        "days_since_last_commit": 0.0,
        "pct_tasks_overdue": 0.0,
        "missed_checkins": 0,   # not derivable from GitHub - always manual
        "workload_ratio": 1.0,
    }
    notes = []

    try:
        d = days_since_last_commit(owner, repo, branch)
        if d is not None:
            features["days_since_last_commit"] = d
            notes.append(f"Last commit was {d} day(s) ago.")
        else:
            notes.append("Repo has no commits yet.")
    except requests.HTTPError as e:
        notes.append(f"Could not fetch commits ({e.response.status_code}): "
                      f"check the repo name and that it's accessible.")
    except Exception as e:
        notes.append(f"Could not fetch commits: {e}")

    try:
        pct, note = pct_open_issues_overdue(owner, repo)
        if pct is not None:
            features["pct_tasks_overdue"] = pct
        notes.append(note)
    except requests.HTTPError as e:
        notes.append(f"Could not fetch issues ({e.response.status_code}).")
    except Exception as e:
        notes.append(f"Could not fetch issues: {e}")

    try:
        ratio, note = workload_ratio(owner, repo, team_size)
        features["workload_ratio"] = ratio
        notes.append(note)
    except requests.HTTPError as e:
        notes.append(f"Could not compute workload ratio ({e.response.status_code}).")
    except Exception as e:
        notes.append(f"Could not compute workload ratio: {e}")

    notes.append("'Missed check-ins' has no GitHub equivalent - set it manually below.")

    return features, notes
