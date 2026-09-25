"""
generate_sample_data.py
------------------------
Creates a synthetic dataset of student-project "snapshots" for training
the Risk-Assessment Agent's ML model, structured around the risk factors
most frequently reported in the capstone-project literature (time
management, tool/skill gaps, missed check-ins, workload imbalance).

Run once from the project root: python data/generate_sample_data.py
Replace with real, anonymized student-team telemetry once available.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 400

days_since_commit = np.random.exponential(scale=2.0, size=N).clip(0, 14)
pct_tasks_overdue = np.random.beta(1.5, 4, size=N)
missed_checkins = np.random.poisson(lam=0.6, size=N).clip(0, 5)
workload_ratio = np.random.normal(loc=1.0, scale=0.35, size=N).clip(0.3, 2.5)

# Ground-truth label generated from a weighted, noisy combination of the
# same factors a real instructor would look at - this is what the model
# learns to approximate.
risk_signal = (
    0.30 * (days_since_commit / 14)
    + 0.35 * pct_tasks_overdue
    + 0.20 * (missed_checkins / 5)
    + 0.15 * ((workload_ratio - 1.0).clip(0, None) / 1.5)
)
noise = np.random.normal(0, 0.08, size=N)
risk_label = ((risk_signal + noise) > 0.22).astype(int)

df = pd.DataFrame({
    "days_since_last_commit": days_since_commit.round(1),
    "pct_tasks_overdue": pct_tasks_overdue.round(2),
    "missed_checkins": missed_checkins,
    "workload_ratio": workload_ratio.round(2),
    "risk_label": risk_label,
})

df.to_csv("data/sample_risk_data.csv", index=False)
print(f"Wrote data/sample_risk_data.csv with {len(df)} rows, "
      f"{df['risk_label'].mean()*100:.1f}% labeled high-risk")
