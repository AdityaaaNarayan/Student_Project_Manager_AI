Student software teams face the same planning and risk challenges as industry teams — unclear scope, uneven skills, missed deadlines — but with far less oversight. This project is a working prototype that addresses that gap using a small team of cooperating AI agents instead of a static spreadsheet.

How it works: A Planning Agent breaks a project brief into a task plan tailored to each teammate's skills and available days. A Monitoring Agent pulls live signals directly from GitHub — commit activity, overdue milestone-tracked issues, and workload distribution across contributors. A Risk-Assessment Agent scores project health using both a transparent rule-based model and a trained Random Forest classifier. When risk crosses a threshold, a Coordinator agent automatically triggers a replan — reassigning tasks and adjusting the schedule — closing the loop between detecting a problem and acting on it.

Key features:

🧠 LLM-powered planning (Google Gemini or Anthropic Claude), with a fully functional offline mock mode — no API key required to explore the core loop
🔗 Live GitHub integration for automatic risk-signal collection
📊 Trained ML risk classifier (72% accuracy, 0.79 ROC-AUC on synthetic data modeled after real capstone-risk research)
🔁 Automatic adaptive replanning when risk crosses a threshold
🖥️ Streamlit UI for generating plans and monitoring project health
✅ Automated retry/fallback handling for LLM provider outages
