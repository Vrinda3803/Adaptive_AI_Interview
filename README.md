# Adaptive AI Interview Simulator — Presentation Application

This is the clean, independent Streamlit application built from the research implementation and the project's original guide and `RL_Nov2_1.ipynb`. Its hierarchical RL controller makes the adaptive interview decisions: a custom PyTorch PPO policy selects skill, difficulty, and question type, and a trained category Q-policy selects the question category. Bayesian IRT, the Knowledge Graph, coding evaluation, Gemini integration, and research reports support that loop.

## Run on Windows

From the existing research project root:

```powershell
cd C:\Users\user\Adaptive_AI_Interview_RL\Adaptive_AI_Interview_Final
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY to enable live Gemini calls.
streamlit run app.py
```

Open the local Streamlit URL (usually `http://localhost:8501`). No API key is included. The app only reads `.env` from this project folder and never searches the parent research workspace. Without Gemini credentials, the app uses category-specific curated questions, local resume parsing, and a category-aware evaluator fallback. Gemini live generation was not tested without an API key; mocked Gemini behavior is covered by the test suite.

Live Gemini requests try `gemini-3.5-flash-lite`, then `gemini-3.6-flash`, then `gemini-3.8-flash`; an unavailable endpoint moves to the next model. These IDs are verified against [Google's model catalog](https://ai.google.dev/gemini-api/docs/models).

For scanned PDFs, install the Tesseract OCR executable and set `TESSERACT_CMD` in `.env` if it is not on `PATH`. Text PDFs, DOCX, and TXT do not require OCR.

## Tests and offline demonstration

```powershell
cd C:\Users\user\Adaptive_AI_Interview_RL\Adaptive_AI_Interview_Final
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python demo_interview.py
```

The demo uses the selected PPO checkpoint to rank actions under explicit stage masks so one run demonstrates resume-based, technical, HR, and coding turns. Its Gemini responses are mocked; the demo prints this distinction. In normal interviews there are no stage masks: the trained policy selects skill, difficulty, and conceptual/coding type at each turn.

See [FEATURE_MATRIX.md](FEATURE_MATRIX.md) for implementation and test results.

## Adaptive decision flow

The primary PPO checkpoint remains unchanged. Its `ActionNet` scores `g[8]`, `f[48,19]`, and the valid-action mask, then selects one legal action from the original 48-action space: `(skill slot, difficulty, conceptual/coding type)`. A second, trained tabular Q-learning policy selects the assessment category from context-legal choices using previous category coverage, resume/project context, follow-up eligibility, answer score, and remaining budget. Its 50,000-episode policy is saved in `models/category_q_policy.json` and can be reproduced with `python train_category_policy.py --episodes 50000 --seed 20261103`. Coding and behavioral categories are constrained by the PPO-selected coding type and Communication & Leadership skill, respectively. When one category's unique questions are exhausted, the category policy tries another legal category for the same PPO-selected action; if needed, PPO then chooses another action under a temporary validity mask. Gemini only generates question wording and evaluates the candidate's answer after the RL decisions. It cannot choose the next action or category.

The learned policies control adaptive question selection. The live interview honors the selected question budget, and confidence-based early stopping can only occur after at least eight questions (or the entire smaller configured budget). Legal-action masks and the time limit remain hard environment constraints.

Communication & Leadership is a non-coding assessment skill added in a separate Behavioral Competencies domain for HR coverage. It has no graph edges to technical skills, so behavioural evidence cannot bleed into technical posteriors. This does not change ActionNet tensor shapes or checkpoint keys. The policy's feature-based design scores it from its feature row; all 40 original technical skills and graph relationships remain unchanged. The dashboard distinguishes the 40 technical skills from this added behavioral assessment dimension.

Question IDs, normalized text, category, selected skill, difficulty, question type, action ID, answer, evaluation, and initial/follow-up status are retained in `InterviewSession.question_history` for the Streamlit session. Before display, generated questions pass exact and synonym-normalized lexical similarity checks. Gemini gets two attempts with explicit exclusions; if a selected action's compatible fallbacks are exhausted, the session temporarily masks that action and asks the RL policy for its next-best legal action. The interview ends cleanly only if no legal action can generate a new question. The same session/controller object is held in Streamlit session state between turns.

## BKT and Bayesian IRT

Both trackers update from evaluated answers, but they are not substituted for one another:

- **BKT** (initial mastery 0.20, learning 0.15, guess 0.20, slip 0.10) provides a parallel, interpretable discrete mastery estimate shown in the report.
- **Bayesian IRT** is the research model used for the RL observation, continuous competency estimate, uncertainty, 95% intervals, expected information gain, stopping, and evidence propagation through the Knowledge Graph. Its 71 point grid and confidence-weighted likelihood remain the primary adaptive state.

The synthetic research environment retains hidden mastery: it generates latent candidate state internally and does not include it in observations. Interview budgets, time limits, action masks, the 40-skill knowledge graph, and held-out domain evaluation are preserved.

## Modes

- **Practice Interview:** resume upload or manual profile, personalized adaptive assessment, coding, feedback, competency report, and transcript.
- **Hiring & Candidate Comparison:** job context, question/time/competency settings, human-review indicators, and CSV comparison across completed candidates in the current Streamlit session.
- **Research Dashboard:** select a compatible custom `.pt` checkpoint in `models/`, inspect static multi-algorithm benchmark/ablation/generalization/noise results, view research plots, run the selected policy on held-out synthetic domains, inspect live action and uncertainty traces, and export results.

The dashboard also shows the paired, three-checkpoint-seed benchmark generated by `run_research_benchmark.py`. See [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for the endpoint definitions, bootstrap procedure, checkpoint provenance, and limitations. The report is exploratory: only three trained checkpoint seeds are available per RL method, and all candidates are synthetic. It is not a real-world hiring validity or fairness study.

The standalone folder currently includes the selected PPO checkpoint. Additional compatible custom `.pt` checkpoints can be placed in `models/` and selected from the research page; older Stable-Baselines3 ZIP checkpoints use a different architecture and are not accepted by this custom policy loader.

## Checkpoint selection and compatibility

`models/adaptive_ppo_seed1.pt` is copied from `outputs/checkpoints/PPO_seed1.pt`. The PPO seeds were compared with the original 40-skill registry on 100 deterministic simulated interviews through the research `InterviewEnv` and `load_policy` path. Seed 1 achieved the highest success rate (15%, versus 10% for seeds 0 and 2) and lowest weighted RMSE (0.697, versus 0.745 and 0.762). It was selected based on same-protocol behavior, not timestamp. Metadata identifies PPO seed 1, 4,000 training episodes.

The architecture is `ActionNet(H=96)`, with an 8-value global feature, 19 features per action, 48 actions, and a masked greedy PPO scorer. The copied checkpoint's tensors load strictly. The semantic action mapping is `((skill_slot * 3) + difficulty) * 2 + question_type`. The registry extension changes neither dimensions nor state-dict keys. The checkpoint's SHA-256 was verified equal to the original.

## Structure

- `app.py` — candidate/hiring flows and researcher dashboard
- `core/assessment.py` — question normalization and duplicate checks
- `core/category_agent.py`, `models/category_q_policy.json` — learned category-selection Q-policy
- `core/action_net.py`, `core/environment.py` — trained policy, masks, synthetic/real interview states
- `core/bayesian_irt.py`, `core/bkt.py`, `core/knowledge_graph.py` — separate tracking models and skill graph
- `core/question_bank.py`, `core/evaluator.py`, `core/session.py` — question generation, code runner, scoring, history, reports
- `interview/` — controller connecting UI and shared session engine
- `llm/` — Gemini client and grounded profile extraction
- `resume/` — PDF, DOCX, TXT parsing
- `models/` — selected PPO checkpoint, the category Q-policy, and 15 benchmark checkpoints (five RL methods × three seeds)
- `reports/` — only dashboard tables and figures
- `tests/` — offline unit/integration checks
- `demo_interview.py` — complete mocked-Gemini demonstration using PPO action scores
- `train_category_policy.py` — reproducible training entry point for the category Q-policy
- `run_research_benchmark.py`, `core/benchmark.py` — paired seed-based research evaluation and report generation


