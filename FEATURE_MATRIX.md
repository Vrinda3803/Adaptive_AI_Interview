# Feature and Test Matrix

All 31 tests run offline with Python's built-in `unittest` runner. Gemini question/profile responses are mocked for these tests. The Gemini failure cases are also mocked; no live Gemini API request was verified because credentials were not configured for this standalone application.

| Required feature | Implementation | Tested | Result |
|---|---|---:|---|
| Resume parsing and grounded Gemini profile extraction | `resume/resume_parser.py`, `llm/profile_extractor.py` | Yes | PASS: TXT parsed; mocked extracted facts retained only when grounded; failure uses local parser |
| Resume-based questions | `core/category_agent.py`, `core/question_bank.py` | Yes | PASS: learned category policy selects resume-based items when legal; prompt receives resume context |
| Skill-based questions | `core/category_agent.py`, `core/question_bank.py` | Yes | PASS: selected skill and difficulty reach the question prompt |
| Theoretical questions | `core/question_bank.py` | Yes | PASS: distinct category accepted through the engine |
| Scenario-based questions | `core/question_bank.py` | Yes | PASS: distinct category accepted through the engine |
| Project-based questions | `core/question_bank.py` | Yes | PASS: project context reaches generator and fallback |
| HR and behavioural questions | `core/knowledge_graph.py`, `core/category_agent.py`, `core/evaluator.py` | Yes | PASS: RL-selected Communication & Leadership action is constrained to HR; STAR-aware fallback evaluated |
| Contextual follow-ups | `core/category_agent.py`, `core/question_bank.py` | Yes | PASS: learned category policy can select a same-skill follow-up when legal |
| Coding prompt, samples, additional tests, execution | `core/question_bank.py`, `core/evaluator.py`, `app.py` | Yes | PASS: mocked coded item executed; all hidden tests passed; reasoning scored; imports blocked |
| Exact and semantic duplicate prevention | `core/assessment.py`, `core/session.py`, `core/question_bank.py` | Yes | PASS: exact/paraphrase duplicates rejected; distinct aspect permitted |
| Question bank exhaustion recovery | `core/question_bank.py`, `core/session.py`, `tests/test_application.py` | Yes | PASS: exhausted action is masked and another legal RL action is selected; interview ends cleanly only when all legal actions are exhausted |
| Persistent per-session question and answer history | `core/session.py`, `interview/live_controller.py`, `app.py` | Yes | PASS: IDs, category, skill, difficulty, type, answer, evaluation and follow-up status retained across turns |
| RL action selection and valid mask | `core/action_net.py`, `core/environment.py` | Yes | PASS: selected checkpoint loaded; chosen action valid within 48-action mask |
| Learned assessment-category selection | `core/category_agent.py`, `models/category_q_policy.json`, `train_category_policy.py` | Yes | PASS: trained Q-policy selects from context-legal categories and records state/action values; coding and HR categories follow PPO-selected legal constraints |
| RL-owned fallback decision after bank exhaustion | `core/session.py`, `core/action_net.py` | Yes | PASS: exhausted actions are temporarily masked and the RL action policy selects the next legal action; policy failures are surfaced rather than replaced with first-valid heuristics |
| Correct LLM conditioning by RL action | `core/session.py`, `core/question_bank.py` | Yes | PASS: prompt contains selected skill, difficulty, type, category, resume and transcript |
| Structured answer evaluation and feedback | `core/evaluator.py` | Yes | PASS: mocked structured Gemini response validated; outage fallback tested |
| BKT parameters and updates | `core/bkt.py`, `core/session.py` | Yes | PASS: BKT mastery changed independently from IRT |
| Bayesian IRT, uncertainty and confidence intervals | `core/bayesian_irt.py`, `core/session.py` | Yes | PASS: 71-grid posterior updated and report intervals produced |
| Knowledge Graph evidence propagation | `core/knowledge_graph.py`, `core/environment.py` | Yes | PASS: related-skill posterior changed after evidence update |
| Adaptive difficulty | `core/environment.py`, `core/session.py` | Yes | PASS: policy-selected easy then hard action observed |
| Interview time/budget termination and final report | `interview/live_controller.py`, `core/session.py`, `app.py` | Yes | PASS: one-question budget terminates and report includes BKT/IRT metrics |
| Practice, hiring and research shared engine | `app.py`, `interview/live_controller.py`, `core/session.py` | Yes | PASS: all three controller modes exercise the same assessment/session engine |
| Seeded paired RL benchmark and uncertainty summaries | `core/benchmark.py`, `run_research_benchmark.py`, `reports/seeded_benchmark/` | Yes | PASS: 5 RL methods × 3 saved checkpoint seeds plus 3 heuristics; 100 common held-out episodes per checkpoint; 10,000 hierarchical bootstrap resamples |
| Episode-seeded simulation reproducibility | `core/environment.py`, `tests/test_application.py` | Yes | PASS: same episode seed reproduces the candidate and answer outcome sequence |
| Gemini model fallback order | `llm/gemini_client.py`, `tests/test_application.py` | Yes | PASS: 3.5 Flash-Lite → 3.6 Flash → 3.8 Flash; mocked 404 on first endpoint falls back to second |
| Research algorithms, held-out, ablation and robustness | `core/research.py`, `reports/`, `app.py` | Yes | PASS: held-out smoke evaluation and bundled comparison tables/plots |
| Synthetic hidden mastery | `core/environment.py` | Yes | PASS: latent mastery is not returned in observations |
| Gemini outages | `llm/profile_extractor.py`, `core/question_bank.py`, `core/evaluator.py` | Yes | PASS: mocked failures use local/category fallback without duplicate return |
| Full demonstration: resume, technical, HR, coding | `demo_interview.py` | Yes | PASS: four-turn demo completed using PPO scores and mocked Gemini responses |
| Streamlit launch and seeded benchmark dashboard | `app.py`, `reports/seeded_benchmark/` | Yes | PASS: fresh launch returned HTTP 200; dashboard summary and figure assets are present; full mode interaction was not browser-automated |
