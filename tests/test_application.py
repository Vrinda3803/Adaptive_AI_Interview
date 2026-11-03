import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

import numpy as np

from core.action_net import load_policy
from core.assessment import is_duplicate
from core.bayesian_irt import Belief, item_prob_table
from core.category_agent import CategoryRLAgent
from core.environment import InterviewEnv
from core.evaluator import AnswerEvaluator
from core.knowledge_graph import REG
from core.question_bank import QuestionBank, QuestionExhaustedError, run_code_tests
from core.research import evaluate_ablation, evaluate_heldout, evaluate_noise_robustness
from interview.live_controller import LiveInterviewController
from llm.profile_extractor import extract_resume_profile
from llm import gemini_client
from resume.resume_parser import build_candidate_profile


APP = Path(__file__).resolve().parents[1]
MODEL = APP / "models" / "adaptive_ppo_seed1.pt"


class FixedActionPolicy:
    """Constrained-policy test harness; selects a requested legal RL action."""
    name = "test-action-policy"

    def __init__(self, skill=None, difficulty=1, question_type=0, difficulty_sequence=None):
        self.skill = skill
        self.difficulty = difficulty
        self.question_type = question_type
        self.difficulty_sequence = list(difficulty_sequence or [])
        self.calls = 0

    def __call__(self, obs, env):
        valid = np.flatnonzero(obs["m"])
        if self.difficulty_sequence:
            d = self.difficulty_sequence[min(self.calls, len(self.difficulty_sequence) - 1)]
        else:
            d = self.difficulty
        self.calls += 1
        for action in valid:
            slot, sk, actual_d, ty = env.decode(int(action))
            if actual_d == d and ty == self.question_type and (self.skill is None or env.reg.names[sk] == self.skill):
                return int(action)
        return int(valid[0])


class MockGemini:
    def __init__(self):
        self.prompts = []
        self.repeat = None

    def __call__(self, prompt):
        self.prompts.append(prompt)
        category = re.search(r"REQUIRED ASSESSMENT CATEGORY: ([a-z_]+)", prompt).group(1)
        skill = re.search(r"TARGET SKILL: (.+)", prompt).group(1).strip()
        difficulty = re.search(r"TARGET DIFFICULTY: (.+)", prompt).group(1).strip()
        if self.repeat is not None:
            question = self.repeat
        elif category == "coding":
            question = f"Implement solve(values) for this {skill} coding challenge and return the number of values."
            return json.dumps({
                "category": category, "question": question,
                "reference_answer": "def solve(values):\n    return len(values)",
                "expected_evidence": ["Correct implementation"],
                "evaluation_criteria": "Run hidden unit tests.",
                "sample_tests": [["solve([1, 2])", 2]],
                "tests": [["solve([1, 2])", 2], ["solve([])", 0]],
            })
        else:
            question = f"For a {category} assessment of {skill} at {difficulty}, what decision would you make in a realistic engineering situation?"
        return json.dumps({
            "category": category, "question": question,
            "expected_evidence": ["reasoned decision", "supporting evidence"],
            "evaluation_criteria": "Assess the answer against its requested category.",
            "reference_answer": "Explain the decision, evidence, and trade-offs.",
            "tests": None,
        })


class FirstLegalPolicy:
    name = "first-legal-test-policy"

    def __call__(self, obs, env):
        return int(np.flatnonzero(obs["m"])[0])


def create_controller(skill_policy=None, max_questions=10, mode="practice"):
    policy = skill_policy or FixedActionPolicy()
    profile = {
        "resume_text": "Python SQL machine learning. Project: built a data platform with Python and SQL. Led teamwork and resolved a deadline conflict.",
        "skills": ["Python", "SQL", "Machine Learning"],
        "projects": "Built a data platform with Python and SQL.",
        "experience": "Led teamwork and resolved a deadline conflict.",
    }
    controller = LiveInterviewController(profile, "Backend role using Python SQL and distributed systems", max_questions=max_questions, policy=policy, mode=mode)
    mock = MockGemini()
    controller.bank.llm_generate = mock
    controller.evaluator.use_llm = False
    return controller, mock


class ApplicationRecoveryTests(unittest.TestCase):
    def test_question_bank_keeps_distinct_fallbacks_across_live_turns(self):
        bank = QuestionBank(reg=REG)
        bank.llm_generate = lambda prompt: None
        history = []
        for _ in range(6):
            question = bank.get(
                REG.idx["Python"], 1, 0,
                {"category": "skill_based", "history": history},
            )
            self.assertFalse(is_duplicate(question["question"], history))
            history.append({"question": question["question"], "category": question["category"]})

    def test_live_budget_is_not_reduced_by_detected_skill_count(self):
        controller, _ = create_controller(max_questions=12)
        self.assertEqual(controller.session.env.B, 12)
        self.assertEqual(controller.session.env.cfg.min_questions_before_stop, 8)

    def test_confidence_stopping_waits_until_live_minimum_question_count(self):
        controller, _ = create_controller(max_questions=10)
        env = controller.session.env
        self.assertEqual(env.cfg.min_questions_before_stop, 8)

        env.cnt[:env.K] = 1
        env.t = 6
        env.belief.logp[:] = -100.0
        env.belief.logp[:, 35] = 0.0
        env.belief._refresh()

        action = int(np.flatnonzero(env._obs()["m"])[0])
        _, _, done, info = env.step(action, score=0.5)
        self.assertFalse(done)
        self.assertFalse(info["stopped_early"])

        action = int(np.flatnonzero(env._obs()["m"])[0])
        _, _, done, info = env.step(action, score=0.5)
        self.assertTrue(done)
        self.assertTrue(info["stopped_early"])

    def test_live_interview_can_generate_question_seven_without_gemini(self):
        controller, _ = create_controller(max_questions=10)
        controller.session.policy = load_policy(MODEL)
        controller.bank.llm_generate = lambda prompt: None

        for _ in range(6):
            question = controller.generate_next_question()
            self.assertFalse(question["finished"])
            controller.submit_answer(
                "I would test edge cases, measure the trade-offs, and validate the result against a baseline."
            )

        next_question = controller.generate_next_question()
        self.assertFalse(next_question["finished"])
        self.assertTrue(next_question["question"])
        self.assertEqual(controller.questions_asked, 6)

    def test_repeated_coding_action_reports_exhaustion_to_controller(self):
        bank = QuestionBank(reg=REG)
        first = bank.get(REG.idx["Python"], 1, 1, {"category": "coding", "history": []})
        history = [{"question": first["question"], "category": "coding"}]
        with self.assertRaises(QuestionExhaustedError):
            bank.get(REG.idx["Python"], 1, 1, {"category": "coding", "history": history})

    def test_session_retries_exhausted_action_with_next_category_policy_choice(self):
        controller, _ = create_controller()
        session = controller.session
        session.policy = FirstLegalPolicy()
        first_action = int(np.flatnonzero(session.obs["m"])[0])
        original_get = session.bank.get
        calls = []

        def exhaust_once(skill, difficulty, question_type, state):
            calls.append((skill, difficulty, question_type, state["category"]))
            if len(calls) == 1:
                raise QuestionExhaustedError("simulated exhausted first action")
            return original_get(skill, difficulty, question_type, state)

        session.bank.get = exhaust_once
        question = session.next_question()
        self.assertIsNotNone(question)
        self.assertEqual(len(calls), 2)
        self.assertEqual(session.cur[0], first_action)
        self.assertNotEqual(calls[0][3], calls[1][3])
        self.assertEqual(question["category"], calls[1][3])

    def test_all_exhausted_actions_finish_cleanly_without_generation_error(self):
        controller, _ = create_controller()
        session = controller.session
        session.policy = FirstLegalPolicy()

        def always_exhaust(*args, **kwargs):
            raise QuestionExhaustedError("simulated exhausted action")

        session.bank.get = always_exhaust
        result = controller.generate_next_question()
        self.assertTrue(result["finished"])
        self.assertIn("No new valid question remains", result["message"])
        self.assertTrue(session.env.done)
        self.assertEqual(session.end_reason, "question_bank_exhausted")
        self.assertIsNone(session.cur)

    def test_eight_turn_interview_keeps_generating_after_repeated_llm_question(self):
        controller, mock = create_controller(max_questions=10)
        controller.session.policy = FirstLegalPolicy()
        mock.repeat = "Describe a relevant technical trade-off and how you would validate it in production."
        for _ in range(8):
            question = controller.generate_next_question()
            self.assertFalse(question["finished"])
            controller.submit_answer(
                "I would measure the impact, test edge cases, and compare results against a baseline."
            )
        asked = [item["question"] for item in controller.question_history]
        self.assertEqual(len(asked), 8)
        self.assertEqual(len(set(asked)), 8)

    def test_gemini_fallback_prefers_lite_then_36_then_38(self):
        self.assertEqual(gemini_client.MODEL_NAMES, [
            "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"
        ])
        first, second = MagicMock(), MagicMock()
        first.models.generate_content.side_effect = gemini_client.errors.ClientError(
            code=404, response_json={"error": {"message": "model endpoint unavailable"}}
        )
        second.models.generate_content.return_value = MagicMock(text="fallback response")
        with patch.object(gemini_client, "_client", side_effect=[first, second]), patch.object(gemini_client.time, "sleep"):
            result = gemini_client.generate_response("hello")
        self.assertEqual(result, "fallback response")
        self.assertEqual(first.models.generate_content.call_args.kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(second.models.generate_content.call_args.kwargs["model"], "gemini-3.6-flash")

    def test_environment_episode_is_reproducible_for_same_seed(self):
        def run_once():
            env = InterviewEnv()
            obs = env.reset(seed=7788)
            done = False
            while not done:
                action = int(np.flatnonzero(obs["m"])[0])
                obs, _, done, info = env.step(action)
            return info["final"]
        self.assertEqual(run_once(), run_once())

    def test_01_checkpoint_loads_and_observation_contract_matches(self):
        policy = load_policy(MODEL)
        env = InterviewEnv()
        obs = env.reset(seed=42)
        self.assertEqual(obs["g"].shape, (8,))
        self.assertEqual(obs["f"].shape, (48, 19))
        self.assertEqual(obs["m"].shape, (48,))
        action = policy(obs, env)
        self.assertTrue(obs["m"][action])

    def test_02_rl_action_conditions_skill_difficulty_type_and_llm_prompt(self):
        controller, mock = create_controller(load_policy(MODEL))
        q = controller.generate_next_question()
        _, skill_id, difficulty, question_type = controller.env.decode(q["action"])
        self.assertEqual(q["skill"], REG.names[skill_id])
        self.assertEqual(q["difficulty"], ("easy", "medium", "hard")[difficulty])
        self.assertEqual(q["question_type"], ("conceptual", "coding")[question_type])
        self.assertIn(f"TARGET SKILL: {q['skill']}", mock.prompts[0])
        self.assertIn(f"TARGET DIFFICULTY: {q['difficulty']}", mock.prompts[0])
        self.assertIn("REQUIRED ASSESSMENT CATEGORY:", mock.prompts[0])

    def test_03_all_question_categories_are_real_engine_outputs(self):
        bank = QuestionBank()
        bank.llm_generate = MockGemini()
        categories = ["resume_based", "skill_based", "theoretical", "scenario_based", "project_based", "follow_up", "hr_behavioral"]
        for category in categories:
            state = {
                "category": category, "resume_text": "Python project with SQL data pipeline",
                "projects": "Built a Python data pipeline", "history": [],
                "last_answer": "We improved throughput with streaming.",
            }
            q = bank.get(REG.idx["Python"], 1, 0, state)
            self.assertEqual(q["category"], category)
            self.assertTrue(q["question"])
        self.assertEqual(len(bank.generation_log), len(categories))

    def test_04_hr_is_selected_by_communication_skill_action(self):
        controller, _ = create_controller(FixedActionPolicy(skill="Communication & Leadership", question_type=0))
        q = controller.generate_next_question()
        self.assertEqual(q["skill"], "Communication & Leadership")
        self.assertEqual(q["category"], "hr_behavioral")

    def test_05_coding_generation_includes_executable_sample_and_hidden_tests(self):
        bank = QuestionBank()
        bank.llm_generate = MockGemini()
        q = bank.get(REG.idx["Python"], 1, 1, {"category": "coding", "history": []})
        self.assertEqual(q["category"], "coding")
        self.assertTrue(q["sample_tests"])
        self.assertGreater(len(q["tests"]), len(q["sample_tests"]))
        self.assertEqual(run_code_tests(q["reference_answer"], q["tests"])[0], 1.0)

    def test_06_coding_evaluation_executes_code_and_uses_reasoning(self):
        bank = QuestionBank()
        bank.llm_generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("mock Gemini outage"))
        q = bank.get(REG.idx["Python"], 0, 1, {"category": "coding", "history": []})
        evaluator = AnswerEvaluator(use_llm=False)
        result = evaluator.score(q, q["reference_answer"], reasoning="The algorithm is linear time because it processes each token once and handles edge cases.")
        self.assertEqual(result["code_score"], 1.0)
        self.assertGreaterEqual(result["logic_score"], 0.9)
        self.assertGreaterEqual(result["score"], 0.99)

    def test_06b_coding_runner_rejects_imports(self):
        score, result = run_code_tests("import os\ndef solve(): return 1", [["solve()", 1]])
        self.assertEqual(score, 0.0)
        self.assertIn("Imports are disabled", result["error"])

    def test_06c_coding_logic_uses_mocked_gemini_after_execution(self):
        q = {"question": "Implement solve(values).", "tests": [["solve([1,2])", 2]], "reference_answer": "def solve(values):\n    return len(values)", "expected_evidence": ["linear scan"]}
        payload = {"logic_score": 0.8}
        with patch("llm.gemini_client.generate_response", return_value=json.dumps(payload)):
            result = AnswerEvaluator(use_llm=True).score(q, q["reference_answer"], reasoning="A linear scan is enough.")
        self.assertEqual(result["code_score"], 1.0)
        self.assertEqual(result["logic_score"], 0.8)
        self.assertAlmostEqual(result["score"], 0.97)

    def test_07_resume_parser_enriches_grounded_facts_with_mock_gemini(self):
        text = "Skills: Python, SQL\nProjects\nBuilt a Python reporting pipeline"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            profile = build_candidate_profile(path, profile_generator=lambda prompt: json.dumps({"skills": ["Python", "Rust"], "projects": ["Built a Python reporting pipeline"], "summary": "Built a reporting pipeline."}))
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertIn("Python", profile["skills"])
        self.assertNotIn("Rust", profile["skills"])
        self.assertEqual(profile["profile_source"], "gemini_enriched")

    def test_08_resume_enrichment_failure_uses_local_parser(self):
        profile = extract_resume_profile("Python developer", {"skills": ["Python"]}, generator=lambda _: 1 / 0)
        self.assertEqual(profile["profile_source"], "local_parser_fallback")
        self.assertEqual(profile["skills"], ["Python"])

    def test_09_exact_and_semantic_duplicate_detection(self):
        history = [{"question": "How did you improve Python performance in your data pipeline?"}]
        self.assertTrue(is_duplicate("How did you improve Python performance in your data pipeline?", history))
        self.assertTrue(is_duplicate("How did you improve Python performance within your data pipeline?", history))
        self.assertTrue(is_duplicate("What did you do to make your Python data pipeline faster?", history))
        self.assertFalse(is_duplicate("Which Python edge cases did your tests cover?", history))

    def test_10_duplicate_gemini_responses_fall_back_to_new_question(self):
        bank = QuestionBank()
        mock = MockGemini()
        mock.repeat = "How do you design a Python system for reliability under load?"
        bank.llm_generate = mock
        first = bank.get(REG.idx["Python"], 1, 0, {"category": "skill_based", "history": []})
        history = [{"question": first["question"], "category": "skill_based"}]
        second = bank.get(REG.idx["Python"], 1, 0, {"category": "skill_based", "history": history})
        self.assertFalse(is_duplicate(second["question"], history))
        self.assertEqual(len(mock.prompts), 3)
        self.assertIn("EXACT QUESTIONS THAT MUST NOT BE REPEATED", mock.prompts[-1])

    def test_11_same_skill_follow_up_can_probe_a_new_aspect(self):
        agent = CategoryRLAgent()
        decision = agent.decide("Python", 0, {
            "history": [{"skill": "Python", "category": "skill_based", "question": "How did you optimize it?", "answer": "We used caching."}],
            "last_answer": "We used caching.", "last_score": 0.35, "followup_skills": [], "resume_text": "Python", "projects": "",
        })
        self.assertIn(decision["category"], decision["legal_categories"])
        self.assertEqual(decision["policy"], "category_q_policy")
        self.assertEqual(decision["category"], max(decision["q_values"], key=decision["q_values"].get))
        coding = agent.decide("Python", 1, {})
        behavioral = agent.decide("Communication & Leadership", 0, {})
        self.assertEqual(coding["category"], "coding")
        self.assertEqual(behavioral["category"], "hr_behavioral")

    def test_12_question_history_persists_answer_and_evaluation_between_turns(self):
        controller, mock = create_controller()
        q1 = controller.generate_next_question()
        self.assertEqual(q1["category_policy_decision"]["policy"], "category_q_policy")
        controller.submit_answer("I profiled the system, measured throughput and chose streaming due to memory limits.")
        q2 = controller.generate_next_question()
        self.assertEqual(len(controller.question_history), 2)
        self.assertEqual(controller.question_history[0]["category_policy"], "category_q_policy")
        self.assertTrue(controller.question_history[0]["category_policy_q_values"])
        self.assertTrue(controller.question_history[0]["answer"])
        self.assertIsNotNone(controller.question_history[0]["evaluation"])
        self.assertIn(controller.question_history[0]["answer"], mock.prompts[-1])
        self.assertNotEqual(q1["question"], q2["question"])

    def test_13_policy_action_adapts_difficulty_and_mask(self):
        controller, _ = create_controller(FixedActionPolicy(skill="Python", difficulty_sequence=[0, 2]))
        first = controller.generate_next_question()
        self.assertEqual(first["difficulty"], "easy")
        controller.submit_answer("A useful explanation with evidence and practical trade-offs.")
        second = controller.generate_next_question()
        self.assertEqual(second["difficulty"], "hard")

    def test_14_bkt_and_irt_both_update_and_remain_distinct(self):
        controller, _ = create_controller(FixedActionPolicy(skill="Python"))
        idx = REG.idx["Python"]
        before_irt = controller.env.belief.mu[idx]
        before_bkt = controller.session.bkt.get_mastery("Python")
        q = controller.generate_next_question()
        controller.submit_answer("Python uses iterators and generators to stream data with memory efficient processing.")
        self.assertNotEqual(controller.env.belief.mu[idx], before_irt)
        self.assertNotEqual(controller.session.bkt.get_mastery("Python"), before_bkt)
        self.assertEqual(len(controller.env.belief.p[idx]), 71)

    def test_15_knowledge_graph_propagates_irt_evidence(self):
        py = REG.idx["Python"]
        ml = REG.idx["Machine Learning"]
        self.assertGreater(REG.W[py, ml], 0)
        belief = Belief(np.zeros(REG.N), np.full(REG.N, 1.3))
        before = belief.mu[ml]
        belief.update(py, 1.0, item_prob_table()[1, 0], W=REG.W, rho=0.5)
        self.assertNotEqual(belief.mu[ml], before)
        communication = REG.idx["Communication & Leadership"]
        self.assertEqual(REG.W[communication, py], 0.0)

    def test_16_interview_budget_termination_and_final_reporting(self):
        controller, _ = create_controller(max_questions=1)
        q = controller.generate_next_question()
        controller.submit_answer("A complete practical response with enough details.")
        self.assertTrue(controller.is_finished())
        report = controller.get_report()
        self.assertIn("bkt_mastery", report.columns)
        self.assertIn("ci95_lo", report.columns)
        self.assertEqual(controller.question_history[0]["question_id"], q["id"])

    def test_17_practice_hiring_and_research_share_controller_engine(self):
        for mode in ("practice", "hiring", "research"):
            controller, _ = create_controller(mode=mode)
            q = controller.generate_next_question()
            self.assertEqual(controller.mode, mode)
            self.assertTrue(q["question"])
            self.assertIs(controller.env, controller.session.env)

    def test_18_synthetic_training_keeps_mastery_hidden_from_observation(self):
        env = InterviewEnv()
        obs = env.reset(seed=11)
        self.assertIsNotNone(env.theta)
        self.assertNotIn("theta", obs)
        self.assertEqual(obs["g"].shape, (8,))

    def test_19_heldout_research_evaluation_runs(self):
        rows = evaluate_heldout(load_policy(MODEL), episodes=2, seed=100)
        self.assertEqual(len(rows), 2)
        self.assertIn("weighted_rmse", rows[0])
        self.assertIn("questions", rows[0])

    def test_19b_ablation_and_noise_studies_run(self):
        policy = load_policy(MODEL)
        ablation = evaluate_ablation(policy, episodes=1, seed=200)
        noise = evaluate_noise_robustness(policy, episodes=1, seed=300)
        self.assertEqual(len(ablation), 6)
        self.assertEqual(len(noise), 4)
        self.assertIn("Without Knowledge Graph", [row["variant"] for row in ablation])

    def test_20_gemini_question_failure_uses_compatible_fallback(self):
        bank = QuestionBank()
        bank.llm_generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("mock unavailable"))
        q = bank.get(REG.idx["Python"], 0, 0, {"category": "theoretical", "history": []})
        self.assertIn(q["source"], {"curated_fallback", "research_question_bank"})
        self.assertEqual(q["category"], "theoretical")
        self.assertTrue(q["question"])

    def test_21_gemini_answer_failure_uses_category_aware_fallback(self):
        q = {"question": "Describe a team conflict.", "category": "hr_behavioral", "expected_evidence": ["communication"]}
        evaluator = AnswerEvaluator(use_llm=True)
        with patch("llm.gemini_client.generate_response", side_effect=RuntimeError("mock unavailable")):
            result = evaluator.score(q, "The situation was a deadline conflict. My task was to lead the team. I communicated, took action, and the result was a successful delivery.")
        self.assertTrue(result["needs_human_review"])
        self.assertIn("behavioural", result["practical_reasoning"].lower())

    def test_22_gemini_answer_response_is_structured_and_used(self):
        q = {"question": "How would you debug a query?", "category": "scenario_based", "skill": "SQL", "expected_evidence": ["query plan"]}
        payload = {"score": 0.8, "evaluator_confidence": 0.9, "feedback": "Good reasoning.", "missing_concepts": [], "status": "correct"}
        with patch("llm.gemini_client.generate_response", return_value=json.dumps(payload)):
            result = AnswerEvaluator(use_llm=True).score(q, "I would inspect the query plan and indexes.")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["feedback"], "Good reasoning.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
