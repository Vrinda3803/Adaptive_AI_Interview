import time
import numpy as np

from core.knowledge_graph import REG
from core.bayesian_irt import DIFF_NAMES, TYPE_NAMES
from core.action_net import ActionNet, Scorer
from core.question_bank import QuestionBank
from core.evaluator import AnswerEvaluator
from core.session import InterviewSession, DEFAULT_TARGET_COMPETENCY
from llm.gemini_client import generate_response
from pathlib import Path


class LiveInterviewController:
    """
    Live Interview Controller bridging candidate UI interactions with the
    Primary Research Architecture:
    - Driven by 71-grid Bayesian IRT Belief Engine with Knowledge Graph propagation
    - Policy-guided adaptive sequencing via ActionNet PPO
    - Multi-modal question bank (conversational questions, debugging, workplace scenarios, coding sandbox)
    - Evaluator confidence calibration: uncertain evaluations do not distort belief states
    - Adaptive early stopping when confidence target is reached
    - Full competency diagnostics: 95% Confidence Intervals, z-score thresholds, human-review flags
    """

    def __init__(
        self,
        candidate_profile,
        role_text="",
        interview_duration_minutes=20,
        max_questions=10,
        policy=None,
        mode="practice",
        target_competency=DEFAULT_TARGET_COMPETENCY,
    ):
        # Extract text from candidate_profile dict or string
        if isinstance(candidate_profile, dict):
            resume_text = candidate_profile.get("resume_text", "")
            if not resume_text:
                resume_text = f"Skills: {', '.join(candidate_profile.get('skills', []))}\nProjects: {candidate_profile.get('projects', '')}\nExperience: {candidate_profile.get('experience', '')}"
            self.profile_dict = candidate_profile
        else:
            resume_text = str(candidate_profile)
            self.profile_dict = {"resume_text": resume_text}

        self.resume_text = resume_text
        self.role_text = role_text
        self.max_questions = max_questions
        self.interview_duration_seconds = interview_duration_minutes * 60
        self.interview_start_time = time.monotonic()
        self.time_remaining_minutes = float(interview_duration_minutes)

        # 1. Initialize RL Policy Scorer
        if policy is not None:
            self.policy = policy
        else:
            net = ActionNet()
            self.policy = Scorer(net, "ppo", "PPO")

        # 2. Initialize Research QuestionBank and Evaluator
        self.bank = QuestionBank(reg=REG, cache_path=str(Path(__file__).resolve().parents[1] / "question_cache.json"))
        self.evaluator = AnswerEvaluator(use_llm=True)

        # 3. Initialize Primary Research InterviewSession (no hardcoded senior level)
        self.session = InterviewSession(
            policy=self.policy,
            bank=self.bank,
            evaluator=self.evaluator,
            resume_text=self.resume_text,
            role_text=self.role_text,
            budget=self.max_questions,
            reg=REG,
            target_competency=target_competency,
            mode=mode,
            candidate_profile=self.profile_dict,
        )

        self.current_question = None
        self.current_q_data = None
        self.current_action = None
        self.current_evaluation = None
        self.finished = False

        # Compatibility bridge for existing legacy references
        self.env = self.session.env
        self.mode = mode

    @property
    def question_history(self):
        """Full session transcript metadata used for deduplication and audit."""
        return self.session.question_history

    @property
    def questions_asked(self):
        return self.session.env.t

    def update_timer(self):
        elapsed = time.monotonic() - self.interview_start_time
        remaining_seconds = max(0.0, self.interview_duration_seconds - elapsed)
        self.time_remaining_minutes = remaining_seconds / 60.0
        return remaining_seconds

    def is_finished(self):
        self.update_timer()
        return (
            self.session.env.done
            or self.session.env.t >= self.max_questions
            or self.time_remaining_minutes <= 0.0
        )

    def generate_hint(self):
        """Generate a realistic coaching hint using Gemini."""
        if not self.current_question:
            return "No active question."
        prompt = f"""
        You are a friendly technical interviewer giving a helpful hint during a live interview.
        The candidate is asking for guidance on this question:
        "{self.current_question}"

        Provide ONE short, conversational hint (1-2 sentences) that nudges their thinking in the right direction
        without giving away the final solution.
        """
        try:
            return generate_response(prompt)
        except Exception:
            return "Consider the edge cases and how data flow behaves between memory and the processor."

    def generate_next_question(self):
        """Select next adaptive question using PPO policy and Bayesian IRT state."""
        self.update_timer()
        if self.is_finished():
            self.finished = True
            reason = getattr(self.session, "end_reason", None)
            if reason == "confidence_target":
                msg = "The confidence target was reached after the minimum question count."
            elif reason == "question_budget":
                msg = "The configured question budget has been reached."
            elif reason in {"question_bank_exhausted", "legal_actions_exhausted"}:
                msg = "No new valid question remains for the selected skills. Please review the competency report and adjust the skills or question budget before another session."
            else:
                msg = "The configured interview session has ended."
            return {"finished": True, "message": msg, "stopped_early": self.session.env.stopped_early}

        q_data = self.session.next_question()
        if q_data is None:
            self.finished = True
            reason = getattr(self.session, "end_reason", None)
            msg = (
                "No new valid question remains for the selected skills. Please review the competency report and adjust the skills or question budget before another session."
                if reason == "question_bank_exhausted"
                else "All valid assessment actions have been used."
            )
            return {
                "finished": True,
                "message": msg,
                "stopped_early": self.session.env.stopped_early,
            }

        self.current_q_data = q_data
        self.current_question = q_data["question"]
        skill = q_data["skill"]
        diff_name = DIFF_NAMES[q_data["difficulty"]]
        type_name = TYPE_NAMES[q_data["type"]]

        action_idx = self.session.cur[0] if self.session.cur else 0
        self.current_action = action_idx
        category = q_data.get("category", "skill_based")
        category_label = category.replace("_", " ").title()
        strategy_label = f"RL selected: {skill} ({diff_name.title()} • {category_label})"

        result = {
            "finished": False,
            "action": action_idx,
            "id": q_data.get("id", f"turn-{self.questions_asked}"),
            "initial_or_follow_up": "follow_up" if q_data.get("category") == "follow_up" else "initial",
            "strategy": strategy_label,
            "skill": skill,
            "difficulty": diff_name,
            "question_type": type_name,
            "category": category,
            "category_label": category_label,
            "category_policy_decision": q_data.get("category_decision"),
            "question": q_data["question"],
            "tests": q_data.get("tests"),
            "sample_tests": q_data.get("sample_tests"),
            "reference_answer": q_data.get("reference_answer", ""),
            "expected_evidence": q_data.get("expected_evidence", []),
            "source": q_data.get("source", "curated"),
            "hint": None,
        }
        return result

    def submit_answer(self, candidate_answer, reasoning=""):
        """
        Submit candidate response, update Bayesian IRT 71-grid posterior,
        and propagate evidence across Knowledge Graph neighbors.
        """
        if self.current_q_data is None:
            raise ValueError("No active question to answer. Generate a question first.")
        if not candidate_answer.strip():
            raise ValueError("Candidate answer cannot be empty.")

        q_data = self.current_q_data
        skill = q_data["skill"]

        # Submit to research session (updates Belief over 71-point logit grid with evaluator confidence)
        sub_res = self.session.submit(candidate_answer, reasoning=reasoning)

        score = float(sub_res["score"])
        c_eval = float(sub_res.get("evaluator_confidence", 0.85))
        eval_dict = sub_res.get("evaluation", {})
        feedback = sub_res["feedback"]
        mu_before = sub_res["competency_before"]
        mu_after = sub_res["competency_after"]
        sd_before = sub_res["uncertainty_before"]
        sd_after = sub_res["uncertainty_after"]
        stopped_early = sub_res.get("stopped_early", False)

        # Reset current question state
        self.current_question = None
        self.current_q_data = None
        self.update_timer()
        self.finished = self.is_finished()
        self.current_evaluation = eval_dict

        return {
            "evaluation": eval_dict,
            "reasoning": reasoning,
            "score": score,
            "evaluator_confidence": c_eval,
            "feedback": feedback,
            "skill": skill,
            "difficulty": DIFF_NAMES[q_data["difficulty"]],
            "type": TYPE_NAMES[q_data["type"]],
            "category": q_data.get("category", "technical_experience"),
            "old_mastery": mu_before,
            "new_mastery": mu_after,
            "competency_before": mu_before,
            "competency_after": mu_after,
            "uncertainty_before": sd_before,
            "uncertainty_after": sd_after,
            "questions_asked": self.questions_asked,
            "time_remaining": self.time_remaining_minutes,
            "finished": self.finished,
            "stopped_early": stopped_early,
        }

    def get_report(self):
        """Return candidate competency evaluation report as DataFrame."""
        return self.session.report()

    def plot_report(self):
        """Generate Matplotlib figure of candidate competency with 95% CI."""
        return self.session.plot()
