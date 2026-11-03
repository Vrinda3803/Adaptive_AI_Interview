import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .knowledge_graph import REG
from .bayesian_irt import KMAX, DIFF_NAMES, TYPE_NAMES
from .environment import InterviewEnv, EnvCfg
from .bkt import BayesianKnowledgeTracer
from .category_agent import CategoryRLAgent
from .question_bank import QuestionExhaustedError

# Neutral benchmark threshold on the logit scale (0.0 represents solid competent baseline)
DEFAULT_TARGET_COMPETENCY = 0.0


class InterviewSession:
    """
    Live Interview Session driven by the Primary Research Architecture:
    - Extracts candidate competencies from resume & job description using the Knowledge Graph
    - Drives adaptive question sequencing via the trained RL policy
    - Tracks candidate latent competency via 71-grid Bayesian IRT with Knowledge Graph neighbor sharing
    - Calibrates updates using Evaluator Confidence (c_eval) to prevent noise propagation
    - Employs adaptive stopping when target confidence (uncertainty sd <= threshold) is attained
    - Computes 95% Confidence Intervals, z-scores, and human-review flags
    """

    def __init__(
        self,
        policy,
        bank,
        evaluator,
        resume_text,
        role_text="",
        budget=8,
        reg=REG,
        target_competency=DEFAULT_TARGET_COMPETENCY,
        mode="practice",
        candidate_profile=None,
    ):
        self.reg = reg
        self.policy = policy
        self.bank = bank
        self.ev = evaluator
        self.mode = mode
        self.target_competency = target_competency
        self.resume_text = resume_text or ""
        self.role_text = role_text or ""
        candidate_profile = candidate_profile or {}
        self.projects_text = candidate_profile.get("projects", "")
        self.experience_text = candidate_profile.get("experience", "")
        if isinstance(self.projects_text, list):
            self.projects_text = "\n".join(self.projects_text)
        if isinstance(self.experience_text, list):
            self.experience_text = "\n".join(self.experience_text)

        # 1. Skill extraction from resume and JD
        res = reg.extract(resume_text)
        jd = reg.extract(role_text or resume_text)
        counts = {
            k: jd["found"].get(k, 0) + (0.5 * res["found"].get(k, 0) if role_text else 0)
            for k in (jd["found"] if role_text else res["found"])
        }
        # Communication is a real, non-coding assessment dimension. The existing
        # feature-based policy can score it without changing the checkpoint shape.
        if "Communication & Leadership" in reg.idx:
            counts[reg.idx["Communication & Leadership"]] = max(
                counts.get(reg.idx["Communication & Leadership"], 0.0), 0.35
            )

        # Fallback to general skills if fewer than 2 recognized
        if len(counts) < 2:
            default_skills = ["Python", "DSA", "SQL", "Machine Learning"]
            for s in default_skills:
                if s in reg.idx:
                    counts[reg.idx[s]] = 1

        top = sorted(counts, key=lambda k: -counts[k])[:KMAX]
        communication = reg.idx.get("Communication & Leadership")
        if communication is not None and communication not in top:
            top[-1] = communication
        self.unknown = sorted(set(res.get("unknown", []) + jd.get("unknown", [])))

        # 2. Extract resume claim levels (0=beginner, 1=intermediate, 2=expert)
        claim = np.full(reg.N, -1)
        for line in resume_text.splitlines():
            lvl = (
                2
                if re.search(r"expert|advanced|lead|proficient", line, re.I)
                else 0
                if re.search(r"beginner|basic|learning|familiar|novice", line, re.I)
                else 1
            )
            for k in reg.extract(line)["found"]:
                claim[k] = lvl
        for k in top:
            if claim[k] < 0 and k in res.get("found", {}):
                claim[k] = 1

        # 3. Initialize InterviewEnv in real-candidate mode
        cfg = EnvCfg(
            use_graph=True,
            use_eig_feat=True,
            use_sigma_feat=True,
            use_mask=True,
            use_adaptive_stopping=True,
            stop_uncertainty_threshold=0.40,
            min_questions_before_stop=min(int(budget), max(8, len(top) + 2)),
        )
        self.env = InterviewEnv(cfg, reg=reg)
        self.role = top
        self.obs = self.env.reset_custom(
            top,
            [counts[k] for k in top],
            [self.target_competency] * len(top),
            claim,
            int(budget),
        )
        self.cur = None
        self.category_agent = CategoryRLAgent()
        self.asked_stems = {}
        self.evidence = {k: [] for k in top}
        self.history = []
        self.question_history = []
        self.end_reason = None
        self.bkt = BayesianKnowledgeTracer(p_init=0.20, p_learn=0.15, p_guess=0.20, p_slip=0.10)
        self.last_candidate_answer = ""
        self.last_question_text = ""

    def next_question(self):
        if self.env.done:
            return None
        # If an action has no distinct question left, temporarily mask it and
        # let the trained policy choose its next-best legal action.
        available = np.asarray(self.obs["m"], dtype=bool).copy()
        excluded_categories_by_action = {}
        while available.any():
            choice_obs = dict(self.obs)
            choice_obs["m"] = available.copy()
            valid_actions = np.flatnonzero(available)
            try:
                a = int(self.policy(choice_obs, self.env))
            except Exception as exc:
                raise RuntimeError("The RL action policy failed to choose an interview action.") from exc
            if a < 0 or a >= len(available) or not available[a]:
                raise RuntimeError("The RL action policy returned an action outside the current legal-action mask.")

            slot, sk, d, ty = self.env.decode(a)
            is_followup = bool(self.history and self.history[-1].get("skill") == self.reg.names[sk])
            followup_skills = [h["skill"] for h in self.history if h.get("category") == "follow_up"]
            category_decision = self.category_agent.decide(
                self.reg.names[sk], ty,
                {
                    "history": self.history,
                    "followup_skills": followup_skills,
                    "resume_text": self.resume_text,
                    "projects": getattr(self, "projects_text", ""),
                    "last_answer": self.last_candidate_answer,
                    "last_score": self.history[-1].get("score") if self.history else None,
                    "remaining_fraction": max(0.0, (self.env.B - self.env.t) / max(self.env.B, 1)),
                },
                excluded_categories=excluded_categories_by_action.get(a, ()),
            )
            category = category_decision["category"]
            if category is None:
                available[a] = False
                continue

            state = dict(
                mu=float(self.env.belief.mu[sk]),
                sd=float(self.env.belief.sd[sk]),
                asked=self.asked_stems.get(sk, []),
                last_answer=self.last_candidate_answer,
                last_question=self.last_question_text,
                is_followup=is_followup,
                category=category,
                resume_text=self.resume_text,
                projects=getattr(self, "projects_text", ""),
                job_description=self.role_text,
                history=self.question_history,
            )
            try:
                q = self.bank.get(sk, d, ty, state)
                break
            except QuestionExhaustedError:
                excluded_categories_by_action.setdefault(a, set()).add(category)
        else:
            # Complete only when no legal action can produce a distinct
            # question. Keep a clear reason for the candidate-facing UI.
            self.cur = None
            self.env.done = True
            self.end_reason = "question_bank_exhausted"
            return None

        q["category"] = category
        q["category_decision"] = category_decision
        q["skill"] = self.reg.names[sk]
        q["difficulty"] = d
        q["type"] = ty
        q["conversation_history"] = [
            {"question": h.get("question", ""), "answer": h.get("answer", ""), "category": h.get("category", "")}
            for h in self.history
        ]
        self.cur = (a, sk, q)
        self.last_question_text = q["question"]
        self.question_history.append({
            "question_id": q.get("id", ""),
            "normalized_question": q.get("normalized_question", ""),
            "question": q["question"],
            "category": category,
            "skill": self.reg.names[sk],
            "difficulty": DIFF_NAMES[d],
            "question_type": TYPE_NAMES[ty],
            "answer": None,
            "evaluation": None,
            "initial_or_follow_up": "follow_up" if category == "follow_up" else "initial",
            "action": int(a),
            "category_policy": category_decision["policy"],
            "category_policy_state": category_decision["state_id"],
            "category_policy_q_values": category_decision["q_values"],
        })
        return q

    def submit(self, answer, reasoning=""):
        if self.cur is None:
            raise ValueError("No active question to submit an answer for.")
        a, sk, q = self.cur
        eval_dict = self.ev.score(q, answer, reasoning=reasoning)

        score = float(eval_dict.get("score", 0.5))
        c_eval = float(eval_dict.get("evaluator_confidence", 0.85))
        fb = eval_dict.get("feedback", "Evaluation complete.")
        bkt_mastery = self.bkt.update_from_score(self.reg.names[sk], score)

        mu0 = float(self.env.belief.mu[sk])
        sd0 = float(self.env.belief.sd[sk])

        # Step environment with reliability-calibrated Bayesian update
        self.obs, r, done, info = self.env.step(a, score=score, evaluator_confidence=c_eval)
        if done:
            if info.get("stopped_early"):
                self.end_reason = "confidence_target"
            elif self.env.t >= self.env.B:
                self.end_reason = "question_budget"
            else:
                self.end_reason = "legal_actions_exhausted"

        mu1 = float(self.env.belief.mu[sk])
        sd1 = float(self.env.belief.sd[sk])

        self.last_candidate_answer = answer
        self.asked_stems.setdefault(sk, []).append(q["question"][:80])

        record = dict(
            question_id=q.get("id", ""),
            normalized_question=q.get("normalized_question", ""),
            question=q["question"],
            category=q.get("category", "technical_experience"),
            category_policy=q.get("category_decision", {}).get("policy", "category_q_policy"),
            category_policy_state=q.get("category_decision", {}).get("state_id"),
            category_policy_q_values=q.get("category_decision", {}).get("q_values", {}),
            skill=self.reg.names[sk],
            difficulty=DIFF_NAMES[q["difficulty"]],
            type=TYPE_NAMES[q["type"]],
            score=score,
            evaluator_confidence=c_eval,
            relevance=eval_dict.get("relevance", 1.0),
            technical_correctness=eval_dict.get("technical_correctness", score),
            feedback=fb,
            evidence_of_understanding=eval_dict.get("evidence_of_understanding", ""),
            missing_concepts=eval_dict.get("missing_concepts", []),
            practical_reasoning=eval_dict.get("practical_reasoning", ""),
            needs_human_review=eval_dict.get("needs_human_review", False),
            status=eval_dict.get("status", "evaluated"),
            source=q.get("source", "curated"),
            answer=answer,
            reasoning=reasoning,
            bkt_mastery=round(bkt_mastery, 4),
            action=int(a),
            mu_before=mu0,
            mu_after=mu1,
            sd_before=sd0,
            sd_after=sd1,
            stopped_early=info.get("stopped_early", False),
        )
        self.evidence[sk].append(record)
        self.history.append(record)
        if self.question_history:
            self.question_history[-1]["answer"] = answer
            self.question_history[-1]["evaluation"] = eval_dict
            self.question_history[-1]["bkt_mastery"] = round(bkt_mastery, 4)
        self.cur = None

        return dict(
            score=score,
            evaluator_confidence=c_eval,
            feedback=fb,
            competency_before=mu0,
            competency_after=mu1,
            uncertainty_before=sd0,
            uncertainty_after=sd1,
            evaluation=eval_dict,
            done=done,
            stopped_early=info.get("stopped_early", False),
        )

    def report(self):
        """Construct candidate competency evaluation report."""
        b = self.env.belief
        rows = []
        for slot, k in enumerate(self.role):
            mu, sd = float(b.mu[k]), float(b.sd[k])
            req = float(self.env.req[slot])
            ev = self.evidence[k]
            claimed = int(self.env.claim[k])
            z = (mu - req) / max(sd, 1e-6)
            verdict = "meets" if z > 1.0 else "below" if z < -1.0 else "borderline"
            review = (
                (sd > 0.55)
                or (abs(z) < 0.6)
                or (claimed == 2 and mu < 0)
                or (len(ev) == 0)
                or any(e.get("needs_human_review", False) for e in ev)
            )
            rows.append(
                dict(
                    skill=self.reg.names[k],
                    competency=round(mu, 2),
                    ci95_lo=round(mu - 1.96 * sd, 2),
                    ci95_hi=round(mu + 1.96 * sd, 2),
                    uncertainty_sd=round(sd, 2),
                    required=round(req, 2),
                    verdict=verdict,
                    questions=len(ev),
                    mean_score=round(float(np.mean([e["score"] for e in ev])), 2) if ev else None,
                    bkt_mastery=round(self.bkt.get_mastery(self.reg.names[k]), 2),
                    needs_human_review=bool(review),
                    resume_claim=claimed,
                )
            )
        return pd.DataFrame(rows)

    def plot(self, ax=None):
        """Plot competency estimates with 95% confidence intervals against requirements."""
        r = self.report()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(r) + 1.5)))
        else:
            fig = ax.figure
        y = np.arange(len(r))
        ax.errorbar(
            r["competency"],
            y,
            xerr=[r["competency"] - r["ci95_lo"], r["ci95_hi"] - r["competency"]],
            fmt="o",
            color="#2563EB",
            ecolor="#60A5FA",
            elinewidth=2,
            capsize=5,
            label="Estimate ± 95% CI",
        )
        ax.scatter(
            r["required"],
            y,
            marker="|",
            s=350,
            color="#DC2626",
            linewidths=3,
            label="Baseline Threshold",
            zorder=5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(r["skill"], fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Latent Competency θ (Logit Scale)")
        ax.set_title("Bayesian IRT Competency Report (Estimate ± 95% CI)", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
