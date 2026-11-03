"""Offline end-to-end demonstration with a mocked Gemini response generator."""

import json
import re
import tempfile
from pathlib import Path

from core.action_net import load_policy, to_t
from interview.live_controller import LiveInterviewController
from resume.resume_parser import build_candidate_profile


class CoveragePolicy:
    """For demonstration only: masks actions by stage, then lets trained PPO rank them."""
    def __init__(self, scorer):
        self.scorer = scorer
        self.turn = 0
        self.seen_skills = set()

    def __call__(self, obs, env):
        g, f, m = to_t(obs)
        with __import__("torch").no_grad():
            logits = self.scorer.net(g, f, m)[0][0].numpy()
        allowed = []
        for action in range(len(obs["m"])):
            if not obs["m"][action]:
                continue
            _, skill_id, _, question_type = env.decode(action)
            skill = env.reg.names[skill_id]
            if self.turn in (0, 1):
                ok = skill != "Communication & Leadership" and question_type == 0
                if self.turn == 1:
                    ok = ok and skill not in self.seen_skills
            elif self.turn == 2:
                ok = skill == "Communication & Leadership" and question_type == 0
            else:
                ok = question_type == 1
            if ok:
                allowed.append(action)
        if not allowed:
            allowed = [int(x) for x in __import__("numpy").flatnonzero(obs["m"])]
        chosen = max(allowed, key=lambda a: logits[a])
        _, skill_id, _, _ = env.decode(chosen)
        self.seen_skills.add(env.reg.names[skill_id])
        self.turn += 1
        return chosen


def main():
    calls = []

    def mock_gemini(prompt):
        calls.append(prompt)
        category = re.search(r"REQUIRED ASSESSMENT CATEGORY: ([a-z_]+)", prompt).group(1)
        skill = re.search(r"TARGET SKILL: (.+)", prompt).group(1).strip()
        if category == "coding":
            return json.dumps({
                "category": "coding",
                "question": f"Implement `solve(values)` for this {skill} challenge and return the number of values.",
                "reference_answer": "def solve(values):\n    return len(values)",
                "expected_evidence": ["Correct implementation", "Edge-case handling"],
                "evaluation_criteria": "Executable hidden tests and reasoning quality.",
                "sample_tests": [["solve([1, 2])", 2]],
                "tests": [["solve([1, 2])", 2], ["solve([])", 0], ["solve([1, 2, 3])", 3]],
            })
        return json.dumps({
            "category": category,
            "question": f"In a {category} assessment for {skill}, describe one decision you made and the evidence that supported it.",
            "expected_evidence": ["specific example", "reasoned decision", "outcome"],
            "evaluation_criteria": "Evidence, reasoning and relevance to the selected category.",
            "reference_answer": "A concrete example with a decision and measurable outcome.",
            "tests": None,
        })

    resume_text = """Jordan Lee — Software Engineer
Skills: Python, SQL, DSA
Projects: Built a Python reporting pipeline using SQL and optimized batch processing.
Experience: Coordinated a small team, resolved a deadline conflict, and delivered the system.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as file:
        file.write(resume_text)
        resume_path = Path(file.name)
    try:
        profile = build_candidate_profile(
            resume_path,
            profile_generator=lambda prompt: json.dumps({
                "skills": ["Python", "SQL", "DSA"],
                "projects": ["Built a Python reporting pipeline using SQL and optimized batch processing."],
                "responsibilities": ["Coordinated a small team, resolved a deadline conflict, and delivered the system."],
                "education": "",
                "summary": "Software engineer with reporting pipeline experience.",
            }),
        )
    finally:
        resume_path.unlink(missing_ok=True)

    scorer = load_policy(Path(__file__).resolve().parent / "models" / "adaptive_ppo_seed1.pt")
    controller = LiveInterviewController(
        profile, "Software engineer: Python, SQL, data pipelines", max_questions=8,
        policy=CoveragePolicy(scorer),
    )
    controller.bank.llm_generate = mock_gemini
    controller.evaluator.use_llm = False

    print("Demo note: Gemini outputs are mocked; the selected PPO network scores every legal action, with stage masks only to demonstrate coverage.")
    for _ in range(4):
        q = controller.generate_next_question()
        print(f"\nRL action {q['action']} -> {q['skill']} / {q['difficulty']} / {q['question_type']} -> {q['category']}")
        print(f"LLM prompt conditions: {q['skill']}, {q['difficulty']}, {q['question_type']}, {q['category']}")
        print(f"Question: {q['question']}")
        if q["category"] == "coding":
            answer = q["reference_answer"]
            result = controller.submit_answer(answer, reasoning="Linear-time approach because it handles the empty input edge case.")
        else:
            if q["category"] == "hr_behavioral":
                answer = "The situation was a tight delivery deadline. My task was to coordinate the team; I communicated, took action, and the result was a tested delivery."
            else:
                answer = "A specific example was a Python data pipeline. My reasoned decision was to use chunked processing because profiling showed memory pressure; the outcome was a stable, tested delivery."
            result = controller.submit_answer(answer)
        print(f"Evaluation: {result['score']:.2f} · {result['evaluation'].get('feedback')}")

    print(f"\nSession transcript: {len(controller.question_history)} turns; {len(calls)} mocked Gemini prompts.")
    print(controller.get_report()[["skill", "competency", "ci95_lo", "ci95_hi", "bkt_mastery"]].to_string(index=False))


if __name__ == "__main__":
    main()
