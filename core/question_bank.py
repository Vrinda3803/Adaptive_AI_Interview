import os
import sys
import ast
import json
import re
import tempfile
import subprocess
import hashlib
import datetime
import numpy as np

from .knowledge_graph import REG
from .bayesian_irt import DIFF_NAMES, TYPE_NAMES, DIFF_B
from .assessment import is_duplicate, normalize_question

# =====================================================================
# REALISTIC, CONVERSATIONAL QUESTION BANK
# Structured across real engineering domains:
# (skill, difficulty) -> dict with conversational workplace questions,
# expected evidence, practical scenarios, and evaluation criteria.
# =====================================================================

REALISTIC_QUESTION_BANK = {
    ("Python", 0): {
        "category": "technical_experience",
        "question": "In your Python projects, when do you typically choose a tuple over a list, and what practical advantages did that give your code?",
        "reference_answer": "Tuples are immutable and hashable, making them ideal for fixed records, dictionary keys, and preventing unintended side-effects. They also offer slight memory and performance advantages.",
        "expected_evidence": ["Immutability to protect data", "Hashability for dictionary keys or set members", "Memory/performance efficiency"],
        "evaluation_criteria": "Assesses understanding of mutability trade-offs and Python data structures in production code.",
        "keywords": ["immutable", "hashable", "dict key", "memory", "safety"],
    },
    ("Python", 1): {
        "category": "practical_problem_solving",
        "question": "Imagine you have a background job processing a 5GB log file on an instance with 2GB of RAM. How would you structure your Python script to parse it efficiently without crashing the process?",
        "reference_answer": "Use lazy evaluation via generators or iterating over the file line-by-line using `with open(...)` which yields chunks or lines on demand instead of reading the entire file into memory.",
        "expected_evidence": ["Generator / lazy evaluation using yield", "Iterating line-by-line / streaming chunk", "Memory-efficient streaming over memory loading"],
        "evaluation_criteria": "Assesses real-world memory management and streaming data processing in Python.",
        "keywords": ["generator", "yield", "stream", "chunk", "line by line", "memory"],
    },
    ("Python", 2): {
        "category": "debugging_troubleshooting",
        "question": "Suppose your team deployed a multithreaded Python service to compute heavy image embeddings, but CPU utilization maxed out at exactly one core. How would you diagnose this and what architectural change would you make?",
        "reference_answer": "This is caused by CPython's Global Interpreter Lock (GIL), which prevents concurrent bytecode execution for CPU-bound threads. The solution is migrating to multiprocessing, ProcessPoolExecutor, or offloading computation to native C/Rust extensions.",
        "expected_evidence": ["Identifying the GIL as bottleneck for CPU-bound tasks", "Distinguishing CPU-bound vs I/O-bound concurrency", "Transitioning to multiprocessing or ProcessPoolExecutor"],
        "evaluation_criteria": "Assesses concurrency architecture and deep understanding of the CPython runtime.",
        "keywords": ["gil", "global interpreter lock", "multiprocessing", "cpu-bound", "concurrency"],
    },
    ("DSA", 0): {
        "category": "design_decision_making",
        "question": "Can you share a situation where you had to choose between a stack and a queue in an application feature, like an undo action or a task dispatcher?",
        "reference_answer": "An undo feature relies on LIFO (Stack) to reverse recent actions first. A background job dispatcher or print queue requires FIFO (Queue) to process items in arrival order.",
        "expected_evidence": ["LIFO behavior for undo/backtrack", "FIFO behavior for fair task dispatching", "Clear application context"],
        "evaluation_criteria": "Assesses practical application of fundamental data structures to system features.",
        "keywords": ["lifo", "fifo", "stack", "queue", "order", "undo", "buffer"],
    },
    ("DSA", 1): {
        "category": "practical_problem_solving",
        "question": "If your key-value cache starts experiencing frequent hash collisions under high write loads, how does that degrade read performance, and how would you resolve it?",
        "reference_answer": "Hash collisions degrade lookup complexity from average O(1) toward O(n) as buckets degenerate into linked lists or probe chains. Mitigations include resizing (lowering load factor), improving hash distribution, or robin-hood hashing.",
        "expected_evidence": ["O(1) degrading toward O(n)", "Bucket chaining or open addressing explanation", "Resizing / adjusting load factor or hash function"],
        "evaluation_criteria": "Assesses performance optimization and internal mechanics of hash tables.",
        "keywords": ["collision", "chaining", "load factor", "resize", "o(1)", "o(n)", "bucket"],
    },
    ("DSA", 2): {
        "category": "design_decision_making",
        "question": "In a road routing or microservice dependency resolution engine, when would Dijkstra's algorithm fail to find the shortest path, and what algorithm would you switch to?",
        "reference_answer": "Dijkstra fails when edge weights can be negative, because its greedy assumption of finality is broken. In such cases, the Bellman-Ford algorithm or SPFA is required.",
        "expected_evidence": ["Negative edge weights break greedy assumption", "Identification of Bellman-Ford or topological DP for DAGs", "Understanding shortest-path algorithmic constraints"],
        "evaluation_criteria": "Assesses advanced graph algorithms and trade-off analysis under constraint violations.",
        "keywords": ["negative", "bellman-ford", "greedy", "relaxation", "cycle"],
    },
    ("SQL", 0): {
        "category": "practical_problem_solving",
        "question": "Suppose your manager asks for a report of departments with total sales above $100,000, but only counting transactions from this year. How do you separate the filtering between WHERE and HAVING in your query?",
        "reference_answer": "Use WHERE to filter individual transaction rows by date (this year) before aggregation, and use HAVING to filter the aggregated group totals (SUM(sales) > 100,000).",
        "expected_evidence": ["WHERE filters rows before GROUP BY", "HAVING filters aggregated metrics after GROUP BY", "Proper logical SQL execution order"],
        "evaluation_criteria": "Assesses practical query construction and aggregation mechanics.",
        "keywords": ["where", "having", "aggregate", "group by", "filter", "before", "after"],
    },
    ("SQL", 1): {
        "category": "debugging_troubleshooting",
        "question": "You ran an INNER JOIN between an orders table and a customers table, but half of your expected orders are missing from the dashboard. What is the most likely cause, and how do you fix it?",
        "reference_answer": "Orders with NULL or missing customer IDs are dropped by an INNER JOIN. Changing to a LEFT JOIN ensures all orders are retained while showing NULL for missing customer attributes.",
        "expected_evidence": ["INNER JOIN requires matching foreign keys", "LEFT JOIN retains all primary rows", "Handling missing or null relational references"],
        "evaluation_criteria": "Assesses debugging of relational joins and data integrity issues.",
        "keywords": ["left join", "inner join", "missing", "null", "foreign key", "match"],
    },
    ("SQL", 2): {
        "category": "design_decision_making",
        "question": "When would you reach for a window function like ROW_NUMBER() or DENSE_RANK() instead of a simple GROUP BY? Can you walk me through an example from your work?",
        "reference_answer": "Window functions compute rankings or running totals across partitions without collapsing individual rows into summary rows, e.g., finding the top 3 highest revenue orders per customer.",
        "expected_evidence": ["Retaining row-level granularity without collapsing", "Partitioning and ordering within windows", "Practical use case like top-N per category or running totals"],
        "evaluation_criteria": "Assesses advanced SQL reporting and analytical database architecture.",
        "keywords": ["window function", "partition by", "order by", "row_number", "rank", "collapse"],
    },
    ("Docker", 0): {
        "category": "technical_experience",
        "question": "How do you explain the difference between a Docker image and a running container to a new engineer joining your team?",
        "reference_answer": "An image is an immutable, read-only template with application code and dependencies; a container is a running, stateful instance with an active writable layer and isolated process namespace.",
        "expected_evidence": ["Image as blueprint/template", "Container as running instance", "Writable layer and process isolation"],
        "evaluation_criteria": "Assesses foundational containerization principles and technical communication.",
        "keywords": ["image", "container", "template", "instance", "read-only", "writable"],
    },
    ("Docker", 1): {
        "category": "debugging_troubleshooting",
        "question": "Your CI build pipeline is taking 15 minutes because Docker rebuilds npm/pip packages on every single code commit. How would you reorder your Dockerfile to take advantage of build caching?",
        "reference_answer": "Copy dependency lockfiles (package.json / requirements.txt) and run the install step BEFORE copying the rest of the application source code. That way, cached layers are reused unless dependencies change.",
        "expected_evidence": ["Layer caching mechanism", "Copying dependency definitions before source files", "Isolating frequently-changing steps to the bottom"],
        "evaluation_criteria": "Assesses Dockerfile optimization and CI/CD build acceleration.",
        "keywords": ["cache", "layer", "order", "requirements.txt", "copy", "rebuild"],
    },
    ("Docker", 2): {
        "category": "design_decision_making",
        "question": "Under the hood on Linux, what kernel primitives give containers their process isolation and resource limits, and how does this differ fundamentally from virtual machines?",
        "reference_answer": "Containers leverage Linux namespaces (PID, NET, MNT, IPC) for visibility isolation and cgroups for CPU/memory resource limits. They share the host kernel, whereas VMs run distinct guest kernels atop a hypervisor.",
        "expected_evidence": ["Namespaces for isolation (process, network, mounts)", "Cgroups for resource accounting and limits", "Shared host kernel vs hypervisor/guest OS in VMs"],
        "evaluation_criteria": "Assesses operating-system level systems engineering and container security understanding.",
        "keywords": ["namespace", "cgroup", "kernel", "hypervisor", "isolation", "vm"],
    },
    ("Machine Learning", 0): {
        "category": "practical_problem_solving",
        "question": "During training, your model achieves 99% accuracy on training data but drops to 68% on your validation set. How do you interpret this, and what are your first three steps to fix it?",
        "reference_answer": "The model is overfitting—memorizing training noise rather than generalizable patterns. Solutions include adding regularization (L1/L2, dropout), gathering/augmenting data, reducing model capacity, or applying early stopping.",
        "expected_evidence": ["Diagnosing overfitting / high variance", "Adding regularization (dropout, weight decay)", "Data augmentation or early stopping"],
        "evaluation_criteria": "Assesses diagnosis of generalization error in production ML workflows.",
        "keywords": ["overfitting", "validation gap", "regularization", "dropout", "data augmentation"],
    },
    ("Machine Learning", 1): {
        "category": "design_decision_making",
        "question": "How do you navigate the bias-variance trade-off when selecting between a simpler linear model and a complex ensemble or neural network for a customer churn problem?",
        "reference_answer": "Simpler models have higher bias but lower variance and superior interpretability/speed. Complex ensembles reduce bias and capture non-linear interactions at the cost of higher variance and potential overfitting, demanding rigorous cross-validation.",
        "expected_evidence": ["Bias vs variance relationship", "Interpretability vs complexity considerations", "Validation strategy to prevent overfitting"],
        "evaluation_criteria": "Assesses practical model selection and business trade-off reasoning.",
        "keywords": ["bias", "variance", "interpretability", "ensemble", "trade-off", "complexity"],
    },
    ("Machine Learning", 2): {
        "category": "debugging_troubleshooting",
        "question": "Why do we avoid using Mean Squared Error as the loss function for binary classification with a sigmoid output, and what mathematical problem does cross-entropy solve here?",
        "reference_answer": "MSE with sigmoid yields a non-convex optimization surface with vanishing gradients when predictions saturate far from targets. Binary cross-entropy produces gradients directly proportional to prediction error and matches maximum likelihood estimation.",
        "expected_evidence": ["Vanishing gradients / saturated sigmoid", "Non-convex optimization landscape with MSE", "Log-loss / cross-entropy gives steeper, well-behaved gradients"],
        "evaluation_criteria": "Assesses mathematical intuition behind loss functions and gradient-based optimization.",
        "keywords": ["cross-entropy", "mse", "gradient", "saturate", "sigmoid", "convex", "loss"],
    },
}

# Practical, realistic coding challenges: (skill, difficulty) -> dict
CODING = {
    ("Python", 0): dict(
        q="In many text preprocessing pipelines, we need to clean and reverse token order. Write a Python function `reverse_words(s)` that takes a string of words and returns them in reverse order, joined by single spaces.",
        ref="def reverse_words(s):\n    return ' '.join(s.strip().split()[::-1])",
        tests=[
            ["reverse_words('a b c')", "c b a"],
            ["reverse_words('hello world')", "world hello"],
            ["reverse_words('  fast   and   furious ')", "furious and fast"],
        ],
    ),
    ("Python", 1): dict(
        q="In real-time trading or financial ledgers, we need to locate matching pairs quickly. Write a Python function `two_sum(nums, target)` that returns the two indices whose values sum to the target in O(n) time.",
        ref="def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return sorted([seen[target - x], i])\n        seen[x] = i\n    return []",
        tests=[
            ["two_sum([2,7,11,15], 9)", [0, 1]],
            ["two_sum([3,2,4], 6)", [1, 2]],
            ["two_sum([3,3], 6)", [0, 1]],
        ],
    ),
    ("Python", 2): dict(
        q="When managing server maintenance schedules or calendar bookings, overlapping windows must be consolidated. Write `merge_intervals(intervals)` that merges all overlapping [start, end] ranges and returns them sorted.",
        ref="def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda x: x[0])\n    merged = [intervals[0]]\n    for current in intervals[1:]:\n        prev = merged[-1]\n        if current[0] <= prev[1]:\n            prev[1] = max(prev[1], current[1])\n        else:\n            merged.append(current)\n    return merged",
        tests=[
            ["merge_intervals([[1,3],[2,6],[8,10],[15,18]])", [[1, 6], [8, 10], [15, 18]]],
            ["merge_intervals([[1,4],[4,5]])", [[1, 5]]],
            ["merge_intervals([])", []],
        ],
    ),
}


class QuestionExhaustedError(RuntimeError):
    """Raised when one action has no non-duplicate item left in its bank."""


FALLBACK_ASPECTS = (
    "trade-offs", "edge cases", "performance", "failure recovery", "testing strategy",
    "scalability", "security", "observability", "data quality", "backward compatibility",
    "deployment and rollback", "concurrency", "latency constraints", "maintainability",
    "incident response", "resource limits", "validation evidence", "migration risk",
)

RUNNER = """
import json, sys
ns = {}
try:
    code = open(sys.argv[1], encoding='utf-8').read()
    tests = json.load(open(sys.argv[2], encoding='utf-8'))
    exec(compile(code, "<submission>", "exec"), ns)
except BaseException as e:
    print(json.dumps({"passed": 0, "total": len(tests) if 'tests' in locals() else 1, "error": repr(e)}))
    sys.exit(0)

passed = 0
for expr, exp in tests:
    try:
        if eval(expr, ns) == exp:
            passed += 1
    except BaseException:
        pass
print(json.dumps({"passed": passed, "total": len(tests)}))
"""


def run_code_tests(code, tests, timeout=5):
    """Isolated subprocess execution of candidate code with timeout."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports are disabled in the coding runner")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Private/dunder attribute access is disabled")
            if isinstance(node, ast.Name) and (
                node.id.startswith("__")
                or node.id in {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "help", "quit", "exit"}
            ):
                raise ValueError(f"Use of {node.id} is disabled in the coding runner")
    except (SyntaxError, ValueError) as exc:
        return 0.0, {"passed": 0, "total": len(tests), "error": str(exc)}
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/sub.py", "w", encoding="utf-8") as f_sub:
            f_sub.write(code)
        with open(f"{d}/tests.json", "w", encoding="utf-8") as f_tests:
            f_tests.write(json.dumps(tests))
        with open(f"{d}/runner.py", "w", encoding="utf-8") as f_run:
            f_run.write(RUNNER)
        try:
            p = subprocess.run(
                [sys.executable, "-I", f"{d}/runner.py", f"{d}/sub.py", f"{d}/tests.json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=d,
            )
            res = json.loads(p.stdout.strip().splitlines()[-1])
            return res["passed"] / max(res["total"], 1), res
        except Exception as e:
            return 0.0, {"error": repr(e)}


def validate_question(q):
    """Validation gate ensuring clear, conversational, unambiguous questions."""
    try:
        assert isinstance(q, dict), "Question must be a dictionary"
        assert all(k in q for k in ["question", "expected_evidence"])
        assert len(q["question"].strip()) >= 15, "Question text too short"
        # Disallow multi-part textbook examination patterns
        question_text = q["question"].lower()
        bad_exam_patterns = [
            "explain the root causes",
            "define clean and invalidate",
            "write down the 5 steps",
            "answer parts a, b and c",
            "compare and contrast all",
        ]
        assert not any(p in question_text for p in bad_exam_patterns), "Question sounds like an examination prompt"
        if q.get("tests"):
            s, _ = run_code_tests(q.get("reference_answer", ""), q["tests"])
            assert s == 1.0, "Reference solution failed test suite"
        return True, "ok"
    except Exception as e:
        return False, repr(e)


class QuestionBank:
    """
    Realistic Technical Interview Question Generator & Bank:
    - Conversational, workplace-grounded technical questions.
    - Contextualized to candidate resume, job description, and previous answer.
    - Supported categories: resume-based, project-based, debugging, architecture, follow-ups.
    - Strictly ONE question at a time.
    """

    def __init__(self, reg=REG, cache_path="outputs/question_cache.json"):
        self.reg = reg
        self.cache_path = cache_path
        self.cache = {}
        self.llm_generate = None
        self.generation_log = []
        if os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, "r", encoding="utf-8"))
            except Exception:
                pass

    def _generate_gemini(self, prompt):
        try:
            if self.llm_generate:
                resp = self.llm_generate(prompt)
            else:
                from llm.gemini_client import generate_response
                resp = generate_response(prompt)
            clean_resp = resp.strip()
            clean_resp = re.sub(r"^```(?:json)?\s*", "", clean_resp)
            clean_resp = re.sub(r"\s*```$", "", clean_resp)
            m = re.search(r"\{.*\}", clean_resp, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return None

    def _build_conversational_prompt(self, sk, d, ty, state):
        skill_name = self.reg.names[sk]
        diff_str = DIFF_NAMES[d]
        type_str = TYPE_NAMES[ty]
        category = state.get("category", "skill_based")

        # Context details
        resume_snippet = state.get("resume_text", "")[:1800]
        jd_snippet = state.get("job_description", "")[:700]
        last_answer = state.get("last_answer", "")
        last_question = state.get("last_question", "")
        history = state.get("history", [])
        excluded = [h.get("question", "") for h in history]
        asked_stems = [h.get("question", "")[:180] for h in history]

        # Prerequisite & KG context
        kg_pre = [self.reg.names[u] for u in self.reg.prereq[sk]]
        kg_rel = [
            self.reg.names[j]
            for j in np.argsort(-self.reg.W[sk])[:3]
            if self.reg.W[sk, j] > 0
        ]

        followup_context = ""
        if last_answer and state.get("is_followup", False):
            followup_context = f"""
            PREVIOUS QUESTION: "{last_question}"
            CANDIDATE'S PREVIOUS ANSWER: "{last_answer[:300]}"
            TASK: Generate a direct, natural FOLLOW-UP question that probes their reasoning, tests edge cases, or asks for clarification on what they just explained.
            """

        coding_context = (
            "This is a practical coding question. Keep the prompt natural (e.g. 'Can you implement a function to...'). "
            "Provide 3 unit test cases as [[\"func_call(args)\", expected_val], ...] and a correct working reference_answer.\n"
            if ty == 1
            else "This is a conversational spoken interview question. Do NOT ask for multi-part textbook definitions.\n"
        )
        category_rules = {
            "resume_based": "Anchor the question to a named technology, responsibility, or experience actually present in the resume.",
            "project_based": "Ask about a specific listed project, including architecture, implementation, a challenge, or a technical trade-off.",
            "theoretical": "Ask one concise conceptual question that tests understanding rather than memorized lists.",
            "scenario_based": "Present one realistic engineering situation and ask for the candidate's decision and reasoning.",
            "skill_based": "Assess practical proficiency in the selected skill at the requested difficulty.",
            "follow_up": "Probe a different, specific aspect of the candidate's previous answer; do not paraphrase the previous question.",
            "hr_behavioral": "Ask one HR or behavioural question about communication, teamwork, conflict, deadlines, leadership, motivation, or self-reflection, personalized to resume context.",
            "coding": "Give one executable Python coding challenge with a clear function signature and JSON-compatible sample and hidden tests.",
        }
        transcript = "\n".join(
            f"Q: {h.get('question','')}\nA: {str(h.get('answer') or '')[:350]}"
            for h in history
        )

        prompt = f"""
        You are a friendly, senior engineering interviewer conducting a realistic live technical interview.

        TARGET SKILL: {skill_name}
        TARGET DIFFICULTY: {diff_str}
        QUESTION TYPE: {type_str}
        REQUIRED ASSESSMENT CATEGORY: {category}
        CATEGORY REQUIREMENT: {category_rules.get(category, category_rules['skill_based'])}
        CANDIDATE RESUME AND EXPERIENCE: "{resume_snippet}\n{state.get('projects','')[:900]}"
        JOB REQUIREMENTS: "{jd_snippet}"
        KNOWLEDGE GRAPH PREREQUISITES: {kg_pre}
        RELATED TOPICS: {kg_rel}
        RECENT QUESTIONS ASKED: {asked_stems}
        RECENT INTERVIEW TRANSCRIPT (for context only):
        {transcript}
        EXACT QUESTIONS THAT MUST NOT BE REPEATED: {excluded}
        Explore a different aspect if the same skill appears again. Semantic paraphrases of excluded questions are also duplicates.
        RETRY INSTRUCTION: {state.get('retry_instruction', 'No retry; create a new question.')}
        {followup_context}
        {coding_context}

        INTERVIEW STYLE RULES:
        1. Keep the question conversational, concise (1-2 sentences), and natural to ask in a verbal conversation.
        2. Anchor it to practical workplace situations, projects, debugging incidents, or design trade-offs.
        3. Do NOT make it sound like a university exam question. Avoid multi-part prompts like "Explain A, define B, compare C and list 5 causes of D".
        4. Focus on assessing practical understanding, real-world reasoning, and actual experience.
        5. Return valid JSON only with this structure:
        {{
            "question": "Conversational question text (e.g. 'I noticed you worked on... How did you handle...')",
            "category": "{category}",
            "expected_evidence": ["Key concept 1 candidate should demonstrate", "Key concept 2"],
            "evaluation_criteria": "How an interviewer distinguishes strong vs weak answers",
            "reference_answer": "Concise key points of a strong answer",
            "sample_tests": [["function_call(...) ", "expected result"]],
            "tests": [["function_call(...) ", "expected JSON-compatible result"]]
        }}
        """
        return prompt

    def get(self, sk, d, ty, state=None):
        state = state or {}
        name = self.reg.names[sk]
        key = f"{name}|{d}|{ty}"

        category = state.get("category", "skill_based")
        history = state.get("history", [])
        self.generation_log.append({"skill": name, "difficulty": d, "type": ty, "category": category})

        # Try structured, action-conditioned generation twice; each retry includes
        # the complete exclusions and asks the model to choose another aspect.
        for attempt in range(2):
            retry_state = dict(state)
            if attempt:
                retry_state["retry_instruction"] = "Previous response was duplicate or invalid. Create a distinct question about a different aspect."
            prompt = self._build_conversational_prompt(sk, d, ty, retry_state)
            q_data = self._generate_gemini(prompt)
            if q_data:
                ok, why = validate_question(q_data)
                category_ok = q_data.get("category", category) == category
                code_ok = ty == 0 or (q_data.get("tests") and q_data.get("reference_answer"))
                if ok and category_ok and code_ok and not is_duplicate(q_data.get("question", ""), history):
                    q_data.update(
                        skill=name,
                        difficulty=d,
                        type=ty,
                        category=category,
                        source="llm:gemini",
                        validated=True,
                        created=datetime.datetime.utcnow().isoformat(),
                        id=hashlib.md5(q_data["question"].encode()).hexdigest()[:10],
                        normalized_question=normalize_question(q_data["question"]),
                    )
                    self.cache.setdefault(key, []).append(q_data)
                    return q_data

        # Category-specific, non-repeating fallbacks are used when Gemini is
        # unavailable or returns duplicate/invalid JSON. Coding remains executable.
        q_data = self._curated_for_category(name, d, ty, state)
        if not is_duplicate(q_data["question"], history):
            return q_data

        # Rotate through distinct aspects before failing loudly. Never return a
        # duplicate silently, even if a long session exhausted the fallback set.
        for aspect in FALLBACK_ASPECTS:
            candidate = self._curated_for_category(name, d, ty, {**state, "aspect": aspect, "variant": True})
            if not is_duplicate(candidate["question"], history):
                return candidate
        raise QuestionExhaustedError(
            f"No unique question remains for {name} at difficulty {d} "
            f"({category}); the controller should select another legal action."
        )

    def _curated_for_category(self, name, d, ty, state):
        category = state.get("category", "skill_based")
        if ty == 1:
            item = CODING.get((name, d)) or self._generic_coding_item(name, d)
            if item:
                tests = list(item["tests"])
                return dict(
                    id=hashlib.md5(f"{name}|{item['q']}".encode()).hexdigest()[:10], skill=name,
                    difficulty=d, type=1, category="coding", question=item["q"],
                    reference_answer=item["ref"], tests=tests, sample_tests=tests[:2],
                    expected_evidence=["Correct implementation", "Edge-case handling", "Reasoning about complexity"],
                    evaluation_criteria="Hidden unit tests evaluate executable behavior; the rubric also considers the stated approach.",
                    source="curated", validated=True,
                    normalized_question=normalize_question(item["q"]),
                )
        curated = REALISTIC_QUESTION_BANK.get((name, d))
        if curated and category in {"skill_based", "theoretical", "scenario_based"} and not state.get("aspect"):
            question = curated["question"]
            return dict(
                id=hashlib.md5(f"{category}|{name}|{d}|{question}".encode()).hexdigest()[:10],
                skill=name, difficulty=d, type=0, category=category, question=question,
                reference_answer=curated["reference_answer"],
                expected_evidence=curated["expected_evidence"],
                evaluation_criteria=curated["evaluation_criteria"],
                tests=None, sample_tests=None, source="research_question_bank", validated=True,
                normalized_question=normalize_question(question),
            )
        aspect = state.get("aspect", "trade-offs")
        resume = (state.get("resume_text", "") or "").strip()
        projects = (state.get("projects", "") or "").strip()
        project_excerpt = (projects or resume).replace("\n", " ")[:180]
        variant = FALLBACK_ASPECTS.index(aspect) if aspect in FALLBACK_ASPECTS else 0
        frames = {
            "resume_based": (
                f"Your resume mentions {project_excerpt or name}. What decision involving {name} did you make, and what evidence showed it worked for {aspect}?",
                f"Looking at {project_excerpt or name}, describe how {name} shaped your approach to {aspect}. What evidence informed the choice?",
                f"If you revisited the work described as {project_excerpt or name}, what would you change about {name} to better handle {aspect}?",
            ),
            "project_based": (
                f"In {project_excerpt or 'your project'}, how did you use {name} when {aspect} mattered? What trade-off did you make?",
                f"Choose one decision from {project_excerpt or 'your project'} involving {name}. How would {aspect} change that decision?",
                f"What evidence from {project_excerpt or 'your project'} would show that your use of {name} handled {aspect} well?",
            ),
            "theoretical": (
                f"At a {DIFF_NAMES[d]} level, what principle about {name} helps explain its behavior under {aspect}?",
                f"Why can {aspect} affect a system built with {name}, and what underlying mechanism explains the effect?",
                f"How would you distinguish a sound use of {name} from a weak one when reasoning about {aspect}?",
            ),
            "scenario_based": (
                f"A production service using {name} faces a new challenge involving {aspect} during a traffic increase. What would you investigate first?",
                f"Suppose a release using {name} causes a problem related to {aspect}. What action would you take first and what would you measure?",
                f"A team must change a system using {name} while facing {aspect}. How would you choose a safe course of action?",
            ),
            "hr_behavioral": (
                f"Tell me about a time you worked with others through {aspect} on a project involving {project_excerpt or name}. What did you do and what changed?",
                f"Describe a situation where communication or teamwork helped resolve {aspect}. What was your role and how did you know the outcome improved?",
                f"When have you had to explain a difficult {name} decision to a teammate? How did you handle disagreement about {aspect}?",
            ),
            "follow_up": (
                f"You mentioned: '{str(state.get('last_answer',''))[:100]}'. What evidence would help you validate that approach to {name} under {aspect}?",
                f"Building on your answer, what alternative would you consider for {name} if {aspect} became the main constraint?",
                f"You described your {name} approach. Which assumption would you test first if {aspect} changed?",
            ),
            "skill_based": (
                f"How would you apply {name} in an engineering task that requires attention to {aspect}? Which trade-off would guide your choice?",
                f"Imagine you are responsible for a system using {name}. How would you detect and address {aspect}?",
                f"What practical evidence would convince you that your approach to {name} handles {aspect} reliably?",
            ),
        }
        category_frames = frames.get(category, frames["skill_based"])
        question = category_frames[variant % len(category_frames)]
        evidence = [f"Reasoning about {name}", f"Evidence related to {aspect}", "Practical trade-off or validation"]
        return dict(
            id=hashlib.md5(f"{category}|{name}|{d}|{aspect}|{question}".encode()).hexdigest()[:10],
            skill=name, difficulty=d, type=0, category=category, question=question,
            reference_answer=f"A sound response explains {name}, considers {aspect}, and supports the decision with evidence.",
            expected_evidence=evidence, evaluation_criteria="Assess practical reasoning, relevant evidence and clarity.",
            tests=None, sample_tests=None, source="curated_fallback", validated=True,
            normalized_question=normalize_question(question),
        )

    @staticmethod
    def _generic_coding_item(name, difficulty):
        """Distinct executable Python challenges for other coding-capable skills."""
        items = {
            0: dict(
                q=f"Implement `count_occurrences(items, target)` for a {name} exercise. Return how many times target appears, including correct behavior for an empty input.",
                ref="def count_occurrences(items, target):\n    return sum(1 for item in items if item == target)",
                tests=[["count_occurrences([1,2,1], 1)", 2], ["count_occurrences([], 1)", 0], ["count_occurrences(['a','b'], 'c')", 0]],
            ),
            1: dict(
                q=f"Implement `first_unique_char(text)` for a {name} exercise. Return the first character that occurs once, or None if no such character exists.",
                ref="def first_unique_char(text):\n    counts = {c: text.count(c) for c in text}\n    return next((c for c in text if counts[c] == 1), None)",
                tests=[["first_unique_char('swiss')", "w"], ["first_unique_char('aabb')", None], ["first_unique_char('z')", "z"]],
            ),
            2: dict(
                q=f"Implement `longest_unique_window(text)` for a {name} exercise. Return the length of the longest substring with no repeated characters.",
                ref="def longest_unique_window(text):\n    left = best = 0\n    last = {}\n    for right, ch in enumerate(text):\n        left = max(left, last.get(ch, -1) + 1)\n        last[ch] = right\n        best = max(best, right - left + 1)\n    return best",
                tests=[["longest_unique_window('abcabcbb')", 3], ["longest_unique_window('bbbbb')", 1], ["longest_unique_window('')", 0]],
            ),
        }
        return items[difficulty]
