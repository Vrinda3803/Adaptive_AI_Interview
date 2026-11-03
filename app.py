import os
import json
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path

from core.knowledge_graph import REG
from core.action_net import load_policy
from resume.resume_parser import build_candidate_profile
from interview.live_controller import LiveInterviewController

APP_ROOT = Path(__file__).resolve().parent
POLICY = load_policy(APP_ROOT / "models" / "adaptive_ppo_seed1.pt")


@st.cache_resource
def load_selected_policy(path):
    return load_policy(path)

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive AI Technical Interview Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern corporate / engineering aesthetic
st.markdown(
    """
    <style>
    .interview-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-diff { background-color: #dbeafe; color: #1e40af; }
    .badge-cat { background-color: #fef3c7; color: #92400e; }
    .badge-skill { background-color: #e0e7ff; color: #3730a3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.title("🎙️ AI Interviewer")
    st.caption("Evidence-Grounded Adaptive Technical Interview Platform")

    st.markdown("---")
    portal_mode = st.radio(
        "Platform View",
        [
            "Practice Interview",
            "Hiring & Candidate Comparison",
            "Research Dashboard",
        ],
        index=0,
    )
    checkpoint_candidates = sorted((APP_ROOT / "models").glob("*.pt"))
    selected_checkpoint = APP_ROOT / "models" / "adaptive_ppo_seed1.pt"
    if portal_mode == "Research Dashboard" and checkpoint_candidates:
        selected_checkpoint = st.selectbox(
            "Research policy checkpoint",
            checkpoint_candidates,
            format_func=lambda p: p.name,
            help="Place a compatible custom PyTorch state-dict checkpoint in models/ to make it selectable.",
        )
    active_policy = load_selected_policy(str(selected_checkpoint)) if selected_checkpoint.exists() else POLICY

    st.markdown("---")
    st.subheader("⚙️ Interview Session Settings")
    max_questions = st.slider("Max Question Budget", min_value=5, max_value=15, value=10)
    interview_duration = st.slider("Duration Limit (Minutes)", min_value=10, max_value=30, value=20)
    required_competency = st.slider("Required competency threshold (θ)", -1.5, 1.5, 0.0, 0.1) if portal_mode == "Hiring & Candidate Comparison" else 0.0

    st.markdown("---")
    st.caption(
        "Powered by Evidence-Grounded Reinforcement Learning (PPO), "
        "71-Grid Bayesian Item Response Theory, and Google Gemini."
    )


# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "controller" not in st.session_state:
    st.session_state.controller = None
if "candidate_profile" not in st.session_state:
    st.session_state.candidate_profile = None
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "target_role" not in st.session_state:
    st.session_state.target_role = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "hiring_results" not in st.session_state:
    st.session_state.hiring_results = []
if "heldout_results" not in st.session_state:
    st.session_state.heldout_results = None
if "ablation_run_results" not in st.session_state:
    st.session_state.ablation_run_results = None
if "robustness_run_results" not in st.session_state:
    st.session_state.robustness_run_results = None


# =============================================================
# VIEW 1: CANDIDATE INTERVIEW PORTAL
# (Clean, professional, conversational — NO senior dropdowns, NO raw RL state)
# =============================================================
if portal_mode in ("Practice Interview", "Hiring & Candidate Comparison"):

    # STAGE 1: ONBOARDING & RESUME UPLOAD
    if not st.session_state.interview_started:
        is_hiring = portal_mode == "Hiring & Candidate Comparison"
        st.title("Candidate Assessment Setup" if is_hiring else "Welcome to Your Technical Interview")
        if is_hiring and st.session_state.hiring_results:
            st.subheader("Completed Candidate Assessments")
            st.dataframe(pd.DataFrame(st.session_state.hiring_results), use_container_width=True)
            st.download_button(
                "Export Candidate Comparison",
                pd.DataFrame(st.session_state.hiring_results).to_csv(index=False),
                "candidate_comparison.csv", "text/csv",
            )
        st.write(
            "This interview platform assesses your practical software engineering, problem-solving, "
            "and architectural design experience. Questions adapt dynamically to your background and answers."
        )

        st.markdown("---")
        st.header("1. Candidate Details & Target Position")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            candidate_name = st.text_input("Your Full Name", value=st.session_state.candidate_name or "Alex Morgan")
            st.session_state.candidate_name = candidate_name
        with col_c2:
            preset_roles = [
                "Backend Software Engineer",
                "Machine Learning & AI Engineer",
                "Embedded Software Engineer",
                "Data Platform Engineer",
                "Full Stack Web Engineer",
                "Custom Job Description",
            ]
            selected_preset = st.selectbox("Target Engineering Role", options=preset_roles, index=0)

        job_desc_input = ""
        if selected_preset == "Custom Job Description":
            job_desc_input = st.text_area(
                "Paste the Target Job Description / Key Responsibilities:",
                height=120,
                placeholder="e.g. Seeking an engineer with deep expertise in Python, distributed databases, Docker, and REST APIs...",
            )
        else:
            job_desc_input = f"Position for {selected_preset}. Core competencies include system architecture, debugging, production deployment, and core technical skills."

        st.session_state.target_role = selected_preset

        st.markdown("---")
        st.header("2. Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, or TXT format)",
            type=["pdf", "docx", "txt"],
            help="Supports standard and scanned PDF documents with optical character recognition (OCR).",
        )

        if uploaded_file is not None:
            if st.button("📄 Parse Resume & Prepare Interview Session", use_container_width=True):
                temp_path = None
                try:
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    with st.spinner("Analyzing resume experience and mapping engineering context..."):
                        profile = build_candidate_profile(temp_path)
                        st.session_state.candidate_profile = profile
                    st.success("Resume parsed successfully! Review your briefing below.")
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

        with st.expander("Or enter a profile manually"):
            manual_skills = st.multiselect("Skills to assess", REG.names, key="manual_skills")
            manual_resume = st.text_area("Resume highlights or experience", height=100, key="manual_resume")
            manual_projects = st.text_area("Projects and responsibilities", height=90, key="manual_projects")
            if st.button("Use Manual Profile", key="manual_profile_button"):
                resume_text = "\n".join([manual_resume, "Skills: " + ", ".join(manual_skills), manual_projects])
                st.session_state.candidate_profile = {
                    "resume_text": resume_text,
                    "skills": manual_skills,
                    "projects": manual_projects,
                    "experience": manual_resume,
                    "education": "",
                }
                st.success("Manual candidate profile is ready.")

        # INTERVIEW BRIEFING
        profile = st.session_state.candidate_profile
        if profile is not None:
            st.markdown("---")
            st.header("3. Interview Briefing")
            st.markdown(
                f"**Candidate:** `{st.session_state.candidate_name}` &nbsp;|&nbsp; "
                f"**Target Role:** `{st.session_state.target_role}`"
            )

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.subheader("Recognized Technical Focus Areas")
                if profile.get("skills"):
                    skill_badges = " ".join([f"`{s}`" for s in profile["skills"]])
                    st.markdown(skill_badges)
                else:
                    st.info("General technical background detected; standard role competencies will be assessed.")
                st.write(f"**Education:** {profile.get('education') or 'Standard Degree'}")

            with b_col2:
                st.subheader("Highlighted Projects & Experience")
                proj_text = profile.get("projects") or "Practical project experience identified."
                if isinstance(proj_text, list):
                    proj_text = "\n".join(proj_text)
                st.write(proj_text[:280] + ("..." if len(proj_text) > 280 else ""))

            st.markdown("---")
            st.subheader("Instructions for the Interview")
            st.markdown(
                """
                - The interview consists of short, conversational questions about your actual technical experience, problem solving, and architecture decisions.
                - You will be asked **one question at a time**.
                - Respond as you would in a live spoken interview.
                - For coding questions, a code input window will be provided.
                - You can request an optional coaching hint if you need guidance.
                """
            )

            start_col, _ = st.columns([1, 2])
            with start_col:
                if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Initializing your adaptive interview session..."):
                            controller = LiveInterviewController(
                                candidate_profile=profile,
                                role_text=job_desc_input,
                                interview_duration_minutes=interview_duration,
                                max_questions=max_questions,
                                policy=active_policy,
                                mode="hiring" if is_hiring else "practice",
                                target_competency=required_competency,
                            )
                            first_q = controller.generate_next_question()
                            st.session_state.controller = controller
                            st.session_state.current_question = first_q
                            st.session_state.interview_history = []
                            st.session_state.interview_started = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize interview: {e}")

    # STAGE 2: LIVE CONVERSATIONAL INTERVIEW
    controller = st.session_state.controller
    if controller is not None and st.session_state.interview_started:
        st.title("🎙️ Live Candidate Assessment")

        remaining_sec = controller.update_timer()
        mins = int(remaining_sec // 60)
        secs = int(remaining_sec % 60)

        # Header with clean metrics for candidate (NO raw RL vectors)
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Time Remaining", f"{mins:02d}:{secs:02d}")
        m2.metric("📋 Question Progress", f"{controller.session.env.t} / {max_questions}")
        m3.metric("🎯 Candidate", st.session_state.candidate_name or "Candidate")

        # Termination Check
        if controller.is_finished():
            controller.finished = True

        q_data = st.session_state.current_question

        # DISPLAY ACTIVE QUESTION
        if not controller.finished and q_data and not q_data.get("finished", False):
            st.markdown("---")

            # Conversational tags
            skill = q_data.get("skill", "General")
            category = q_data.get("category_label", q_data.get("category", "Skill Based").replace("_", " ").title())

            st.markdown(
                f'<span class="badge badge-skill">{skill}</span>'
                f'<span class="badge badge-cat">{category}</span>',
                unsafe_allow_html=True,
            )

            st.subheader(f"Interviewer:")
            st.info(f"💬 \"{q_data.get('question')}\"")

            # Guided Hint
            with st.expander("💡 Need a Hint or Clarification?"):
                if st.button("Ask Interviewer for a Hint"):
                    with st.spinner("Interviewer is providing guidance..."):
                        hint_msg = controller.generate_hint()
                        st.write(f"💭 **Interviewer Hint:** {hint_msg}")

            # Answer Input Area
            is_coding = q_data.get("question_type") == "coding" or bool(q_data.get("tests"))

            if is_coding:
                st.markdown("💻 **Coding Challenge — Python**")
                st.caption("Python is the supported execution language. Sample cases are shown; the full test suite runs after submission.")
                if q_data.get("sample_tests"):
                    st.code(json.dumps(q_data["sample_tests"], indent=2), language="json")
                st.selectbox("Language", ["Python 3"], key="coding_language")
                code_ans = st.text_area(
                    "Write your implementation below:",
                    value="def solution():\n    # write code here\n    pass",
                    height=220,
                    key=f"cand_code_input_{q_data.get('id', controller.questions_asked)}",
                )
                candidate_response = code_ans
                candidate_reasoning = st.text_area(
                    "Explain your algorithm and complexity",
                    height=90,
                    placeholder="Describe the approach, edge cases, and time/space trade-offs.",
                    key=f"cand_code_reasoning_{q_data.get('id', controller.questions_asked)}",
                )
            else:
                st.markdown("✍️ **Your Answer:**")
                text_ans = st.text_area(
                    "Speak or type your response:",
                    height=160,
                    placeholder=("Use a specific situation, your action, and the outcome." if q_data.get("category") == "hr_behavioral" else "Explain your practical approach, technical decisions, and relevant trade-offs..."),
                    key=f"cand_text_input_{q_data.get('id', controller.questions_asked)}",
                )
                candidate_response = text_ans
                candidate_reasoning = ""

            sub_col, _ = st.columns([1, 2])
            with sub_col:
                if st.button("📤 Submit Answer", type="primary", use_container_width=True):
                    if not candidate_response.strip():
                        st.warning("Please provide a response before submitting.")
                    else:
                        try:
                            with st.spinner("Interviewer is assessing your response..."):
                                result = controller.submit_answer(candidate_response, reasoning=candidate_reasoning)
                                st.session_state.interview_history.append(result)
                                st.session_state.current_question = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error evaluating response: {e}")

        # DISPLAY RECENT FEEDBACK
        if st.session_state.interview_history:
            recent = st.session_state.interview_history[-1]
            st.markdown("---")
            st.subheader("📝 Interviewer Feedback on Last Response")
            score_pct = recent["score"] * 100
            st.write(f"**Score for this question:** {score_pct:.0f}%")
            st.write(f"**Feedback:** {recent['evaluation'].get('feedback')}")

            missing = recent['evaluation'].get('missing_concepts')
            if missing and score_pct < 85:
                st.write(f"**Key points to consider:** {', '.join(missing)}")

        # PROCEED BUTTON
        if not controller.is_finished() and st.session_state.current_question is None:
            st.markdown("---")
            if st.button("➡️ Continue to Next Question", type="primary", use_container_width=True):
                try:
                    with st.spinner("Interviewer is preparing the next question..."):
                        next_q = controller.generate_next_question()
                        st.session_state.current_question = next_q
                        st.rerun()
                except Exception as e:
                    st.error(f"Error retrieving next question: {e}")

        # STAGE 3: INTERVIEW COMPLETION & REPORT
        if controller.is_finished():
            controller.finished = True
            st.session_state.interview_started = False
            st.balloons()

            st.markdown("---")
            st.header("🎉 Interview Completed")

            end_reason = getattr(controller.session, "end_reason", None)
            if end_reason == "confidence_target":
                st.success("Assessment concluded after the minimum question count because the confidence target was reached.")
            elif end_reason == "question_budget":
                st.success("The configured question budget has been reached.")
            elif end_reason in {"question_bank_exhausted", "legal_actions_exhausted"}:
                st.warning("The interview stopped because no new valid question was available for the selected skills. Review the competency report and broaden the skill profile for a future session.")
            else:
                st.success("The interview session has finished.")

            st.subheader("Final Competency Assessment Report")
            report_df = controller.get_report()

            # Render clean candidate table (omitting internal raw math)
            display_df = report_df[
                [
                    "skill",
                    "competency",
                    "ci95_lo",
                    "ci95_hi",
                    "verdict",
                    "questions",
                    "mean_score",
                    "bkt_mastery",
                    "needs_human_review",
                ]
            ].copy()
            display_df.columns = [
                "Technical Competency",
                "Estimated Mastery (θ)",
                "95% CI Lower",
                "95% CI Upper",
                "Assessment Verdict",
                "Questions Asked",
                "Average Score",
                "BKT Mastery (parallel estimate)",
                "Human Review",
            ]
            st.dataframe(display_df, use_container_width=True)

            # Plot Competency Chart
            st.markdown("---")
            st.subheader("Skill Competency Profile (with 95% Confidence Intervals)")
            fig = controller.plot_report()
            st.pyplot(fig)

            # Interaction history expander
            with st.expander("📜 Review Question-by-Question Interaction Transcript"):
                for idx, item in enumerate(st.session_state.interview_history, start=1):
                    st.markdown(f"**Q{idx} · {item.get('category', 'assessment')} · {item['skill']}:** {item['evaluation'].get('feedback')}")
                    st.write(f"Score: {item['score']*100:.0f}% | Confidence: {item.get('evaluator_confidence', 0.85)*100:.0f}%")
                    st.markdown("---")

            report_rows = report_df.to_dict(orient="records")
            hiring_key = id(controller)
            if portal_mode == "Hiring & Candidate Comparison" and st.session_state.get("saved_hiring_controller") != hiring_key:
                for row in report_rows:
                    st.session_state.hiring_results.append({"candidate": st.session_state.candidate_name or "Candidate", **row})
                st.session_state.saved_hiring_controller = hiring_key
            if st.button("🔄 Start Next Candidate" if portal_mode == "Hiring & Candidate Comparison" else "🔄 Start Another Practice Interview"):
                st.session_state.controller = None
                st.session_state.candidate_profile = None
                st.session_state.current_question = None
                st.session_state.interview_history = []
                st.session_state.interview_started = False
                st.session_state.saved_hiring_controller = None
                st.rerun()


# =============================================================
# VIEW 2: RESEARCHER & ANALYTICS DASHBOARD
# (Full research metrics, ablation, generalization, noise robustness)
# =============================================================
elif portal_mode == "Research Dashboard":
    st.title("📊 Research & Empirical Performance Dashboard")
    st.write(
        "The hierarchical RL controller owns adaptive question selection: PPO chooses skill, difficulty, "
        "and question type; the trained category Q-policy chooses the content category. The LLM generates "
        "and evaluates content after those decisions. This dashboard shows the policy trace and research results."
    )

    r_tabs = st.tabs(
        [
            "🏆 Held-Out Algorithm Benchmarks",
            "📈 Learning Curves & Regret",
            "🔬 Novelty & Component Ablation",
            "🌐 Generalization & Noise Robustness",
            "🗺️ Knowledge Graph Topology",
            "🧭 Live Policy Trace",
        ]
    )

    # 1. HELD-OUT BENCHMARK
    with r_tabs[0]:
        st.subheader("Seeded held-out algorithm benchmark")
        benchmark_dir = APP_ROOT / "reports/seeded_benchmark"
        summary_path = benchmark_dir / "summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            manifest_path = benchmark_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            st.caption(
                f"Exploratory paired evaluation: {manifest.get('episodes_per_checkpoint', '?')} common episodes "
                f"per checkpoint, {len(manifest.get('checkpoint_sha256', {}))} trained checkpoints, "
                f"held-out domains: {', '.join(manifest.get('domains', []))}. "
                "Intervals resample training checkpoints and episodes; synthetic results only."
            )
            st.dataframe(summary[summary["metric"].isin(["weighted_rmse", "success", "return"])], width="stretch")
            figure_dir = benchmark_dir / "figures"
            for filename, alt in (
                ("weighted_rmse.png", "Weighted RMSE with hierarchical bootstrap confidence intervals by policy"),
                ("success_rate.png", "Simulation success rate with hierarchical bootstrap confidence intervals by policy"),
                ("return.png", "Episode return with hierarchical bootstrap confidence intervals by policy"),
            ):
                if (figure_dir / filename).exists():
                    st.image(figure_dir / filename, caption=alt, alt=alt, width="stretch")
            paired_path = benchmark_dir / "paired_vs_maxeig.csv"
            if paired_path.exists():
                st.markdown("**Paired weighted-RMSE differences versus MaxEIG**")
                st.dataframe(pd.read_csv(paired_path), width="stretch")
            st.caption(
                "Only three trained checkpoints per RL family are currently available. "
                "Paired difference intervals average those checkpoints and condition on them; "
                "they are exploratory, not a definitive ranking. See RESEARCH_PROTOCOL.md."
            )
        else:
            st.info("Run `python run_research_benchmark.py --episodes 300` to create the seeded benchmark report.")

        with st.expander("Archived aggregate comparison (previous protocol)"):
            st.caption("This table and its saved tests come from the earlier aggregate evaluation; they are not the seeded benchmark above.")
            if (APP_ROOT / "reports/main_results.csv").exists():
                st.dataframe(pd.read_csv(APP_ROOT / "reports/main_results.csv"), width="stretch")
            if (APP_ROOT / "reports/figures/final_comparison.png").exists():
                st.image(APP_ROOT / "reports/figures/final_comparison.png", caption="Archived aggregate strategy comparison", alt="Archived aggregate comparison of return, success rate, and weighted RMSE", width="stretch")

    # 2. LEARNING CURVES & REGRET
    with r_tabs[1]:
        st.subheader("Training Trajectories & Cumulative Regret")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            if (APP_ROOT / "reports/figures/learning_curves.png").exists():
                st.image(APP_ROOT / "reports/figures/learning_curves.png", caption="Validation Returns over 4,000 Training Episodes")
        with col_l2:
            if (APP_ROOT / "reports/figures/cumulative_regret.png").exists():
                st.image(APP_ROOT / "reports/figures/cumulative_regret.png", caption="Cumulative Evaluation Regret over 300 Interviews")
            if (APP_ROOT / "reports/convergence.csv").exists():
                st.dataframe(pd.read_csv(APP_ROOT / "reports/convergence.csv"), use_container_width=True)
            if (APP_ROOT / "reports/regret.csv").exists():
                st.dataframe(pd.read_csv(APP_ROOT / "reports/regret.csv"), use_container_width=True)

    # 3. ABLATION STUDY
    with r_tabs[2]:
        st.subheader("Systematic Component Ablation Study")
        st.write(
            "Isolates the individual contribution of **Knowledge Graph evidence sharing**, **Uncertainty features**, "
            "**Expected Information Gain (EIG)**, and **Resume Priors**."
        )
        if (APP_ROOT / "reports/ablation.csv").exists():
            st.dataframe(pd.read_csv(APP_ROOT / "reports/ablation.csv"), use_container_width=True)
        if (APP_ROOT / "reports/figures/ablation.png").exists():
            st.image(APP_ROOT / "reports/figures/ablation.png", caption="Return Degradation upon Ablating Individual Architecture Components")
        ablation_episodes = st.slider("Episodes per ablation variant", 2, 30, 5, key="ablation_episodes")
        if st.button("Run ablation with selected checkpoint"):
            from core.research import evaluate_ablation
            with st.spinner("Evaluating checkpoint with research components ablated..."):
                st.session_state.ablation_run_results = evaluate_ablation(active_policy, episodes=ablation_episodes)
        if st.session_state.ablation_run_results:
            st.dataframe(pd.DataFrame(st.session_state.ablation_run_results), use_container_width=True)

    # 4. GENERALIZATION & NOISE ROBUSTNESS
    with r_tabs[3]:
        st.subheader("Zero-Shot Transfer & Evaluator Noise Robustness")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if (APP_ROOT / "reports/generalization.csv").exists():
                st.dataframe(pd.read_csv(APP_ROOT / "reports/generalization.csv"), use_container_width=True)
            if (APP_ROOT / "reports/figures/generalization.png").exists():
                st.image(APP_ROOT / "reports/figures/generalization.png", caption="Generalization Gap: Seen vs Unseen Domains")
        with col_g2:
            if (APP_ROOT / "reports/robustness.csv").exists():
                st.dataframe(pd.read_csv(APP_ROOT / "reports/robustness.csv"), use_container_width=True)
            if (APP_ROOT / "reports/figures/robustness.png").exists():
                st.image(APP_ROOT / "reports/figures/robustness.png", caption="Evaluator Noise Sensitivity Analysis")
        robustness_episodes = st.slider("Episodes per noise level", 2, 30, 5, key="robustness_episodes")
        if st.button("Run evaluator-noise robustness study"):
            from core.research import evaluate_noise_robustness
            with st.spinner("Running held-out evaluator-noise conditions..."):
                st.session_state.robustness_run_results = evaluate_noise_robustness(active_policy, episodes=robustness_episodes)
        if st.session_state.robustness_run_results:
            st.dataframe(pd.DataFrame(st.session_state.robustness_run_results), use_container_width=True)

    # 5. KNOWLEDGE GRAPH TOPOLOGY
    with r_tabs[4]:
        st.subheader("Hierarchical Skill Knowledge Graph (12 technical domains, 40 technical skills, 34 dependencies) + communication dimension")
        if (APP_ROOT / "reports/figures/knowledge_graph.png").exists():
            st.image(APP_ROOT / "reports/figures/knowledge_graph.png", caption="Skill Knowledge Graph DAG")

    with r_tabs[5]:
        st.write(f"Active custom checkpoint: `{selected_checkpoint.name}` · policy `{active_policy.name}` · {active_policy.n_params():,} parameters.")
        episodes = st.slider("Held-out synthetic interview episodes", 5, 100, 25, key="research_episode_count")
        if st.button("Evaluate selected policy on held-out domains"):
            from core.research import evaluate_heldout
            with st.spinner("Running held-out evaluation with hidden synthetic mastery..."):
                st.session_state.heldout_results = evaluate_heldout(active_policy, episodes=episodes)
        if st.session_state.heldout_results:
            heldout = pd.DataFrame(st.session_state.heldout_results)
            st.dataframe(heldout, use_container_width=True)
            st.download_button("Export held-out evaluation", heldout.to_csv(index=False), "heldout_policy_evaluation.csv", "text/csv")
        trace = st.session_state.controller.session.history if st.session_state.controller else []
        if trace:
            trace_df = pd.DataFrame(trace)
            st.dataframe(trace_df[[c for c in ("action", "skill", "category", "category_policy", "category_policy_state", "category_policy_q_values", "difficulty", "type", "score", "mu_before", "mu_after", "sd_before", "sd_after", "bkt_mastery") if c in trace_df]], use_container_width=True)
        else:
            st.info("Start an interview to inspect its action sequence and posterior updates here.")

    # DOWNLOADS
    st.markdown("---")
    st.subheader("📥 Export Experimental Data")
    d1, d2, d3, d4 = st.columns(4)
    if (APP_ROOT / "reports/main_results.csv").exists():
        d1.download_button("Download Main Results CSV", (APP_ROOT / "reports/main_results.csv").read_bytes(), "main_results.csv", "text/csv")
    if (APP_ROOT / "reports/ablation.csv").exists():
        d2.download_button("Download Ablation CSV", (APP_ROOT / "reports/ablation.csv").read_bytes(), "ablation.csv", "text/csv")
    if (APP_ROOT / "reports/generalization.csv").exists():
        d3.download_button("Download Generalization CSV", (APP_ROOT / "reports/generalization.csv").read_bytes(), "generalization.csv", "text/csv")
    if (APP_ROOT / "reports/results_summary.json").exists():
        d4.download_button("Download Summary JSON", (APP_ROOT / "reports/results_summary.json").read_bytes(), "results_summary.json", "application/json")
