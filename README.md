
# Adaptive AI Interview Simulator

## Project Overview

The Adaptive AI Interview Simulator is a Streamlit-based application developed to conduct personalized technical interviews using Reinforcement Learning and Artificial Intelligence.

Unlike conventional interview systems that follow a fixed set of questions, this application selects questions dynamically based on the candidate's skills, previous responses, and estimated competency levels.

The system uses Proximal Policy Optimization (PPO) for adaptive question selection, a Knowledge Graph for representing skill relationships, Bayesian Item Response Theory (IRT) and Bayesian Knowledge Tracing (BKT) for competency assessment, and Google Gemini for question generation and answer evaluation.

The application supports practice interviews, candidate assessment, and research-based evaluation of reinforcement learning algorithms.

## Implementation

The interview process begins with the candidate uploading a resume or manually entering their skills and project details. A job description can also be provided to identify the skills relevant to a particular role.

The main components of the system are described below.

### 1. Resume Processing and Knowledge Graph

The application extracts relevant skills, projects, and experience from the candidate's profile. These skills are mapped to a Knowledge Graph containing technical concepts and their relationships.

The extracted information is used to identify the skills that should be assessed during the interview.

### 2. Adaptive Question Selection using PPO

A trained PPO policy selects the next interview action based on the current assessment state. Each action specifies the skill to assess, the difficulty level, and the question type.

The policy considers estimated competency, uncertainty, skill coverage, and available questions.

A separate Q-learning agent selects the question category, such as skill-based, resume-based, project-based, scenario-based, coding, or behavioral.

### 3. Question Generation

Questions are retrieved from the question bank or generated using Gemini based on the selected skill, difficulty, question type, and category.

The application checks previously asked questions to reduce repetition. If a suitable question is unavailable, the system attempts another compatible category or valid assessment action.

### 4. Answer Evaluation

Candidate responses are evaluated using Gemini to assess technical correctness, relevance, and reasoning. Coding questions are evaluated using executable test cases along with reasoning assessment.

The evaluation produces an answer score, evaluator confidence, and feedback, which are used to update the candidate's competency estimates.

### 5. Competency Tracking

Bayesian IRT maintains continuous estimates of candidate competency and uncertainty. The Knowledge Graph supports the propagation of assessment evidence between related skills.

BKT independently tracks the probability of mastery for individual skills.

Both models are updated after evaluated responses. Bayesian IRT provides the primary competency state used for adaptive question selection, while BKT provides an additional mastery estimate for reporting.

### 6. Interview Completion and Reporting

The interview continues until the configured question limit, time limit, or applicable stopping condition is reached.

At the end of the interview, the application generates a report containing skill-wise competency estimates, uncertainty values, mastery probabilities, evaluation feedback, and interview history.

## Application Features

The application provides three main modes:

- **Practice Interview:** Conducts personalized technical interviews using resume information, adaptive questions, coding assessment, and feedback.
- **Hiring and Candidate Comparison:** Supports role-specific interviews, competency requirements, candidate reports, and comparison of completed assessments.
- **Research Dashboard:** Displays reinforcement learning evaluation results, algorithm comparisons, competency estimation metrics, and performance visualizations.

## Technologies and Dependencies

The project is implemented in Python and uses the following technologies:

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| Streamlit | Web application interface |
| PyTorch | PPO neural network implementation |
| NumPy | Numerical computations |
| Pandas | Data processing and result analysis |
| Google Gemini API | Question generation and answer evaluation |
| Bayesian IRT | Competency estimation and uncertainty tracking |
| Bayesian Knowledge Tracing | Skill mastery estimation |
| Knowledge Graph | Representation of skill relationships |
| Tesseract OCR | Text extraction from scanned resumes |

The required Python packages are listed in `requirements.txt`.

## Project Structure

The main files and directories are organized as follows:

```text
Adaptive_AI_Interview_Final/
│
├── app.py
├── requirements.txt
├── .env.example
│
├── core/
│   ├── action_net.py
│   ├── environment.py
│   ├── knowledge_graph.py
│   ├── bayesian_irt.py
│   ├── bkt.py
│   ├── category_agent.py
│   ├── question_bank.py
│   ├── evaluator.py
│   ├── assessment.py
│   ├── benchmark.py
│   └── session.py
│
├── interview/
├── llm/
├── resume/
├── models/
├── reports/
├── tests/
│
├── demo_interview.py
├── train_category_policy.py
└── run_research_benchmark.py
```

### File and Directory Description

| File / Directory | Description |
|---|---|
| `app.py` | Main Streamlit application containing the practice interview, hiring, and research dashboard interfaces. |
| `core/assessment.py` | Handles question normalization, similarity checks, and duplicate detection. |
| `core/category_agent.py` | Implements the Q-learning agent for selecting question categories. |
| `core/action_net.py` | Defines the PPO neural network and handles policy-based action selection. |
| `core/environment.py` | Implements the interview environment, state representation, action masks, reward calculation, and episode termination. |
| `core/bayesian_irt.py` | Maintains Bayesian competency estimates and uncertainty values. |
| `core/bkt.py` | Tracks skill mastery probabilities using Bayesian Knowledge Tracing. |
| `core/knowledge_graph.py` | Represents technical skills and their relationships. |
| `core/question_bank.py` | Retrieves or generates interview questions based on the selected assessment action. |
| `core/evaluator.py` | Evaluates candidate answers, including coding submissions, and generates assessment scores. |
| `core/session.py` | Manages interview sessions, question history, competency updates, and final reports. |
| `core/benchmark.py` | Contains functions for benchmark evaluation and result generation. |
| `interview/` | Contains the controller connecting the Streamlit interface with the interview session. |
| `llm/` | Handles Gemini API integration and candidate profile extraction. |
| `resume/` | Processes resumes in PDF, DOCX, and TXT formats. |
| `models/` | Stores the selected PPO checkpoint, category Q-learning policy, and benchmark model checkpoints. |
| `reports/` | Contains research evaluation tables and visualizations used by the dashboard. |
| `tests/` | Contains unit and integration tests for the application. |
| `demo_interview.py` | Runs an offline interview demonstration using the trained PPO policy and mocked Gemini responses. |
| `train_category_policy.py` | Trains the category-selection Q-learning policy. |
| `run_research_benchmark.py` | Executes benchmark evaluations of the reinforcement learning algorithms. |

The trained PPO policy and category Q-learning policy are stored in the `models/` directory.

The application uses these policies to make adaptive interview decisions.

## Installation and Setup

### 1. Prerequisites

Before running the application, ensure that the following are installed:

- Python 3.11
- pip
- An internet connection for Gemini API functionality
- Tesseract OCR (optional, for scanned PDF resumes)

### 2. Open the Project Directory

Open PowerShell or a terminal in the project folder:

```powershell
cd C:\Users\user\Adaptive_AI_Interview_RL\Adaptive_AI_Interview_Final
```

Replace the path with the actual location of the project on your system.

### 3. Create a Virtual Environment

Create a Python virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Upgrade pip and install the required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure the Gemini API

Create a `.env` file using the provided example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and enter the Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key is required for live Gemini-based question generation and answer evaluation.

If no API key is configured, the application uses its available local question bank and fallback evaluation mechanisms.

For scanned PDF resumes, Tesseract OCR must be installed. If necessary, specify its executable path using `TESSERACT_CMD` in `.env`.

Do not upload the `.env` file containing your API key to GitHub.

## Running the Application

From the project directory, with the virtual environment activated, execute:

```powershell
streamlit run app.py
```

The application opens in the default web browser.

If it does not open automatically, navigate to:

```text
http://localhost:8501
```

The user can then select the required application mode, provide candidate information, configure interview settings, and begin the assessment.

## Running Tests

To execute the unit and integration tests:

```powershell
python -m unittest discover -s tests -v
```

To run the offline interview demonstration:

```powershell
python demo_interview.py
```

The demonstration uses the trained PPO policy and mocked Gemini responses, allowing the interview workflow to be tested without making live API requests.

## Model Training and Evaluation

The application uses a pretrained PPO policy for selecting interview actions. The policy is implemented using a custom PyTorch actor–critic network.

The PPO policy selects an action representing the skill, difficulty level, and question type. The trained policy is loaded from the `models/` directory.

A separate Q-learning policy is used for question-category selection. It can be trained using:

```powershell
python train_category_policy.py --episodes 50000 --seed 20261103
```

The research component includes comparisons between:

- PPO
- DQN
- Double DQN
- Dueling Double DQN
- Linear Q-learning
- Baseline question-selection strategies

Evaluation metrics include cumulative reward, weighted RMSE, decision accuracy, and convergence behavior.

The Research Dashboard provides access to the available evaluation results and visualizations.

## Conclusion

The Adaptive AI Interview Simulator combines reinforcement learning, Bayesian competency estimation, Knowledge Graphs, and language models to conduct interviews that adapt to individual candidates.

The system provides a complete workflow from candidate profiling and question selection to answer evaluation and competency reporting, with additional tools for comparing interview strategies and analyzing their performance.
