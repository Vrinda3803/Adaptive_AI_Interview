# Reproducible policy evaluation

## Scope and claims

This is an **exploratory synthetic benchmark** for selecting the next skill, difficulty, and question type. It evaluates trained policies against matched synthetic candidate episodes drawn from held-out skill domains. It does not evaluate Gemini question quality, human interviewer agreement, candidate experience, predictive validity, or hiring fairness. The benchmark is not evidence for automated hiring decisions.

The primary endpoint for future reruns is weighted RMSE between the final Bayesian IRT estimate and hidden simulated competency (lower is better). Secondary endpoints are episode return, threshold decision accuracy, simulation-defined success rate, skill coverage, questions asked, and invalid actions. Because this protocol was added after earlier project analyses, reruns are exploratory rather than preregistered.

## Methods compared

- PPO, DQN, Double DQN, Dueling DDQN, and linear Q-learning: three saved training-seed checkpoints per algorithm.
- Random: uniformly samples a legal action.
- Uncertainty: selects a legal action for the skill with highest posterior standard deviation.
- MaxEIG: selects the legal action with the largest skill-weighted expected-information-gain observation feature.

All strategies use the same environment, action mask, reward, stopping rule, and common candidate episode seeds. Response randomness is generated from the episode-seeded random generator. The evaluation domains are Cybersecurity, Mobile Development, and Specialized Technologies. No hyperparameter selection is performed by the evaluation script.

## Statistical reporting

The script retains one row per method, model seed, and episode seed. Summary intervals use a hierarchical percentile bootstrap: resample trained checkpoint seeds and then episodes within each sampled seed. Rule-based baselines have no training seeds and use episode-only bootstrap. Paired differences versus MaxEIG use identical episode seeds and report percentile intervals; those intervals average over the available trained checkpoints and therefore condition on these checkpoint seeds. No significance claims are made from the three-seed checkpoint set.

Three training seeds are too few for strong claims about training variability. The provided results should be treated as preliminary. The new paired benchmark is not directly comparable to the older archived aggregate CSVs: the domain split, episode-seeded response noise, checkpoint aggregation, and uncertainty reporting differ. A paper submission should add independently trained seeds, justify sample sizes, report compute and hyperparameter selection, and validate the simulator against human-labeled interview data. Report metric definitions, missing outcomes, and any deviations from the protocol.

## Run and outputs

From the application root:

```powershell
python run_research_benchmark.py --episodes 300 --bootstrap-samples 10000
```

Outputs are written to `reports/seeded_benchmark/`: `episodes.csv`, `summary.csv`, `paired_vs_maxeig.csv`, `manifest.json`, and three figures. The manifest records the episode setup and SHA-256 for each evaluated checkpoint. Use a different output directory if preserving an earlier run.
