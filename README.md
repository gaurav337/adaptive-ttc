# Project Technical Brief

## Learning Compute Allocation Policies from Execution Feedback



---



## 1. The One-Sentence Version



This project studies whether the way a language model fails on a coding problem tells you how much extra computation is needed to fix that failure — and builds a system that uses that information to allocate compute efficiently instead of wasting it.



---



## 2. The Problem Being Solved



### What is Test-Time Compute?



Modern large language models can improve their answers by spending more computation during inference — the moment they are actually generating a response. This is called test-time compute (TTC).



Instead of generating one answer and stopping, the model can:

- Try again with a different approach

- Generate multiple candidate solutions and pick the best

- Iteratively debug a failed solution

- Reason step-by-step before committing to an answer



All of these strategies improve accuracy, but they all cost more tokens, more time, and more money.



### The Inefficiency That Exists Today



Current systems allocate the same reasoning budget to every failure. If a problem fails, the system retries it with, say, 2000 extra tokens — regardless of whether the failure was a trivial one-line syntax error or a completely wrong algorithmic approach.



**Example A:** A generated function has a missing colon on line 3. The rest of the code is correct. Fixing this requires approximately 200 tokens.



**Example B:** A generated function solves a sorting problem using O(N²) bubble sort, but the test cases require an O(N log N) solution to pass within the time limit. No amount of minor patching fixes this. The entire algorithm needs to be rethought.



If the system allocates 2000 tokens to both failures equally:

- Example A wastes 1800 tokens that were not needed

- Example B wastes all 2000 tokens on a failure that is not recoverable without a fundamentally different approach



This is the inefficiency this project aims to fix.



### The Central Research Question



> Can the information available immediately after a failure — the exception type, traceback, runtime, code structure — predict how much additional compute is worth spending, before spending it?



---



## 3. The Core Scientific Hypothesis



The hypothesis has three parts:



- **Part 1:** Different failure modes require fundamentally different amounts of compute to repair.

- **Part 2:** The failure signature — the structured information available immediately after execution — contains enough signal to predict this.

- **Part 3:** A lightweight predictor trained on historical failure-to-repair-cost data can route each new failure to the right compute budget adaptively.



Formally, the chain being tested is:



```

Failure Signature → Recoverability Prediction → Budget Routing → Repair Attempt

```



If this chain holds, the system can avoid two categories of waste:

- Spending large budgets on failures that only needed small corrections

- Spending any budget on failures that are fundamentally unrecoverable



---



## 4. What This Project Is NOT



This is easy to confuse with other work. To be explicit:



| This project is NOT about... | This project IS about... |

|---|---|

| Improving code generation quality | Deciding how much compute to spend per failure |

| Building a better coding agent | Resource allocation during inference |

| Training a new language model | Using existing models as fixed black boxes |

| Prompt engineering | Studying failure signatures and their predictive value |

| Self-debugging systems | Compute economics for inference |



The language models are treated as black boxes. They are not modified, fine-tuned, or trained in any way. They are run, and their failures are studied.



---



## 5. Dataset: LiveCodeBench



### Why Not HumanEval?



HumanEval is the most well-known code generation benchmark. It is NOT used in this project for a specific reason:

- Modern 7B–8B models solve 80%+ of HumanEval problems

- This project studies failures — if only 10–20% of problems fail, the dataset of failures is too small to train any predictor

- HumanEval is heavily contaminated — many models have seen it during training, making failure rates artificially low



### Why LiveCodeBench?



LiveCodeBench is a continuously updated competitive programming benchmark:

- Problems are sourced from recent programming contests held after model training cutoffs

- Hard algorithmic reasoning tasks where modern models still fail significantly

- Fully objective evaluation — a solution either passes all tests or it does not

- No contamination risk since problems are added after training



This provides enough failures to build a meaningful dataset.



---



## 6. Models Used



Two models are used deliberately to test generalization:



**Model 1 — Qwen2.5-Coder-7B**

- Standard code generation model

- No built-in chain-of-thought reasoning

- Represents baseline coding performance



**Model 2 — DeepSeek-R1-Distill-Qwen-7B**

- Native reasoning model

- Produces explicit thinking traces before answering

- Tests whether failure signatures differ for reasoning vs. non-reasoning models



Using two architecturally different models answers a critical question: do patterns learned from one model's failures transfer to another model's failures? If yes, the approach generalizes. If no, the system must be retrained per model — which is still useful but less impactful.



---



## 7. The Full System Pipeline



The system has six sequential stages. Each stage feeds into the next.



---



### Stage 1: Baseline Generation



- **Input:** Each LiveCodeBench problem (description, function signature, test cases)

- **Models:** Qwen2.5-Coder-7B and DeepSeek-R1-Distill-Qwen-7B, run separately

- **Settings:** Temperature = 0, greedy decoding (fully deterministic)

- **Output:** One generated Python function per problem

- **Data recorded:** Generated code, token count, inference latency, lines of code



Greedy decoding is used so results are reproducible. Temperature = 0 means the model always picks the highest probability token — there is no randomness.



---



### Stage 2: Execution Evaluation



The generated code is run against LiveCodeBench's official test suite.



- **Outcome A — PASS:** Problem is solved. No further processing.

- **Outcome B — FAIL:** Execution information is captured and passed to Stage 3.



For failures, the following is recorded:

- Exception type: SyntaxError, AssertionError, IndexError, ValueError, TimeoutError, RecursionError

- Runtime in milliseconds

- Traceback depth (how many levels deep the error stack goes)

- Failure line number



No LLM judge is used anywhere. Correctness is fully objective.



---



### Stage 3: Failure Signature Extraction



Every failed solution is converted into a structured numerical feature vector. This is the input to all downstream predictors. Only information available at the moment of failure is used — no future information is allowed.



**Execution features:**

- Exception type (one-hot encoded)

- Traceback depth

- Runtime in ms

- Failure line number



**Code structure features:**

- Token count of generated solution

- Lines of code

- AST complexity score

- Cyclomatic complexity (number of independent code paths)



**Problem features:**

- Prompt length in tokens

- Problem difficulty level from LiveCodeBench



The resulting vector is the failure signature. It represents everything observable about how the model failed.



---



### Stage 4: Ground-Truth Repair Discovery



This stage builds the training dataset. Every failed solution is run through a repair ladder with increasing token budgets. The goal is to find the minimum budget that successfully repairs each failure.



**Critical methodological decision:** The repair prompt is fixed and frozen for all experiments. The only variable being changed across levels is the token budget — not the prompt format, not the instructions. This controls the main confound.



**Fixed repair prompt template (same for all experiments):**

```

Original Problem: [problem description]

Failed Code: [the generated solution]

Execution Traceback: [the full error output]

Instruction: Fix the code and output only the corrected Python function.

```



**The repair budget ladder (exact token caps enforced in code):**



| Level | Exact Token Cap | Suitable For |

|---|---|---|

| Level 0 | 0 tokens | Baseline — no repair |

| Level 1 | 500 tokens | Surface errors: syntax, off-by-one, missing return |

| Level 2 | 1500 tokens | Logic errors needing more reasoning space |

| Level 3 | 3000 tokens | Complex algorithmic failures requiring restructuring |



**How labeling works:**

- Try Level 1 first. If it passes → label is Level 1.

- If Level 1 fails → try Level 2. If it passes → label is Level 2.

- If Level 2 fails → try Level 3. If it passes → label is Level 3.

- If Level 3 fails → label is **Unrecoverable**.



This produces the core dataset: `Failure Signature → Minimum Successful Repair Level`



---



### Stage 5: Recoverability Prediction



- **Task type:** Binary classification

- **Input:** Failure signature feature vector (from Stage 3)

- **Output:** RECOVERABLE or UNRECOVERABLE

- **Model:** XGBoost Classifier

- **Purpose:** Prevent spending any compute on failures that will not succeed regardless of budget



This is the first gate. If a failure is predicted unrecoverable, the system stops and reports the problem as unsolvable. No repair tokens are spent. This alone saves significant compute across a large benchmark.



---



### Stage 6: Repair Strategy Routing



- **Task type:** Ordinal classification (ordered classes: L1 < L2 < L3)

- **Input:** Failure signature of a recoverable failure

- **Output:** Predicted repair level — Level 1, Level 2, or Level 3

- **Model:** XGBoost with ordinal-aware error metric

- **Purpose:** Route each recoverable failure to the cheapest repair strategy likely to succeed



**Why ordinal matters:** The classes are ordered, not categorical. Predicting Level 3 when the correct answer is Level 1 is a larger mistake than predicting Level 2. Standard multi-class classification treats all errors equally — this is incorrect for ordered budget levels. An ordinal-aware metric (mean absolute level error) is used.



---



## 8. Baselines



The adaptive policy is compared against five baselines:



| Baseline | What it does | Why it exists |

|---|---|---|

| No Repair | Generate once, stop | Lower bound. Zero extra tokens. |

| Always Level 1 | Spend 500 tokens on every failure | Tests whether cheap fixed repairs are enough |

| Always Level 3 | Spend 3000 tokens on every failure | Upper bound on cost. Best fixed-budget accuracy. |

| Random Routing | Randomly assign L1, L2, or L3 | Tests whether predictor beats chance |

| Oracle Routing | Always use the true minimum level | Theoretical ceiling — best any router could possibly do |



---



## 9. Evaluation Metrics



| Metric | Formula / Definition | What it answers |

|---|---|---|

| Pass@1 | % of all problems solved (initial + repair) | Does the system get the right answer? |

| Recovery Rate | % of failed problems successfully repaired | How good is the repair system? |

| Total Token Cost | Sum of all tokens across all problems | How expensive is the system? |

| Compute-Normalized Accuracy (CNA) | Pass@1 ÷ Total Tokens Consumed | Primary efficiency metric |

| Inference Latency | Total wall-clock time for all generation | Real-world speed cost |

| Recoverability AUROC | Area under ROC curve for binary classifier | How well does the system identify unrecoverable failures? |

| Budget Prediction MAE | Mean absolute error in predicted vs actual level | How close are the routing predictions? |

| Brier Score | Mean squared error of probability calibration | Are the confidence scores trustworthy? |



**The primary claim being tested:** The adaptive system achieves similar Pass@1 to Always-Level-3 while consuming significantly fewer total tokens, resulting in a higher Compute-Normalized Accuracy.



---



## 10. Interpretability Analysis (SHAP)



SHAP (SHapley Additive exPlanations) is run on both XGBoost classifiers to answer:

- Which features most strongly predict recoverability?

- Which features most strongly predict repair cost?

- Do the two models (Qwen and DeepSeek) produce similar feature importance rankings?



**Hypothesized findings (to be empirically confirmed):**

- Exception type is likely the single most important feature

- SyntaxError → strongly predicts Level 1 (cheap fix)

- TimeoutError → strongly predicts Unrecoverable or Level 3

- High AST complexity → correlates with higher repair cost

- Runtime close to timeout limit → predicts algorithmic failure (expensive)



This analysis transforms the project from "we built a predictor" into "we discovered which execution signals are predictive of repair cost" — which is the actual scientific contribution.



**Additional analysis — error analysis on mispredictions:** When the recoverability classifier predicts RECOVERABLE but the failure is actually unrecoverable, what do those failures look like? This paragraph in the paper turns a classification result into a finding about LLM failure modes.



---



## 11. Generalization Experiments



### Cross-Model Transfer

Train XGBoost on Qwen failures. Test on DeepSeek failures.



> Do failure signatures generalize across model architectures, or is the failure-to-repair-cost relationship model-specific?



### Cross-Difficulty Transfer

Train on easier LiveCodeBench problems. Test on harder ones.



> Does the failure-to-repair-cost relationship remain stable as problem difficulty increases, or does the mapping break down?



These experiments determine the practical scope of the findings — whether this approach can be deployed broadly or only within narrow conditions.



---



## 12. What Success Looks Like



The project is successful if:

- The adaptive router achieves within 2–3% of the Pass@1 of Always-Level-3

- The adaptive router consumes 30–50% fewer total tokens than Always-Level-3

- The Compute-Normalized Accuracy of the adaptive system exceeds all fixed-budget baselines

- SHAP analysis reveals at least 2–3 features with clear, interpretable predictive value

- The recoverability classifier achieves AUROC above 0.75



**The goal is NOT to beat the accuracy of the most expensive system. The goal is to match its accuracy at a fraction of the cost.**



---



## 13. What Failure Looks Like (and Why It Is Still Useful)



Even if the predictions are poor, the project produces value:

- The dataset (failure signatures mapped to repair costs) is novel and reusable by other researchers

- The recoverability analysis reveals what fraction of LLM coding failures are fundamentally unrecoverable — a finding with independent scientific value

- Null results are also publishable as they inform the field about the limits of execution feedback as a signal



---



## 14. Known Limitations and How They Are Handled



| Limitation | How it is handled |

|---|---|

| Repair prompt sensitivity | Fixed frozen prompt template controls this confound completely |

| Single repair attempt per level | Acknowledged scope limitation; future work can test Best-of-N within a budget |

| Two models may not be enough | Framed as preliminary generalization study, not a strong generalization claim |

| XGBoost may miss non-linear interactions | SHAP + feature ablation reveals if this is an issue |

| LiveCodeBench distribution may shift | Benchmark version is recorded; reproducibility maintained via commit hash |



---



## 15. Summary: What Is Being Built



**Step 1:** Run two LLMs on hundreds of hard coding problems from LiveCodeBench. Record every failure.



**Step 2:** For each failure, record everything observable — what error occurred, how deep the traceback was, how long it ran, how complex the code structure was.



**Step 3:** For each failure, try to repair it at increasing exact token budgets (500 / 1500 / 3000). Record the minimum budget that worked. Label failures that never got fixed as unrecoverable.



**Step 4:** Train an XGBoost classifier to predict, from failure features alone, whether a failure is recoverable.



**Step 5:** Train a second XGBoost classifier to predict, from failure features alone, which budget level is needed.



**Step 6:** Compare this adaptive system against five fixed-budget baselines on accuracy and token cost.



**Step 7:** Use SHAP to explain what the classifiers learned, run generalization experiments, and publish the findings.



**The output:** A dataset, two trained classifiers, an empirical analysis of LLM failure modes, and evidence for or against the hypothesis that execution feedback can guide efficient test-time compute allocation.
