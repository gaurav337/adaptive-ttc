# Learning Compute Allocation Policies from Execution Feedback
### A Complete Technical Explanation of the Project

---

## What I Am Building — The One-Liner

I am building a system that watches how a language model *fails* on a coding problem, extracts structured information from that failure, and uses it to predict exactly how much extra computation is needed to fix the failure — before spending that computation. The system then routes each failure to the cheapest repair strategy that is likely to work.

---

## Why This Project Exists — The Problem

### Background: What is Test-Time Compute?

Modern large language models (LLMs) can improve their own answers by spending more computation *at inference time* — the moment they are actually generating a response. This is called **Test-Time Compute (TTC)**.

Instead of generating one answer and stopping, the model can:
- Try again with a different approach
- Generate multiple candidate solutions and pick the best one
- Iteratively debug a failed solution using its own error output
- Reason step-by-step at length before committing to an answer

All of these strategies improve accuracy — but they all cost more tokens, more time, and more money.

### The Inefficiency Being Solved

Current systems that use test-time compute for code generation are dumb about *how much* compute they allocate. When a model fails on a problem, the system retries with a fixed budget — say, 2000 extra tokens — regardless of whether the failure was a trivial one-line syntax error or a completely wrong algorithmic approach.

**Concrete Example A:**
A generated Python function has a missing colon on line 3. The rest of the code is correct. Fixing this requires approximately 100–200 tokens. The system allocates 2000. Result: ~1800 tokens wasted.

**Concrete Example B:**
A generated function solves a sorting problem using O(N²) bubble sort, but the test cases involve arrays of 100,000 elements and require O(N log N) performance. The entire algorithm is wrong — no surface-level patching fixes it. The system allocates 2000 tokens. Result: all 2000 tokens wasted on a failure that is fundamentally unrecoverable without a full rewrite.

If the system knew *in advance* that Example A only needs 200 tokens and Example B is unrecoverable, it could:
- Save the 1800 wasted tokens in Example A
- Save the 2000 wasted tokens in Example B
- Spend those saved tokens on other problems that genuinely need them

**The core insight is:** the failure itself — the exception type, the traceback, the runtime, the structure of the code — contains enough signal to predict how expensive the fix will be. The project tests whether this is true and builds a system around it.

---

## The Central Scientific Hypothesis

The hypothesis has three parts, forming a chain:

**Part 1:** Different failure modes require fundamentally different amounts of compute to repair.
- A SyntaxError is structurally different from a TimeoutError.
- A shallow traceback (1 level deep) is structurally different from a deep one (8 levels deep).
- These differences should correspond to differences in repair cost.

**Part 2:** The failure signature — the structured information available immediately after execution — contains enough signal to predict repair cost.
- You don't need to attempt the repair to know how expensive it will be.
- The error output already encodes this information, if you know how to read it.

**Part 3:** A lightweight predictor trained on historical failure-to-repair-cost data can route each new failure to the right compute budget adaptively, with better efficiency than any fixed-budget strategy.

The chain being tested:

```
Execution Failure
      ↓
Failure Signature Extraction  (what exception? how deep? how long did it run? how complex is the code?)
      ↓
Recoverability Prediction     (is this failure fixable at all?)
      ↓
Budget Routing                (if fixable, how many tokens does it need?)
      ↓
Targeted Repair Attempt       (spend only what is needed)
```

---

## What This Project Is NOT

This is easy to confuse with adjacent work, so being explicit:

| This project is NOT about... | This project IS about... |
|---|---|
| Improving code generation quality | Deciding how much compute to spend *per failure* |
| Building a better coding agent | Resource allocation during inference |
| Training or fine-tuning a new language model | Using existing models as fixed black boxes |
| Prompt engineering or better repair prompts | Studying failure signatures as a predictive signal |
| Self-debugging systems | Compute economics for inference |

The language models are treated as completely fixed black boxes. They are not modified, fine-tuned, or trained in any way. They are run, and their failures are studied.

---

## Dataset: LiveCodeBench

### Why Not HumanEval (the most common benchmark)?

HumanEval is the most well-known code generation benchmark. It is NOT used in this project because:
- Modern 7B–8B models solve 80%+ of HumanEval problems — this project studies *failures*, so if only 10–20% of problems fail, the failure dataset is too small to train any meaningful predictor
- HumanEval is heavily contaminated — many models have seen it during training, making failure rates artificially low and unrepresentative

### Why LiveCodeBench?

LiveCodeBench is a continuously updated competitive programming benchmark:
- Problems are sourced from recent competitive programming contests held *after* model training cutoffs — so models cannot have memorized solutions
- Tasks are hard algorithmic reasoning problems where modern 7B models still fail significantly (failure rates high enough to build a training dataset)
- Evaluation is fully objective — a solution either passes all hidden test cases or it does not; no LLM judge, no rubric
- No contamination risk since problems are added post-training
- The benchmark version is pinned via commit hash, ensuring full reproducibility

---

## Models Used

Two models are deliberately chosen to test generalization across architectures:

### Model 1 — Qwen2.5-Coder-7B
- A standard code generation model
- No built-in chain-of-thought or reasoning traces
- Generates code directly without thinking tokens
- Represents baseline coding LLM behavior

### Model 2 — DeepSeek-R1-Distill-Qwen-7B
- A native reasoning model distilled from DeepSeek-R1
- Produces explicit `<think>...</think>` internal reasoning traces before writing code
- Architecturally different from Qwen — uses significantly more tokens per problem due to the thinking phase
- Tests whether failure signatures and repair cost patterns from one architecture transfer to another

**Why two models matter:** If the failure-to-repair-cost relationship learned from Qwen's failures also predicts DeepSeek's repair costs, the approach generalizes across architectures and could be deployed broadly. If not, the system needs to be retrained per model — which is still useful but less impactful.

**Important technical note on DeepSeek-R1 (the thinking token problem):**  
DeepSeek-R1 generates its output in two phases — a `<think>` block (internal reasoning, which can be hundreds to thousands of tokens) followed by the actual code. When enforcing token budget caps on repair attempts, a hard cap of 500 total tokens on DeepSeek-R1 might allow only ~300 tokens for thinking and ~200 tokens for code output — which is insufficient to write a complete function. The analysis will parse `<think>` tokens separately from coding tokens to study this problem explicitly. This is a known limitation that is treated as a finding rather than ignored.

---

## The Full System Pipeline — All Six Stages

### Stage 1: Baseline Code Generation

- **Input:** Every problem in the LiveCodeBench dataset (problem description + function signature + hidden test cases)
- **Models:** Qwen2.5-Coder-7B and DeepSeek-R1-Distill-Qwen-7B, run separately and independently
- **Decoding Settings:** Temperature = 0, greedy decoding (fully deterministic — the model always picks the highest probability next token, no randomness)
- **Output:** One generated Python function per problem per model
- **Data Recorded:**
  - The full generated code
  - Total token count of the generation
  - Inference latency in milliseconds
  - Lines of code
  - Whether DeepSeek's output contains `<think>` traces and how long they are

Greedy decoding is used so results are perfectly reproducible. Running the same model on the same problem twice always produces the same code.

---

### Stage 2: Execution and Evaluation

Every generated function is executed against LiveCodeBench's official test suite in an isolated sandbox.

**Outcome A — PASS:** The problem is solved. It is marked as solved and no further processing happens for this problem.

**Outcome B — FAIL:** The execution failed. Full failure information is captured and passed to Stage 3.

For every failure, the following is recorded verbatim:
- **Exception type:** The exact Python exception class raised (`SyntaxError`, `AssertionError`, `IndexError`, `ValueError`, `TypeError`, `TimeoutError`, `RecursionError`, `MemoryError`, `NameError`, `AttributeError`, etc.)
- **Full traceback text:** The complete stack trace as printed by Python
- **Traceback depth:** How many stack frames deep the error occurred
- **Runtime in milliseconds:** How long execution ran before failing
- **Failure line number:** Which line of the generated code triggered the error
- **Test case that triggered the failure:** Which test input caused the failure first

No LLM judge is used anywhere in this pipeline. Correctness is entirely objective.

---

### Stage 3: Failure Signature Extraction

Every failed solution is converted into a structured numerical feature vector. This is the input to every downstream predictor. Only information available at the *exact moment of failure* is used — no future information is allowed, and no repair is attempted yet.

The feature vector contains the following groups of features:

**Execution Features (from running the code):**
- Exception type: one-hot encoded across all exception categories observed in the dataset
- Traceback depth: integer count of stack frames
- Runtime in milliseconds: continuous float
- Failure line number: integer (normalized by total lines of code)
- Whether the failure was a timeout (binary flag, since timeout signals algorithmic failure specifically)

**Code Structure Features (from analyzing the generated code statically):**
- Total token count of the generated solution
- Lines of code (raw count)
- AST complexity score: number of nodes in the abstract syntax tree of the code
- Cyclomatic complexity: number of independent code paths (branches, loops, conditions)
- Number of function definitions in the code
- Nesting depth: maximum depth of nested control flow (loops inside loops, etc.)
- Whether the code contains recursion (binary flag)

**Problem Features (from the original problem statement):**
- Prompt length in tokens
- Problem difficulty level as labeled by LiveCodeBench (Easy / Medium / Hard, encoded ordinally)

**Embedding Features (semantic representation — the upgrade over pure tabular features):**
- A dense vector embedding of the full traceback text, generated by a frozen code embedding model (StarEncoder or CodeBERT)
- A dense vector embedding of the failed code, generated by the same frozen model
- These embeddings capture semantic meaning that the tabular features cannot — e.g., the difference between a `TimeoutError` caused by a missing break statement vs. one caused by using bubble sort on a large array

The tabular features and embedding features are concatenated into a single feature vector. This combined representation is what gets fed to the classifiers.

---

### Stage 4: Ground-Truth Label Discovery (Building the Training Dataset)

This is the most critical stage — it builds the dataset that everything else trains on.

Every failed solution is run through a **repair ladder** with increasing token budgets. The goal is to discover the *minimum budget that successfully repairs each failure*.

**The methodological decision that controls the main confound:**  
The repair prompt template is completely frozen and identical for every experiment. The only variable being changed across budget levels is the maximum token budget allowed for the repair output. This isolates the effect of compute (tokens) from the effect of prompting (instructions).

**Fixed Repair Prompt Template (identical for every experiment, every model, every budget level):**
```
Original Problem: [problem description]
Failed Code: [the full generated solution that failed]
Execution Traceback: [the complete error output]
Instruction: Fix the code and output only the corrected Python function.
```

**The Repair Budget Ladder:**

| Level | Token Cap | Interpretation |
|---|---|---|
| Level 0 | 0 tokens | No repair attempted — baseline |
| Level 1 | 500 tokens | Surface-level fixes: syntax errors, off-by-one, missing return statement, wrong variable name |
| Level 2 | 1500 tokens | Logic errors requiring more reasoning space: wrong conditional, missing edge case, incorrect data structure |
| Level 3 | 3000 tokens | Complex algorithmic failures requiring full restructuring: wrong algorithm, fundamental approach change |

**How ground-truth labels are assigned (sequential testing):**
1. Try Level 1 repair (500-token cap). If the repaired code passes all tests → label this failure as **Level 1**.
2. If Level 1 repair fails → try Level 2 repair (1500-token cap). If it passes → label as **Level 2**.
3. If Level 2 repair fails → try Level 3 repair (3000-token cap). If it passes → label as **Level 3**.
4. If Level 3 repair also fails → label as **Unrecoverable**.

**Critical definitional note — what "Unrecoverable" actually means:**
"Unrecoverable" does NOT mean the problem is mathematically unsolvable. It means: *unrecoverable via zero-shot execution feedback within the tested budget constraints using the fixed repair prompt.* A different prompt, a larger model, or an agentic loop might recover it. This scoping is stated explicitly everywhere the term appears. "Unrecoverable" is always shorthand for "unrecoverable under these specific experimental conditions," never an absolute claim about the problem itself.

This sequential testing ensures the label is always the *minimum* sufficient budget, not just any budget that works.

**Output of Stage 4:** A dataset where each row is:
```
[failure signature feature vector] → [minimum repair level label: L1 / L2 / L3 / Unrecoverable]
```

This dataset is the foundation for Stages 5 and 6.

---

### Stage 5: Recoverability Prediction (The First Gate)

- **Task Type:** Binary classification
- **Input:** The failure signature feature vector (from Stage 3)
- **Output:** RECOVERABLE or UNRECOVERABLE
- **Model:** XGBoost Classifier with embedding features concatenated to tabular features
- **Trained on:** The labeled dataset from Stage 4
- **Purpose:** Before spending any repair tokens, filter out failures that are fundamentally unrecoverable

If a failure is predicted UNRECOVERABLE, the system immediately stops and reports the problem as unsolvable. Zero repair tokens are spent. Across a large benchmark, this alone saves a significant fraction of total inference cost.

**Evaluation metrics for this classifier:**
- AUROC (Area Under ROC Curve) — primary metric; target above 0.75
- Precision and Recall for the UNRECOVERABLE class specifically
- Brier Score — measures whether the predicted probabilities are well-calibrated (i.e., when the model says 80% confident it's unrecoverable, it should be right ~80% of the time)

---

### Stage 6: Repair Budget Routing (The Second Gate)

- **Task Type:** Ordinal classification (ordered classes: L1 < L2 < L3)
- **Input:** Failure signature of a failure that passed Stage 5 (predicted RECOVERABLE)
- **Output:** Predicted repair level — Level 1, Level 2, or Level 3
- **Model:** XGBoost Classifier with ordinal-aware loss
- **Purpose:** Route each recoverable failure to the cheapest repair strategy likely to succeed

**Why ordinal classification and not standard multi-class:**  
The three budget levels are not unordered categories — they have a natural order (L1 < L2 < L3). In standard multi-class classification, predicting L3 when the correct answer is L1 counts the same as predicting L2 when the correct answer is L1. This is mathematically wrong. Predicting L3 instead of L1 wastes 2500 tokens unnecessarily. Predicting L2 instead of L1 wastes only 1000 tokens. The error metric used is **mean absolute level error**, which penalizes larger mispredictions more heavily than smaller ones.

**Evaluation metrics for this classifier:**
- Mean Absolute Level Error (MAE in ordinal levels) — primary metric
- Per-class precision and recall
- Confusion matrix to visualize which levels get confused with which

---

## Baselines

The adaptive policy is compared against five baselines to isolate the value of each component:

| Baseline | What It Does | Why It Is Included |
|---|---|---|
| **No Repair** | Generate once and stop; spend zero extra tokens | Lower bound — zero compute cost, lowest accuracy |
| **Always Level 1** | Spend exactly 500 repair tokens on every failure | Tests whether cheap fixed repairs are sufficient |
| **Always Level 3** | Spend exactly 3000 repair tokens on every failure | Upper bound on cost — best accuracy any fixed strategy can achieve |
| **Random Routing** | Randomly assign each failure to L1, L2, or L3 | Tests whether the predictor beats random chance |
| **Oracle Routing** | Always use the true minimum sufficient level (from Stage 4 labels) | Theoretical ceiling — best any router could possibly achieve |
| **Best-of-N Baseline** | For a given token budget, generate N candidates and pick the first one that passes tests | Tests whether adaptive single-pass routing beats naive parallel sampling |

The Best-of-N baseline is critical: if generating three 500-token repair attempts and picking the passing one beats the adaptive router, then the core claim (that execution feedback can route single-pass compute better than brute-force sampling) is weakened.

---

## Evaluation Metrics

| Metric | Definition | What It Answers |
|---|---|---|
| **Pass@1** | Percentage of all problems solved (initial generation + repair) | Does the system get the right answer? |
| **Recovery Rate** | Percentage of failed problems successfully repaired | How effective is the repair system specifically? |
| **Total Token Cost** | Sum of all tokens consumed across all problems | How expensive is the system? |
| **Compute-Normalized Accuracy (CNA)** | Pass@1 ÷ Total Tokens Consumed | Primary efficiency metric — accuracy per token |
| **Inference Latency** | Total wall-clock time for all generation | Real-world speed cost |
| **Sandbox Execution Overhead** | Wall-clock time spent in the execution sandbox per failure (separate from generation time) | Measures the latency cost of the routing pipeline itself |
| **Recoverability AUROC** | Area under ROC for the binary recoverability classifier | How well does the system identify hopeless failures? |
| **Budget Prediction MAE** | Mean absolute error in predicted vs. actual repair level | How close are the routing predictions? |
| **Brier Score** | Mean squared error of probability calibration | Are the confidence scores trustworthy? |

**The primary claim being tested:**
> The adaptive system achieves similar Pass@1 to Always-Level-3 while consuming 30–50% fewer total tokens, resulting in a higher Compute-Normalized Accuracy.

---

## Interpretability: SHAP Analysis

SHAP (SHapley Additive exPlanations) is applied to both XGBoost classifiers to produce human-readable explanations of what the models learned.

SHAP answers:
- Which features most strongly predict recoverability?
- Which features most strongly predict repair budget?
- Do the Qwen model and DeepSeek model produce similar or different SHAP importance rankings?

**Hypothesized SHAP findings (to be empirically confirmed or rejected):**
- `SyntaxError` exception type → strongly predicts Level 1 (cheap surface fix)
- `TimeoutError` exception type → strongly predicts Unrecoverable or Level 3 (algorithmic failure)
- High AST complexity + high cyclomatic complexity → correlates with higher repair budget
- Runtime very close to the timeout limit → predicts algorithmic failure (expensive or unrecoverable)
- High traceback depth → correlates with complex logic errors

**Error analysis on mispredictions:**  
When the recoverability classifier predicts RECOVERABLE but the failure is actually unrecoverable, what do those failures look like? Clustering mispredictions by their failure signature embeddings will reveal failure modes the classifier systematically misses — this is a finding about LLM failure modes, not just a classifier performance number.

**AssertionError-specific ablation (the anticipated hardest case):**  
`AssertionError` is the most ambiguous exception class in this dataset. It fires whenever the output is simply wrong — which could mean a missing `<` vs `<=` in a comparison (a 50-token Level 1 fix) or a completely wrong dynamic programming recurrence relation (a 3000-token Level 3 rewrite). The tabular feature `exception_type = AssertionError` carries almost no repair cost signal on its own. The ablation study comparing embedding features vs. tabular-only features specifically isolates AssertionError cases to show how much of the classifier's gain on this class comes from reading the actual semantic content of the traceback vs. the exception label alone. This is expected to be where embeddings provide the largest measurable lift.

---

## DeepSeek-R1 Thinking Token Analysis

A dedicated analysis is run specifically on DeepSeek-R1's outputs:

1. Parse every DeepSeek output to separate `<think>` tokens from actual Python code tokens
2. For each failure, record: thinking token count, coding token count, total token count
3. Train a secondary predictor: given the failure signature, predict the thinking budget required vs. the coding budget required separately
4. Show via SHAP analysis which failure types correlate with heavy thinking requirements vs. light thinking requirements

**Expected finding:** Failures involving complex algorithmic reasoning (TimeoutError, wrong output on large inputs) will require massive thinking budgets but moderate coding budgets. Simple failures (SyntaxError, NameError) will require near-zero thinking tokens since the fix is obvious. This would be an empirically novel finding about how reasoning models internally allocate computation across failure types.

---

## Generalization Experiments

### Cross-Model Transfer
- Train XGBoost classifiers on failures from Qwen2.5-Coder-7B
- Test the same classifiers on failures from DeepSeek-R1-Distill-Qwen-7B (no retraining)
- **Question:** Do failure signatures generalize across model architectures, or is the failure-to-repair-cost relationship model-specific?

### Cross-Difficulty Transfer
- Train on failures from Easy and Medium LiveCodeBench problems
- Test on failures from Hard LiveCodeBench problems
- **Question:** Does the failure-to-repair-cost mapping remain stable as problem difficulty increases, or does it break down on harder problems?

---

## What Success Looks Like

The project is successful if:
- The adaptive router achieves within 2–3% of the Pass@1 of Always-Level-3
- The adaptive router consumes 30–50% fewer total tokens than Always-Level-3
- The Compute-Normalized Accuracy of the adaptive system exceeds all fixed-budget baselines
- The Recoverability classifier achieves AUROC above 0.75
- SHAP analysis reveals at least 2–3 features with clear, interpretable predictive relationships
- The DeepSeek-R1 thinking token analysis reveals a meaningful pattern between failure type and thinking budget

---

## What Failure Looks Like (and Why It Is Still Valuable)

Even if the predictions are poor, the project produces independent scientific value:
- The dataset (failure signatures mapped to repair costs) is novel and reusable by other researchers
- The recoverability analysis reveals what fraction of LLM coding failures are fundamentally unrecoverable — a finding with independent value regardless of classifier performance
- Null results inform the field about the limits of execution feedback as a predictive signal

---

## Known Limitations

| Limitation | How It Is Handled |
|---|---|
| Hard token cap may truncate reasoning mid-thought (especially DeepSeek-R1) | Tracked explicitly via `<think>` token parsing; reported as a finding |
| Fixed single repair attempt per budget level (no retries) | Acknowledged scope limitation; future work adds Best-of-N within budget |
| Two models may not generalize beyond Qwen-family architectures | Framed as preliminary generalization study, not a strong universal claim |
| LiveCodeBench is competitive programming — failure patterns may differ in real software engineering tasks | Acknowledged explicitly; SWE-bench generalization is future work |
| XGBoost may miss non-linear semantic interactions that embeddings capture | Embedding features are included; ablation study compares embedding vs. tabular-only |
| Benchmark distribution may shift over time | Benchmark version pinned via commit hash for full reproducibility |
| The full routing pipeline adds wall-clock latency overhead (sandbox execution + embedding forward pass + XGBoost inference) on top of generation time | Sandbox latency is measured and reported separately alongside token counts; the paper acknowledges that token savings must be weighed against execution overhead in latency-sensitive production deployments. In practice, XGBoost inference is microseconds and the embedding step is a single frozen forward pass — the dominant overhead is the sandbox execution itself, which is unavoidable in any execution-guided system. |

---

## Output Deliverables

At the end of the project, the following exist:

1. **A labeled dataset** — failure signatures from two LLMs on LiveCodeBench, each labeled with the minimum repair budget level that succeeded (or Unrecoverable)
2. **Two trained XGBoost classifiers** — the recoverability predictor and the budget router, with saved weights and hyperparameters
3. **SHAP analysis outputs** — feature importance plots, summary plots, and misprediction cluster analysis
4. **DeepSeek-R1 thinking token breakdown** — per-failure analysis of thinking vs. coding token allocation
5. **Generalization experiment results** — cross-model and cross-difficulty transfer performance tables
6. **A Pareto frontier visualization** — plotting Pass@1 vs. Total Tokens for the adaptive router, all fixed baselines, and the oracle, showing the efficiency frontier
7. **A research paper** describing the full experimental setup, results, findings, and limitations

---

## Summary in Plain Terms

Run two language models on hundreds of hard coding problems. Record every single failure in detail. For each failure, try to repair it at three increasing token budgets, finding the cheapest one that works. Use those labels to train two classifiers: one that predicts whether a failure is even fixable, and one that predicts how many tokens the fix needs. Deploy this system so it only spends what is needed per failure — cheap fixes get cheap budgets, expensive fixes get expensive budgets, and hopeless failures get nothing. Compare this to always spending the maximum budget, and show that the same accuracy is achievable at a fraction of the cost. Analyze what the classifiers learned to explain *why* certain failures are cheap or expensive to fix.
