# Learning Compute Allocation Policies from Execution Feedback
### A Practical Technical Plan

---

## What I Am Building — The One-Liner

A system that watches how a language model *fails* on a coding problem, extracts structured information from that failure, and predicts how much extra computation is needed to fix it — before spending that computation. Each failure is then routed to the cheapest repair budget likely to work.

---

## Why This Project Exists — The Problem

### Background: What is Test-Time Compute?

Modern LLMs can improve answers by spending more computation at inference time (Test-Time Compute, TTC): retrying, sampling multiple candidates, or iteratively debugging using their own error output. All of these cost more tokens, time, and money.

### The Inefficiency Being Solved

Current systems retry with a **fixed** token budget regardless of failure severity.

**Real Example A — trivial syntax error:**
```
SyntaxError: expected ':'
  File "<solution>", line 7
    if n > 0
            ^
```
This is a one-line fix. A 500-token repair budget is overkill by 4-8x.

**Real Example B — wrong algorithm under a time limit:**
```
TimeoutError: Execution exceeded 2000ms time limit
  Test case: n = 100000
```
An O(N²) solution against N=100,000 will never pass regardless of repair budget — no surface patch fixes an asymptotic complexity problem. A fixed 2000-token budget is fully wasted here.

These two failures look completely different at the moment they happen, and that difference is detectable *before* any repair tokens are spent. That's the core bet of this project.

---

## The Central Hypothesis (Chain Form)

1. **Different failure modes require different repair effort.** A `SyntaxError` is structurally different from a `TimeoutError`; a 1-frame traceback is different from an 8-frame one. These differences correlate with repair cost.
2. **The failure signature — exception type, traceback, runtime, code structure — contains enough signal to predict repair cost** without attempting the repair.
3. **A lightweight classifier trained on historical failure→repair-cost data can route new failures to the right budget**, beating any fixed-budget strategy on cost-per-correct-answer.

```
Execution Failure
      ↓
Failure Signature Extraction
      ↓
Recoverability Prediction  (is this fixable at all, under this budget regime?)
      ↓
Budget Routing  (if fixable, how many tokens?)
      ↓
Targeted Repair Attempt
```

---

## What This Project Is NOT

| This project is NOT about... | This project IS about... |
|---|---|
| Improving code generation quality | Deciding how much compute to spend *per failure* |
| Building a better coding agent | Resource allocation during inference |
| Training or fine-tuning a new LLM | Using existing models as fixed black boxes |
| Prompt engineering for better repairs | Studying failure signatures as a predictive signal |
| Self-debugging systems | Compute economics for inference |

LLMs are treated as fixed black boxes — never modified or fine-tuned, only run and observed.

---

## Practical Feasibility — Read This Before Building Anything

This section is the part most plans like this skip, and it's the part that determines whether the rest of the design even works.

### How much data will I actually have?

LiveCodeBench (any single pinned release) has on the order of a few hundred problems. Running two 7B models against it, with realistic failure rates of roughly 40-60% on Medium/Hard problems, gives you **roughly 150-500 labeled failures per model** — not thousands. This is the single most important number in the whole project, because it constrains every modeling choice downstream.

Consequences:
- **Raw 768-dim embeddings concatenated with ~20 tabular features, fed to anything (XGBoost or an MLP), will overfit at this sample size.** The embedding dimensionality is larger than the number of rows. This is not a tree-vs-neural-network problem — it's a curse-of-dimensionality problem that affects both equally.
- **Fix: reduce embedding dimensionality before fusion.** PCA or UMAP down to 16-32 components per embedding is the standard, practical approach for small-N + high-D fusion problems, and it's an afternoon of work, not a new model architecture.
- With dimensionality-reduced embeddings, **XGBoost remains the practical default**: it handles small tabular datasets well, gives you `TreeExplainer` SHAP for free, and doesn't need the extra infrastructure of a neural training loop + `DeepExplainer`.

### Is the XGBoost-vs-MLP question worth keeping at all?

Yes — but as a **secondary ablation**, not the headline architecture decision. Comparing "XGBoost on tabular + PCA-reduced embeddings" against "small MLP on tabular + raw embeddings" is a legitimate, citable experiment about whether axis-aligned splits lose information that a continuous fusion layer captures. But running that comparison should not block the main pipeline — build the XGBoost+PCA version first since it's faster to get working end-to-end, then add the MLP comparison if time permits.

### How many LLM calls does this actually require?

For each model (2 total):
- 1 generation per problem (baseline) — a few hundred calls
- For each failure, **up to 3 sequential repair attempts** (Level 1 → Level 2 → Level 3, stopping early on success) — realistically averages ~1.5-2 repair calls per failure, not always 3

Rough order of magnitude: **~400 problems × 2 models × (1 baseline + ~1 avg repair) ≈ 1,500-2,000 total generation calls.** This is feasible on a single modern GPU (24GB+ VRAM, e.g. a 4090 or A100) for local 7B models over a few days, or via API at a cost that should be estimated up front using current provider pricing before committing — token-heavy reasoning traces from DeepSeek-R1 will dominate the cost if using a paid API.

### Sandbox execution cost

Running hundreds to low-thousands of code submissions in an isolated sandbox (subprocess + timeout, or Docker) is cheap computationally but adds real wall-clock time due to process spin-up. Budget for this explicitly — it's usually the slowest part of the pipeline in practice, not the model inference.

---

## Dataset: LiveCodeBench

### Why Not HumanEval?

Modern 7B models solve 80%+ of HumanEval, leaving too few failures to build a training set, and contamination makes failure rates unrepresentative.

### Why LiveCodeBench?

- Problems are sourced from contests held after model training cutoffs — low memorization risk
- High enough failure rates on Medium/Hard problems to build a usable (if modest-sized) failure dataset
- Fully objective pass/fail via hidden test cases — no LLM judge needed
- Benchmark version pinned via commit hash for reproducibility

**Before finalizing the plan, run a quick pilot**: generate on ~50 problems with both models, measure the actual failure rate and exception-type distribution. This 30-minute pilot tells you whether your real dataset will be 150 failures or 500, and whether `AssertionError` really dominates (it usually does) — this directly determines whether dimensionality reduction is "nice to have" or "mandatory."

---

## Models Used

### Model 1 — Qwen2.5-Coder-7B
Standard code generation model, no chain-of-thought, generates code directly.

### Model 2 — DeepSeek-R1-Distill-Qwen-7B
Native reasoning model producing explicit `<think>...</think>` traces before code. Architecturally distinct from Qwen — tests whether the failure→repair-cost relationship transfers across architectures.

**Thinking-token caveat (real, observed behavior):** DeepSeek-R1-style models can spend their *entire* token budget inside `<think>` and emit no code at all if the cap is too low. A 500-token cap might be ~400 tokens of reasoning and ~100 tokens of code — often not enough for a complete function. This is tracked by parsing `<think>` tokens separately and is reported as a finding, not hidden.

---

## The Full System Pipeline

### Stage 1: Baseline Code Generation
- Input: every LiveCodeBench problem (description + signature + hidden tests)
- Models: Qwen2.5-Coder-7B and DeepSeek-R1-Distill-Qwen-7B, run independently
- Decoding: temperature = 0, greedy (fully reproducible)
- Recorded: generated code, total token count, latency (ms), lines of code, `<think>` trace presence/length

### Stage 2: Execution and Evaluation
Every generated function runs against LiveCodeBench's hidden tests in an isolated sandbox.

- **PASS** → done, no further processing
- **FAIL** → full failure info captured for Stage 3:
  - Exception type (`SyntaxError`, `AssertionError`, `IndexError`, `ValueError`, `TypeError`, `TimeoutError`, `RecursionError`, `NameError`, `AttributeError`, etc.)
  - Full traceback text
  - Traceback depth (stack frame count)
  - Runtime in ms
  - Failure line number
  - The specific test case that triggered failure

No LLM judge anywhere — correctness is purely pass/fail against hidden tests.

### Stage 3: Failure Signature Extraction

Every failure becomes a structured feature vector. Only information available *at the moment of failure* is used — no future information, no repair attempted yet.

**Execution features:**
- Exception type (one-hot across observed categories)
- Traceback depth
- Runtime (ms)
- Failure line number (normalized by total lines)
- Timeout binary flag

**Code structure features (static analysis):**
- Total token count, lines of code
- AST node count (complexity proxy)
- Cyclomatic complexity
- Number of function definitions
- Max nesting depth
- Recursion binary flag

**Problem features:**
- Prompt length in tokens
- Difficulty (Easy/Medium/Hard, ordinal)

**Embedding features (dimensionality-reduced):**
- Embed the full traceback text and the failed code using a frozen embedding model (CodeBERT or StarEncoder)
- **Apply PCA (or UMAP) to reduce each embedding from 768 dims to ~16-32 dims** before concatenation — this is the step that makes fusion with a few hundred training rows viable
- Concatenate reduced embeddings with tabular features into the final feature vector

This combined, dimensionality-reduced vector is what feeds the classifiers in Stages 5 and 6.

### Stage 4: Ground-Truth Label Discovery

Every failure runs through a **repair ladder** with increasing token budgets, using a **frozen, identical repair prompt** at every level to isolate the effect of compute from the effect of prompting:

```
Original Problem: [problem description]
Failed Code: [the full generated solution that failed]
Execution Traceback: [the complete error output]
Instruction: Fix the code and output only the corrected Python function.
```

**Repair Budget Ladder:**

| Level | Token Cap | Typical Failure Type |
|---|---|---|
| Level 0 | 0 | No repair — baseline |
| Level 1 | 500 | Syntax errors, off-by-one, missing return, wrong variable name |
| Level 2 | 1500 | Wrong conditional, missing edge case, wrong data structure |
| Level 3 | 3000 | Wrong algorithm, full restructuring needed |

**Sequential labeling:** try Level 1 → if pass, label = L1. Else try Level 2 → if pass, label = L2. Else try Level 3 → if pass, label = L3. Else label = **Unrecoverable**.

**On "Unrecoverable":** this means *unrecoverable via zero-shot execution feedback within these specific budget caps and this specific prompt* — not mathematically unsolvable. A different prompt, larger model, or agentic loop might recover it. This scoping is stated everywhere the term is used.

**Output:** `[failure signature] → [L1 / L2 / L3 / Unrecoverable]` — the dataset Stages 5 and 6 train on. Given the realistic dataset size (~150-500 rows per model), expect class imbalance toward L1 and Unrecoverable, with L2/L3 sparsely populated — this should be reported explicitly and may require class-weighting or merging L2/L3 if too sparse.

### Stage 5: Recoverability Prediction (First Gate)

- Binary classification: RECOVERABLE vs UNRECOVERABLE
- Input: failure signature (tabular + PCA-reduced embeddings)
- Model: **XGBoost** (default, given dataset size)
- If predicted UNRECOVERABLE → stop immediately, zero repair tokens spent

**Metrics:** AUROC (target > 0.75, but with ~150-500 examples report confidence intervals — a point estimate alone is misleading at this N), precision/recall on UNRECOVERABLE, Brier score for calibration.

### Stage 6: Repair Budget Routing (Second Gate)

- Ordinal classification: L1 < L2 < L3
- Input: failure signature of failures predicted RECOVERABLE
- Model: **XGBoost** with ordinal-aware loss (or, given small-N, an ordinal logistic regression baseline alongside XGBoost — worth comparing both)
- Why ordinal: predicting L3 when the truth is L1 wastes 2500 tokens; predicting L2 when truth is L1 wastes 1000. A standard multi-class loss treats both errors as equally wrong, which is incorrect here.

**Metric:** Mean Absolute Level Error (primary), per-class precision/recall, confusion matrix.

---

## Architecture Ablation: XGBoost vs MLP Fusion

Given the dimensionality-reduction step above, the headline pipeline stays XGBoost end-to-end (simpler, interpretable, appropriate for the dataset size). As a **secondary experiment**, add:

- **Variant B:** small 2-layer MLP (e.g. `Linear(tabular_dim + 64, 64) → ReLU → Dropout → Linear(64, num_classes)`) trained on tabular + raw (non-PCA) embeddings, interpreted with `shap.DeepExplainer`

Compare Variant B against the XGBoost+PCA pipeline on the same train/test split. This produces a real, reportable finding either way:
- If XGBoost+PCA matches or beats the MLP → dimensionality reduction was sufficient, simpler pipeline wins
- If the MLP meaningfully beats XGBoost+PCA → continuous fusion captures something PCA discards, and that's worth a deeper dive

Either outcome is publishable; neither requires committing to the MLP as the primary system before you've seen the data.

---

## Baselines

| Baseline | What It Does | Why It's Included |
|---|---|---|
| No Repair | Generate once, stop | Lower bound on cost and accuracy |
| Always Level 1 | 500 tokens on every failure | Tests if cheap fixes are usually enough |
| Always Level 3 | 3000 tokens on every failure | Upper bound on cost/accuracy for fixed strategies |
| Random Routing | Randomly assign L1/L2/L3 | Tests if the predictor beats chance |
| Oracle Routing | Always use the true minimum level (Stage 4 labels) | Theoretical ceiling |
| Best-of-N | N parallel 500-token attempts, take first pass | Tests adaptive routing vs brute-force sampling |

Best-of-N is the most important comparison: if 3 parallel cheap attempts beat the adaptive router, the core claim that *execution feedback enables better single-pass routing than brute-force sampling* is weakened — and that's worth knowing.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| Pass@1 | % of problems solved (generation + repair) |
| Recovery Rate | % of failures successfully repaired |
| Total Token Cost | Sum of all tokens across all problems |
| Compute-Normalized Accuracy (CNA) | Pass@1 ÷ Total Tokens |
| Inference Latency | Wall-clock generation time |
| Sandbox Execution Overhead | Wall-clock sandbox time per failure (reported separately) |
| Recoverability AUROC | Stage 5 classifier performance |
| Budget Prediction MAE | Stage 6 ordinal error |
| Brier Score | Probability calibration |

**Primary claim being tested:** the adaptive system reaches close to Always-Level-3's Pass@1 while using meaningfully fewer total tokens. Given the small dataset, report this with confidence intervals (e.g. bootstrap over the test set) rather than a single point estimate — at N~100-150 test failures, a "30-50% savings" claim needs an error bar to be credible.

---

## Interpretability: SHAP Analysis

Applied to the XGBoost classifiers via `TreeExplainer` (and to the MLP variant via `DeepExplainer`, if built).

**Things to actually check (framed as questions, not pre-confirmed findings):**
- Does `SyntaxError` cluster toward L1?
- Does `TimeoutError` cluster toward Unrecoverable/L3?
- Do AST/cyclomatic complexity correlate with higher repair levels?
- Does runtime-near-timeout predict algorithmic (expensive) failures?

These are reasonable hypotheses based on how these exceptions typically behave, but with a few hundred examples some of these may not reach statistical significance — report which ones do and which don't, rather than treating all of them as confirmed.

**AssertionError ablation:** `AssertionError` (wrong output, right execution) is typically the largest and most ambiguous class — it covers everything from a one-character comparison bug to a wrong DP recurrence. Compare the XGBoost+PCA-embeddings model against a tabular-only model **specifically on AssertionError cases** to see how much the embeddings actually help when the exception label itself is uninformative.

---

## DeepSeek-R1 Thinking Token Analysis

For DeepSeek-R1 outputs:
1. Parse `<think>...</think>` separately from code tokens
2. Record thinking tokens, coding tokens, total tokens per failure
3. Train a small secondary regressor: failure signature → thinking budget needed vs coding budget needed
4. SHAP on this regressor to see which failure types drive heavy thinking vs light thinking

**Reasonable expectation, to be checked:** algorithmically complex failures (TimeoutError, large-input wrong-output) lean on thinking tokens more; trivial failures (SyntaxError, NameError) need minimal thinking. Worth checking whether this holds, and reporting honestly if it doesn't — a null result here is still informative about how these models allocate reasoning.

---

## Generalization Experiments

### Cross-Model Transfer
Train on Qwen failures, test (no retraining) on DeepSeek failures. Question: does the failure→repair-cost mapping transfer across architectures, or is it model-specific? With ~150-500 examples per model, treat this as a directional/preliminary result, not a strong claim.

### Cross-Difficulty Transfer
Train on Easy/Medium, test on Hard. Question: does the mapping hold as difficulty increases?

---

## What Success Looks Like

- Adaptive router gets within a few percentage points of Always-Level-3's Pass@1 (with confidence interval)
- Adaptive router uses meaningfully fewer total tokens than Always-Level-3
- CNA of the adaptive system beats fixed-budget baselines
- Recoverability AUROC clears 0.75 (with CI reported given small N)
- At least a couple of SHAP relationships hold up and are interpretable
- DeepSeek thinking-token analysis shows *some* pattern — even a weak or null one is reportable

---

## What Failure Looks Like (and Why It's Still Valuable)

Even with poor predictive performance:
- The labeled failure→repair-cost dataset is novel and reusable
- The recoverability breakdown tells you what fraction of LLM coding failures are fundamentally unrecoverable under zero-shot feedback — independently interesting
- A null result on the embedding ablation tells the field something about the limits of execution feedback as a signal

---

## Known Limitations

| Limitation | How It's Handled |
|---|---|
| Small dataset (~150-500 failures per model) | Dimensionality reduction on embeddings; confidence intervals on all headline metrics; class-imbalance handling for L2/L3 |
| Hard token caps may truncate DeepSeek-R1 mid-reasoning | Tracked via `<think>` token parsing; reported as a finding |
| Single repair attempt per budget level (no retries) | Stated scope limitation; Best-of-N is a separate baseline, not combined with routing |
| Two models may not generalize beyond Qwen-family architectures | Framed as preliminary, not a universal claim |
| LiveCodeBench is competitive programming, not general SWE | Acknowledged; SWE-bench generalization is future work |
| Raw embeddings + small N would overfit either XGBoost or an MLP | PCA/UMAP reduction applied before fusion; ablation compares reduced-XGBoost vs raw-MLP |
| Sandbox execution adds wall-clock overhead | Measured and reported separately from token costs |
| Benchmark distribution may shift over time | Version pinned via commit hash |

---

## Output Deliverables

1. A labeled dataset: failure signatures from two LLMs on a pinned LiveCodeBench version, each with a minimum-repair-level label (or Unrecoverable)
2. Two trained XGBoost classifiers (recoverability, budget routing) with saved weights/hyperparameters, plus the MLP comparison variant if built
3. SHAP outputs: feature importance plots, AssertionError ablation, misprediction cluster analysis
4. DeepSeek-R1 thinking-vs-coding token breakdown
5. Cross-model and cross-difficulty transfer results
6. Pareto frontier plot: Pass@1 vs Total Tokens for adaptive router, fixed baselines, and oracle
7. A writeup describing setup, results (with confidence intervals), and limitations — including an honest pilot-study section reporting the actual failure counts and class distribution observed

---

## Plain-Language Summary

Run two language models on a pinned set of hard coding problems. Record every failure in detail. For each failure, try repairing it at three increasing token budgets and find the cheapest one that works. Reduce the dimensionality of code/traceback embeddings so they can be combined with simple tabular features without overfitting a small dataset. Train two small classifiers — one predicting whether a failure is fixable at all, one predicting how many tokens a fix needs — primarily with XGBoost, with a small MLP comparison as a secondary check. Deploy this so the system spends only what's needed per failure. Compare against always spending the maximum, report results with honest uncertainty given the dataset size, and use SHAP to explain what the classifiers learned.
