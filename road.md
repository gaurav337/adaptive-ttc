# Learning Roadmap: Final Edition (Updated)
## Learning Compute Allocation Policies from Execution Feedback

Organized by the **order you build things**, not by topic.
Each phase tells you what to learn, what to build, exactly where to learn it, and the engineering traps that will cost you weeks if you don't know about them.

This version is aligned with the updated project README: the dataset will likely be **small** (a few hundred failures per model), so dimensionality reduction and honest uncertainty reporting are built into the plan from the start, not bolted on at the end.

---

## Before You Start: Foundations Check

You need these before touching the project. If you have them, skip.

**Python (must be solid)**
- File I/O, subprocess, error handling, classes, decorators
- If shaky: https://cs50.harvard.edu/python/2022/ (free, self-paced)

**Basic ML (must know what a classifier is)**
- What train/test split means, what a feature vector is, what overfitting means
- If shaky: StatQuest's "Machine Learning Fundamentals" playlist — https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF (short, focused videos, no fluff)

**Git (must be comfortable)**
- You will be running long experiments. Version everything from day one.
- Before every experiment run, commit with a message like `run: baseline qwen2.5`.
- When you get a result, you can always trace it back to the exact code that produced it.

---

## Phase 1: Understand What You Are Actually Building
**Duration: 1 week | Build nothing yet, only read**

The goal is to make sure you are not building blindly. Read these in order.

### Papers to read (in this exact order)

**1. The core motivation paper**
- "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"
- Snell et al., 2024 — https://arxiv.org/abs/2408.03314
- Read sections 1, 2, and the conclusion. Skip the PRM math for now.
- What to understand: why more inference compute helps, and why it should be adaptive

**2. The benchmark you are using**
- "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"
- https://arxiv.org/abs/2403.07974
- Read sections 1–3 only.
- What to understand: how the benchmark works, what pass@1 means, why contamination matters, and roughly how many problems are in the version you'll pin — this number directly determines your dataset size (see Phase 3 pilot step)

**3. One related work paper to situate yourself**
- "Is Self-Repair a Silver Bullet for Code Generation?" (Olausson et al.)
- https://arxiv.org/abs/2306.09896
- Read sections 1 and 5.
- What to understand: that self-repair does not always work, and budget matters

### After reading, write one paragraph answering:
"What is the specific gap in existing work that my project addresses?"
If you cannot write it fluently, re-read Phase 1.

---

## Phase 2: Learn How LLMs Generate Code (Just Enough)
**Duration: 1 week | Build: run your first model locally**

You do not need to understand transformers deeply. You need to understand token generation, temperature, and how to call a model.

### Watch

**Andrej Karpathy — Intro to Large Language Models (1 hour)**
- https://www.youtube.com/watch?v=zjkBMFhNj_g
- Watch the full thing. Best single-hour explanation of what LLMs are.

*("Let's build GPT" is genuinely excellent but 2+ hours and not needed to ship this project — skip unless you have spare time and want the deeper intuition.)*

### Read

**Hugging Face — How to generate text**
- https://huggingface.co/blog/how-to-generate
- Understand: greedy decoding vs sampling, temperature, max_new_tokens
- These are the exact settings you will use in Stage 1

### Build

Set up local inference with **vLLM**:
- Docs: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
- For Phase 2, use the OpenAI-compatible server mode to test interactively

**Mini-project:** Write a Python script that:
1. Takes a coding problem as a string
2. Sends it to Qwen2.5-Coder-7B with `temperature=0`, `max_new_tokens=512`
3. Returns the generated code and prints the token count

This is literally Stage 1 of your pipeline.

> **⚠️ CRITICAL — Offline Batching vs. Server Mode**
>
> Server mode is fine for testing. But from Phase 5 onward you'll be running hundreds to low-thousands of generations (problems × models × repair levels). Sequential API calls will take **days**.
>
> **The fix:** Learn **vLLM Offline Batch Inference** now, before you need it.
> ```python
> from vllm import LLM, SamplingParams
> llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct")
> params = SamplingParams(temperature=0, max_tokens=500)
> outputs = llm.generate(list_of_prompts, params)
> ```
> vLLM's PagedAttention and continuous batching processes all of them together — **minutes, not days.**
> Docs: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference

### DeepSeek-R1 specific setup

DeepSeek-R1-Distill-Qwen-7B generates `<think>...</think>` blocks before code. From day one, write a parser that separates the two:

```python
import re

def parse_deepseek_output(raw_output: str) -> dict:
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    think_text = think_match.group(1) if think_match else ""
    code_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    return {
        "think_text": think_text,
        "code_text": code_text,
        "think_tokens": len(think_text.split()),   # approximate; use tokenizer for exact count
        "code_tokens": len(code_text.split()),
    }
```

Store `think_tokens` and `code_tokens` separately for every DeepSeek generation from the very first run. You will need this for the Phase 8 thinking token analysis, and retrofitting it later means re-running all generations.

---

## Phase 3: Set Up LiveCodeBench, Run a Pilot, Then Run Baselines
**Duration: 1–2 weeks | Build: a small pilot run, then Stage 1 and Stage 2 at full scale**

### Set up the benchmark

**Official repo:** https://github.com/LiveCodeBench/LiveCodeBench

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
pip install poetry
poetry install
python -m lcb_runner.runner.main --model qwen2.5-coder-7b-instruct --scenario codegeneration
```

This gives you pass@1 scores out of the box. What you need to do next: study `lcb_runner/runner/main.py` and `lcb_runner/evaluation/` to understand where execution happens. You will hook your failure capture logic there.

### ⚠️ NEW — Run a pilot before committing to the full pipeline

Before building anything in Phases 4–8, run both models on **~50 problems** and record the pass/fail outcomes and exception-type distribution. This takes under a day and tells you:

- Your actual failure rate (commonly 40–60% on Medium/Hard problems, but varies by model and benchmark version)
- Roughly how many total failures you'll have once you scale to the full benchmark — this is the number that determines whether your full-dimensional embedding features will overfit (see Phase 4)
- Whether `AssertionError` really dominates the failure distribution (it usually does)

Write these numbers down. They go directly into your methodology section, and they decide whether the "PCA-reduced features as primary" path in Phase 4 is optional or mandatory.

### Learn Python's `ast` module and code execution

- Python `subprocess` docs: https://docs.python.org/3/library/subprocess.html
- Real Python guide on AST: https://realpython.com/python-ast/ (covers the `ast` module well in one article — no need for the full official docs unless you need a specific node type reference)

> **⚠️ CRITICAL — Sandboxing: subprocess alone will crash your machine**
>
> LLMs frequently generate dangerous code: `while True: os.fork()` (fork bombs), `import shutil; shutil.rmtree('/')`, infinite memory allocations. `subprocess` with a timeout often **fails to kill child processes**, which will crash your host machine, corrupt your environment, or fill your disk.
>
> **You have two options:**
>
> **Option A (recommended for speed): E2B sandboxes**
> E2B is an API that spins up isolated microVMs for code execution. Free tier is sufficient for research.
> Docs: https://e2b.dev/docs
> ```python
> from e2b_code_interpreter import Sandbox
> with Sandbox() as sbx:
>     execution = sbx.run_code(generated_code)
>     print(execution.error)  # None if passed
> ```
>
> **Option B (recommended for full control): Docker with hard limits**
> ```bash
> docker run --rm --memory="512m" --cpus="1" --network none \
>   --read-only python:3.11 python -c "{generated_code}"
> ```
> The `--network none` flag prevents any external network calls. `--memory` and `--cpus` caps prevent resource bombs.
>
> **Do not use raw subprocess for untrusted code execution.** This is not a minor detail — it is a machine-safety issue.

**Mini-project:** Write `safe_execute(code_string, timeout=5)` using either E2B or Docker that returns:
```python
{"status": "pass"}
# or
{"status": "fail", "exception_type": "TimeoutError", "traceback": "...", "runtime_ms": 4821}
```

---

## Phase 4: Build the Failure Signature Extractor
**Duration: 1–2 weeks | Build: Stage 3 — tabular + semantic embedding features, with dimensionality reduction built in**

### Part A — Tabular Features

**Radon — cyclomatic complexity and LOC:**
```bash
pip install radon
```
- Docs: https://radon.readthedocs.io/en/latest/intro.html
- You want `radon.complexity.cc_visit()` for cyclomatic complexity
- And `radon.raw.analyze()` for lines of code, logical lines, blank lines

**Python `ast` module — AST node count:**
```python
import ast
tree = ast.parse(code)
ast_node_count = sum(1 for _ in ast.walk(tree))
```

**Error message keyword extraction:**
```python
import re
last_line = traceback_str.strip().split('\n')[-1]
keyword = last_line.split(':')[0].strip().lower().replace(' ', '_')
# "keyerror", "indentationerror", "assertionerror", etc.
```
One-hot or label-encode the top 10–15 most common keywords observed in your **pilot** (Phase 3). With a small dataset, don't pre-allocate categories for exceptions you haven't actually seen — it just adds empty columns.

Your tabular feature dict should look like:
```python
{
  "exception_type_syntaxerror": 1,
  "exception_type_timeouterror": 0,
  "exception_type_assertionerror": 0,
  "exception_type_valueerror": 0,
  "exception_type_recursionerror": 0,
  "exception_type_other": 0,
  "traceback_depth": 3,
  "runtime_ms": 4821,
  "failure_line_normalized": 0.5,
  "is_timeout": 0,
  "token_count": 187,
  "lines_of_code": 24,
  "cyclomatic_complexity": 7,
  "ast_node_count": 143,
  "has_recursion": 0,
  "max_nesting_depth": 3,
  "prompt_length_tokens": 312,
  "difficulty_level": 2,
  "error_msg_keyword": "assertionerror",
}
```

### Part B — Semantic Embedding Features (required)

Two `AssertionError` failures can look identical in tabular features but require completely different repair costs. Embeddings let the classifier read the actual content of the error.

**What to learn first:**

**3Blue1Brown — Neural Networks, Chapters 1–3 only**
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Just enough to understand what an embedding vector is and why similar inputs produce similar vectors. You do not need backprop for this project.

**Hugging Face — Feature Extraction pipeline**
- https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FeatureExtractionPipeline
- 15 minutes. Shows you how to get a dense vector from any text.

**Model to use — microsoft/codebert-base:**
- https://huggingface.co/microsoft/codebert-base
- 125M parameters, runs on any GPU with 4GB VRAM, produces 768-dimensional vectors
- Pre-trained on code and natural language — suited to code + traceback text

**How to embed tracebacks and code:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()  # freeze — never train this model

def embed_text(text: str, max_length: int = 512) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding as the sentence vector
    cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return cls_vector  # 768-dimensional list of floats

traceback_embedding = embed_text(traceback_str)   # 768 floats
code_embedding = embed_text(failed_code)           # 768 floats
```

**Batching for speed:**
```python
def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vecs = outputs.last_hidden_state[:, 0, :].tolist()
        all_embeddings.extend(cls_vecs)
    return all_embeddings
```

Embed all tracebacks and all failed code snippets in batches before training. Save the vectors to disk (numpy `.npy` or HDF5). Never re-embed the same text twice.

### ⚠️ NEW — Dimensionality reduction is part of the main pipeline, not just a SHAP step

With ~20 tabular features and two 768-dim embeddings, your raw combined vector is ~1556 dimensions. If your pilot (Phase 3) suggests your full dataset will be in the low hundreds of failures, **1556 features for a few hundred rows will overfit** — regardless of which classifier you use downstream.

```python
from sklearn.decomposition import PCA

pca_traceback = PCA(n_components=32)
pca_code = PCA(n_components=32)

traceback_emb_reduced = pca_traceback.fit_transform(all_traceback_embeddings)  # fit on TRAIN only
code_emb_reduced = pca_code.fit_transform(all_code_embeddings)                  # fit on TRAIN only
```

**Build two feature sets and carry both forward:**
- **Reduced (primary):** ~20 tabular + 32 + 32 PCA components ≈ 84 features — this is what Stage 5/6 train on by default
- **Full (secondary check):** ~20 tabular + 768 + 768 ≈ 1556 features — train once for comparison; if it performs similarly or worse than the reduced version (likely, given the row count), that itself is worth reporting

Fit PCA on the training fold only and apply the transform to the test fold — fitting PCA on the full dataset before splitting leaks test information into your features.

> **⚠️ NOTE — Multicollinearity**
>
> `cyclomatic_complexity` and `ast_node_count` will be correlated. Compute both and include both — they measure different structural properties (branching paths vs. total node count). Note this in your methodology and check if both appear in SHAP or one dominates.

---

## Phase 5: Build the Repair Ladder
**Duration: 2 weeks | Build: Stage 4 — creates your entire dataset**

This is the most time-consuming phase. It produces the training data for everything else.

### The loop you are building

```python
TOKEN_CAPS = {1: 500, 2: 1500, 3: 3000}

REPAIR_PROMPT = """Problem: {problem}
Failed Code: {failed_code}
Error: {traceback}
Fix the code. Output only the corrected Python function."""
# This prompt is FROZEN. Never change it across any experiment.

for problem in failed_problems:
    label = "unrecoverable"
    for level in [1, 2, 3]:
        repaired_code = call_model(
            prompt=REPAIR_PROMPT.format(...),
            max_new_tokens=TOKEN_CAPS[level]  # exact cap, not approximate
        )
        result = safe_execute(repaired_code)
        if result["status"] == "pass":
            label = level
            break

    save_record(problem_id, failure_signature, label)
```

### What "Unrecoverable" means — be precise from the start

Label a failure "Unrecoverable" only when Level 3 repair (3000 tokens) fails. In your code comments, dataset documentation, and paper, always define this as:

> *"Unrecoverable via zero-shot execution feedback within the tested budget constraints using the fixed repair prompt."*

Never write "the problem is unsolvable." A different prompt, a larger model, or an agentic loop might recover it. The label is scoped to your specific experimental setup. Write this definition as a docstring in the function that assigns labels.

### Critical things to get right

**1. Frozen prompt.** Use the exact template above for every repair attempt, every level, both models. The token budget is your only independent variable.

**2. Exact token caps enforced in code.** Use `max_new_tokens=500` not `max_new_tokens="around 500"`. The label depends on which cap the fix appeared under.

**3. Use offline batch inference (from Phase 2).** Run all Level 1 repairs for all problems as one batch. Then all Level 2 for the Level 1 failures. Then Level 3. Do not loop with server calls.

**4. Run both models.** Qwen2.5-Coder first. Then DeepSeek-R1-Distill-Qwen. Store `"model"` in every record. For DeepSeek, store `think_tokens` and `code_tokens` separately (use the parser from Phase 2).

> **⚠️ CRITICAL — Checkpointing: this experiment WILL crash**
>
> Even at the realistic smaller scale, you're running hundreds of failures × up to 3 repair attempts × sandbox executions. This will take hours, and CUDA OOM errors, Docker timeouts, or network blips will crash it partway through.
>
> **The fix:** Write a checkpoint system from the start:
> ```python
> import jsonlines, os
>
> CHECKPOINT_FILE = "repair_state.jsonl"
>
> completed = set()
> if os.path.exists(CHECKPOINT_FILE):
>     with jsonlines.open(CHECKPOINT_FILE) as reader:
>         for record in reader:
>             completed.add((record["problem_id"], record["model"]))
>
> for problem in all_problems:
>     if (problem.id, current_model) in completed:
>         continue
>
>     # ... run repair ladder ...
>
>     with jsonlines.open(CHECKPOINT_FILE, mode='a') as writer:
>         writer.write(record)  # write immediately, not at the end
> ```
> When the script crashes partway through, restart it. It reads the checkpoint, skips completed records, and resumes.

### Output format

Each record saved to `repair_state.jsonl`:
```json
{
  "problem_id": "lcb_001",
  "model": "qwen2.5-coder-7b",
  "failure_signature": {"exception_type_syntaxerror": 1, "traceback_depth": 2, "...": "..."},
  "traceback_embedding_path": "embeddings/lcb_001_qwen_traceback.npy",
  "code_embedding_path": "embeddings/lcb_001_qwen_code.npy",
  "label": 1,
  "repair_tokens_used": 312,
  "think_tokens": null,
  "code_tokens": null,
  "git_commit": "a3f9c21"
}
```

For DeepSeek records, `think_tokens` and `code_tokens` are integers. For Qwen records, they are null. Always record the git commit hash.

---

## Phase 6: Train the Classifiers (XGBoost primary, MLP as optional ablation)
**Duration: 1 week | Build: Stage 5 and Stage 6**

### Watch (in this order)

**StatQuest — XGBoost series (4 parts, ~1 hour total)**
- Part 1 (decision trees): https://www.youtube.com/watch?v=OtD8wVaFm6E
- Part 2 (XGBoost math): https://www.youtube.com/watch?v=8b1JEDvenQU
- Part 3 (classification): https://www.youtube.com/watch?v=ZVFeW798-2I
- Part 4 (optimizations): https://www.youtube.com/watch?v=oRrKeUCEbq8

**StatQuest — ROC and AUC explained**
- https://www.youtube.com/watch?v=4jRBRDbJemM
- Required to understand your AUROC metric for Stage 5

### Read

- XGBoost docs: https://xgboost.readthedocs.io/en/latest/get_started.html

> **⚠️ CRITICAL — Data Leakage via problem_id**
>
> You run both Qwen and DeepSeek on the same LiveCodeBench problems. If you use a random `train_test_split`, Problem #42 (Qwen attempt) might land in the training set while Problem #42 (DeepSeek attempt) lands in the test set. Your model will "cheat" — it will learn the difficulty of the problem itself rather than the failure signature.
>
> **The fix:** Use `StratifiedGroupKFold` and group by `problem_id`:
> ```python
> from sklearn.model_selection import StratifiedGroupKFold
> import numpy as np
>
> groups = np.array([record["problem_id"] for record in dataset])
> sgkf = StratifiedGroupKFold(n_splits=5)
>
> for train_idx, test_idx in sgkf.split(X, y, groups):
>     X_train, X_test = X[train_idx], X[test_idx]
>     y_train, y_test = y[train_idx], y[test_idx]
> ```
> This guarantees all attempts for Problem #42 stay in the same fold — train or test, never both. Without this, your reported metrics are inflated and scientifically invalid.

> **⚠️ With a small dataset, 5-fold splits may leave very few examples per fold.** If `StratifiedGroupKFold(n_splits=5)` produces folds too small to be stable (check class counts per fold), drop to 3 folds, or report leave-one-group-out results alongside the k-fold average.

> **⚠️ UPGRADE — Handle class imbalance**
>
> Your dataset will likely be imbalanced toward L1 and Unrecoverable, with L2/L3 sparse. XGBoost will learn to ignore rare classes.
> ```python
> from sklearn.utils.class_weight import compute_class_weight
> weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
> # Pass to XGBoost via sample_weight parameter in .fit()
> ```
> If L2 and L3 each have fewer than ~15-20 examples after splitting, consider merging them into a single "Level 2+" class and reporting this explicitly — a 4-class problem with two near-empty classes will not produce a meaningful confusion matrix.

### Run the ablation study — this is required, not optional

Train these versions of each classifier on the **PCA-reduced** feature set from Phase 4, and report all of them:

| Model version | Features used | Purpose |
|---|---|---|
| Tabular only | ~20 handcrafted features | Baseline — what a non-semantic approach achieves |
| Traceback embedding only | 32-dim PCA-reduced traceback vector | How much does reading the error text help? |
| Code embedding only | 32-dim PCA-reduced code vector | How much does reading the code help? |
| Combined (tabular + both reduced embeddings) | ~84 features | Main model |

The gap between "tabular only" and "combined" on AssertionError cases specifically is your key finding (see Phase 7). Filter your test set to only AssertionError failures and report AUROC for each model version on that subset.

**Optional secondary ablation:** train a small 2-layer MLP (`Linear(tabular_dim + 64, 64) → ReLU → Dropout → Linear(64, num_classes)`) on tabular + *raw* (non-PCA) embeddings, interpreted with `shap.DeepExplainer`. This tests whether a continuous fusion layer captures something PCA discards. Only do this once the XGBoost pipeline above is working end-to-end — it's a bonus comparison, not a blocker.

### Build Stage 5: Recoverability Classifier

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

X = combined_feature_matrix   # tabular + PCA-reduced embeddings
y = binary_labels              # 0 = unrecoverable, 1 = recoverable
groups = problem_id_array

sgkf = StratifiedGroupKFold(n_splits=5)
auroc_scores = []

for train_idx, test_idx in sgkf.split(X, y, groups):
    model = xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight)
    model.fit(X[train_idx], y[train_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    auroc_scores.append(roc_auc_score(y[test_idx], probs))

print("Mean AUROC:", np.mean(auroc_scores), "± std:", np.std(auroc_scores))
```

Report the standard deviation across folds alongside the mean — with a small dataset, fold-to-fold variance is itself informative.

### Build Stage 6: Budget Router

For recoverable failures only (labels 1, 2, 3):

**Primary approach — XGBoost regression rounded to the nearest level.** This avoids adding a separate ordinal-regression dependency (the commonly-cited `mord` package is small and has not been actively maintained, which can cause friction with current numpy/sklearn versions) and is simple to implement and explain:

```python
import xgboost as xgb
import numpy as np

router = xgb.XGBRegressor(objective="reg:squarederror")
router.fit(X_train, y_train)  # y_train: integer levels 1, 2, 3

raw_preds = router.predict(X_test)
preds = np.clip(np.round(raw_preds), 1, 3)
mae = np.mean(np.abs(preds - y_test))
print("Mean Absolute Level Error:", mae)
```

**Optional comparison — proper ordinal model.** If you want a model that explicitly respects the ordinal structure, `statsmodels`' `OrderedModel` is actively maintained and well-documented:
- Docs: https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html
```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

model = OrderedModel(y_train, X_train, distr="logit")
result = model.fit(method="bfgs")
```
Compare its MAE against the XGBoost-regression-rounded approach; report whichever is clearer, or both if they differ meaningfully.

---

## Phase 7: Learn SHAP and Run Interpretability Analysis
**Duration: 3–4 days | Build: the scientific contribution**

### Read

- SHAP official XGBoost notebook: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html
- Gentle intro to SHAP for tree models: https://machinelearningmastery.com/a-gentle-introduction-to-shap-for-tree-based-models/

### Build

```python
import shap

explainer = shap.TreeExplainer(recoverability_model)
shap_values = explainer(X_test)  # already PCA-reduced from Phase 4, so this is readable directly

shap.plots.beeswarm(shap_values)          # global feature importance
shap.plots.waterfall(shap_values[0])      # single prediction explanation
```

Because Phase 4 already reduced the embeddings to ~32 dimensions each before training, you don't need a separate "PCA for interpretability only" step — the model you trained and the model you're explaining are the same one.

> **⚠️ UPGRADE — Use SHAP Dependence Plots for your paper figures**
>
> ```python
> shap.plots.scatter(shap_values[:, "cyclomatic_complexity"])
> shap.plots.scatter(shap_values[:, "exception_type_timeouterror"])
> ```
> Each plot shows one feature on the X-axis and its SHAP contribution on the Y-axis. A clear trend supports your hypothesis visually — these become your best paper figures, if the trend is actually there. With a small dataset, also check how many points are driving the trend; a pattern from 5 points isn't the same as one from 50.

### What to check with SHAP (write these as numbered observations in your paper — report what you find, not what you expected)

1. What is the single most important feature for recoverability prediction?
2. Does `exception_type_timeouterror` push predictions toward Unrecoverable?
3. Does `exception_type_syntaxerror` push predictions toward Level 1?
4. Does high `cyclomatic_complexity` correlate with Level 3?
5. Does `runtime_ms` close to the timeout limit predict unrecoverability?
6. Are the top 3 features the same for both Qwen failures and DeepSeek failures?
7. **For AssertionError failures specifically:** does the combined model outperform tabular-only, and which PCA components of the embeddings drive that difference?

### AssertionError misprediction analysis — write this as a paper section

Filter your test set to only AssertionError failures. For this subset:
- Compare tabular-only AUROC vs. combined AUROC
- List the failure signatures of every case where tabular-only predicted RECOVERABLE but it was UNRECOVERABLE (the hard cases the tabular model misses)
- Check if the combined model correctly classifies those same cases

Write 3–5 observations. This turns a classification result into a finding about what information actually determines repair cost when the exception type is ambiguous. If your AssertionError subset is very small (check this against your Phase 3 pilot numbers), say so explicitly rather than drawing conclusions from a handful of cases.

**Multicollinearity check:** Before SHAP, plot a correlation matrix. If `cyclomatic_complexity` and `ast_node_count` have r > 0.85, note it and justify keeping both.

---

## Phase 8: Baselines, Best-of-N, DeepSeek Analysis, and Final Evaluation
**Duration: 1–2 weeks | Build: all comparisons and the secondary findings**

### Run all six baselines

```python
TOKEN_CAPS = {0: 0, 1: 500, 2: 1500, 3: 3000}

def evaluate_policy(policy_name, policy_fn, test_set):
    total_tokens = 0
    passed = 0
    for problem in test_set:
        level = policy_fn(problem["failure_signature"])
        tokens = TOKEN_CAPS[level]
        result = run_repair(problem, tokens)
        total_tokens += tokens
        if result["status"] == "pass":
            passed += 1
    pass_rate = passed / len(test_set)
    cna = pass_rate / total_tokens
    return {"pass@1": pass_rate, "total_tokens": total_tokens, "CNA": cna}

evaluate_policy("no_repair",   lambda x: 0,                      test_set)
evaluate_policy("always_l1",   lambda x: 1,                      test_set)
evaluate_policy("always_l3",   lambda x: 3,                      test_set)
evaluate_policy("random",      lambda x: random.choice([1,2,3]), test_set)
evaluate_policy("oracle",      lambda x: x["true_label"],        test_set)
evaluate_policy("adaptive",    lambda x: route(x),               test_set)
evaluate_policy("best_of_n",   best_of_n_policy,                 test_set)
```

### ⚠️ NEW — Report results with bootstrap confidence intervals

With a test set in the range of 50–150 failures, a single "adaptive uses 35% fewer tokens" number is easy to overstate. Bootstrap it:

```python
import numpy as np

def bootstrap_ci(values, n_boot=1000, ci=95):
    boots = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lower, upper = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return np.mean(values), lower, upper
```

Apply this to Pass@1, total tokens, and CNA for each policy. Report `mean (95% CI: lower–upper)` in your results table.

### Best-of-N baseline — how to implement it

For a given token budget, generate N independent repair attempts (with temperature > 0) and pick the first one that passes tests.

```python
def best_of_n_policy(failure, n=3, tokens_per_attempt=500):
    """
    Budget equivalent to Level 2 (1500 tokens): 3 attempts × 500 tokens each.
    Uses temperature=0.7 so each attempt is different.
    """
    for attempt in range(n):
        repair = call_model(
            prompt=REPAIR_PROMPT.format(**failure),
            max_new_tokens=tokens_per_attempt,
            temperature=0.7      # must be > 0 or all attempts are identical
        )
        result = safe_execute(repair)
        if result["status"] == "pass":
            return "pass", attempt + 1
    return "fail", n

# Report: for the same total token budget (1500), does Best-of-3 at 500 tokens each
# beat your adaptive router's Level 2 assignment?
```

The comparison you are making: for problems your router sends to Level 2 (1500 tokens), does spending those 1500 tokens as one single repair vs. three 500-token parallel attempts change the pass rate? If Best-of-N wins, your core claim needs reframing. If your router wins, you have evidence that execution-guided routing beats brute-force sampling.

### Sandbox latency measurement — report this honestly

Your pipeline has overhead beyond token generation:
1. Sandbox spin-up + code execution time
2. Traceback extraction
3. Embedding forward pass (CodeBERT)
4. Router inference

Measure and report each component separately:

```python
import time

timings = {}

t0 = time.perf_counter()
execution_result = safe_execute(generated_code)
timings["sandbox_ms"] = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
embedding = embed_text(execution_result["traceback"])
timings["embedding_ms"] = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
predicted_level = route(combined_features)
timings["routing_ms"] = (time.perf_counter() - t0) * 1000
```

In your paper, report the mean values. The expected pattern: routing inference is microseconds, CodeBERT embedding is ~5–20ms per sample on GPU, sandbox execution (100–500ms per run) is the dominant cost and unavoidable in any execution-guided system. Acknowledge in the limitations section that total wall-clock time per problem is higher than token cost alone suggests.

### DeepSeek-R1 thinking token analysis

By this phase you have stored `think_tokens` and `code_tokens` for every DeepSeek generation. Now analyze them:

```python
import pandas as pd

df_deepseek = df[df["model"] == "deepseek-r1-distill-qwen-7b"].copy()

# Group by exception type, compute mean thinking tokens
print(df_deepseek.groupby("exception_type")[["think_tokens", "code_tokens"]].mean())
```

If your dataset is large enough (check group sizes — some exception types may have very few DeepSeek examples), train a secondary XGBoost regressor to predict `think_tokens` from the failure signature, and run SHAP on it. If group sizes are too small for a model, report the groupby table directly as a descriptive finding instead — that's still useful.

**Key questions to answer:**
1. Which exception types require the most thinking tokens on average?
2. Does `think_tokens` correlate with repair label (L1/L2/L3/Unrecoverable)?
3. When DeepSeek is given a 500-token cap for a Level 1 repair, what fraction of that is spent in `<think>` vs. actual code? Is the code output being truncated?

Write these up as a standalone subsection, framed as an exploratory finding given the likely small per-category sample sizes.

### Calibration (Brier Score)

```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, predicted_probs)
print("Brier Score:", brier)
# 0.0 = perfect | 0.25 = no skill | lower is better
```

### Generalization experiments

**Cross-model:** Train on Qwen data only. Test on DeepSeek data only. Report AUROC and MAE with confidence intervals, and note explicitly that this is a preliminary/directional result given the dataset sizes involved.

**Cross-difficulty:** Split LiveCodeBench by difficulty. Train on easy + medium. Test on hard only.

> **⚠️ UPGRADE — Use Weights & Biases for experiment tracking**
>
> Do not track results in text files. Use **Weights & Biases** (free for academics): https://wandb.ai
>
> ```python
> import wandb
> wandb.init(project="compute-allocation", config={"model": "qwen2.5", "policy": "adaptive"})
> wandb.log({"pass@1": 0.73, "total_tokens": 14200, "CNA": 5.14e-5, "AUROC": 0.81})
> ```
>
> Every run logs your git commit hash automatically. Share the dashboard link in your README and in emails to professors.
> Free academic signup: https://wandb.ai/site/research

---

## Parallel Reading: Papers to Read While You Build

One per week, while building phases 3–8.

| While building | Paper | Why |
|---|---|---|
| Phase 3 | "Evaluating Large Language Models Trained on Code" (Codex) — https://arxiv.org/abs/2107.03374 | Understand pass@k and code eval methodology |
| Phase 4 | "Is Self-Repair a Silver Bullet for Code Generation?" — https://arxiv.org/abs/2306.09896 | Direct related work to your project |
| Phase 4 | CodeBERT paper — https://arxiv.org/abs/2002.08155 | Understand what model you are using for embeddings and why |
| Phase 6 | Snell et al. 2024 — full read now | You skimmed this in Phase 1. Now read it completely. |
| Phase 7 | "A Unified Approach to Interpreting Model Predictions" (SHAP paper) — https://arxiv.org/abs/1705.07874 | Understand what SHAP actually computes |
| Phase 8 | "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" — https://arxiv.org/abs/2407.21787 | The Best-of-N literature your baseline comes from |

---

## Tools and Libraries — Complete List

| Tool | What it does in your project | Install |
|---|---|---|
| `vllm` | Offline batch inference for Qwen + DeepSeek | `pip install vllm` |
| `transformers` | CodeBERT embeddings + load models from Hugging Face | `pip install transformers` |
| `torch` | Required for CodeBERT forward pass | `pip install torch` |
| `e2b-code-interpreter` | Secure sandboxed code execution (Option A) | `pip install e2b-code-interpreter` |
| Docker | Isolated execution with hard resource limits (Option B) | Install Docker Desktop |
| `radon` | Cyclomatic complexity, LOC metrics | `pip install radon` |
| `xgboost` | Both classifiers (recoverability + budget router) | `pip install xgboost` |
| `shap` | Feature importance + dependence plots | `pip install shap` |
| `statsmodels` | Optional: `OrderedModel` for proper ordinal regression comparison | `pip install statsmodels` |
| `scikit-learn` | GroupKFold, PCA, class weights, metrics | `pip install scikit-learn` |
| `pandas` | Managing the dataset | `pip install pandas` |
| `numpy` | Feature matrix operations | `pip install numpy` |
| `jsonlines` | Checkpointed experiment data | `pip install jsonlines` |
| `wandb` | Experiment tracking and dashboards | `pip install wandb` |
| LiveCodeBench | Benchmark + evaluation harness | `git clone` + `poetry install` |

*(Dropped from the previous version: `mord` — small, unmaintained ordinal-regression package that can conflict with current numpy/sklearn. Replaced with XGBoost regression + rounding as primary, `statsmodels.OrderedModel` as an optional, actively-maintained comparison.)*

---

## Timeline

| Phase | Duration | Output |
|---|---|---|
| 0 — Foundations check | 1 week | Python, ML, Git solid |
| 1 — Read the papers | 1 week | Understand the problem space |
| 2 — LLM basics + local inference + DeepSeek parser | 1 week | Model running locally, batch inference working, think/code parser built |
| 3 — LiveCodeBench setup + pilot + Stage 1+2 | 1–2 weeks | Pilot failure-rate numbers, raw failure dataset with safe execution |
| 4 — Failure signature extractor (tabular + PCA-reduced embeddings) | 1–2 weeks | Stage 3 complete, reduced + full feature sets saved |
| 5 — Repair ladder | 2 weeks | Full training dataset with checkpointing |
| 6 — Classifiers + ablation | 1 week | Stage 5+6, ablation table, leak-free GroupKFold |
| 7 — SHAP + AssertionError analysis | 4 days | Dependence plots + written observations |
| 8 — Baselines + Best-of-N + DeepSeek analysis + latency + CIs | 1–2 weeks | All results in W&B dashboard, with confidence intervals |
| **Total** | **~12 weeks** | **Complete, reproducible project** |

---

## Final Checklist: What Your Repository Must Have

Before emailing any professor or submitting anywhere, verify all of these:

- [ ] **`pyproject.toml` or `requirements.txt`** with exact pinned versions of all libraries
- [ ] **E2B or Docker sandbox** for all code execution — no raw `subprocess` on untrusted code
- [ ] **Pilot study numbers** (Phase 3) reported: actual failure counts, exception-type distribution, dataset size
- [ ] **`StratifiedGroupKFold` by `problem_id`** — documented in methodology, no data leakage
- [ ] **CodeBERT embeddings** of traceback + failed code stored as `.npy` files, both raw and PCA-reduced, loadable without re-running
- [ ] **Ablation table** comparing tabular-only, embedding-only, and combined (reduced) models, plus full-dimension comparison
- [ ] **AssertionError subset analysis** — AUROC breakdown for this class specifically, with sample size noted
- [ ] **Best-of-N baseline results** — at matched token budgets
- [ ] **Sandbox latency measurements** — per-component timing reported honestly
- [ ] **DeepSeek thinking token breakdown** — per exception type, with secondary analysis (model or descriptive table depending on sample size)
- [ ] **Confidence intervals** on Pass@1, total tokens, and CNA for all policies
- [ ] **W&B dashboard link** in README showing CNA curves for all baselines
- [ ] **"Unrecoverable" definition** written as a docstring in label assignment code and in the paper

---

## The Most Important Advice

**Learn by building, not by watching.**
Every phase has a concrete artifact to produce. Do not spend three weeks watching lectures. Watch one video, build the thing, then move on.

**Your dataset is the project.**
The repair ladder data — failure signatures + embeddings mapped to minimum repair levels — is the novel contribution. Even if your classifiers perform poorly, you have built a dataset that other researchers can use. Protect it. Back it up in two places. Document every field.

**Run the Phase 3 pilot before anything else.** It's a half-day investment that tells you whether your dataset will be 150 or 500 failures — and that number quietly determines almost every modeling decision after it.

**The seven traps that kill research projects:**
1. Using server-mode inference in a loop for hundreds of generations → use offline batch
2. Running untrusted code without sandboxing → use E2B or Docker
3. Random train/test split across same problem IDs → use GroupKFold
4. Skipping the pilot and discovering your dataset is too small after Phase 5 → run the Phase 3 pilot first
5. Feeding 1500+ raw features to a model trained on a few hundred rows → PCA-reduce embeddings before fusion
6. Script crashes partway through with no checkpoint → write checkpoint from day one
7. Forgetting to store `think_tokens` / `code_tokens` on DeepSeek runs → parse and save them in Phase 2, before Phase 5 begins
