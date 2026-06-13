# Learning Roadmap: Final Edition (Updated)
## Learning Compute Allocation Policies from Execution Feedback

Organized by the **order you build things**, not by topic.
Each phase tells you what to learn, what to build, exactly where to learn it, and the engineering traps that will cost you weeks if you don't know about them.

---

## Before You Start: Foundations Check

You need these before touching the project. If you have them, skip.

**Python (must be solid)**
- File I/O, subprocess, error handling, classes, decorators
- If shaky: https://cs50.harvard.edu/python/2022/ (free, YouTube)

**Basic ML (must know what a classifier is)**
- What train/test split means, what a feature vector is, what overfitting means
- If shaky: fast.ai Practical Deep Learning Part 1 — https://course.fast.ai (free)

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
- What to understand: how the benchmark works, what pass@1 means, why contamination matters

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

**Andrej Karpathy — Let's build GPT (optional but gold)**
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- Only if you want deeper intuition. Not required to ship the project.

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
> The server mode is fine for testing. But in Phase 5 (The Repair Ladder), you will need to run ~3,000 LLM generations (500 problems × 2 models × 3 repair levels). Making 3,000 sequential API calls will take **days**.
>
> **The fix:** Learn **vLLM Offline Batch Inference** now, before you need it.
> Instead of calling a server in a loop, pass all prompts at once:
> ```python
> from vllm import LLM, SamplingParams
> llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct")
> params = SamplingParams(temperature=0, max_tokens=500)
> outputs = llm.generate(list_of_1000_prompts, params)
> ```
> vLLM's PagedAttention and continuous batching processes all of them together.
> Same 3,000 generations: **minutes, not days.**
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

## Phase 3: Set Up LiveCodeBench and Run Baselines
**Duration: 1–2 weeks | Build: Stage 1 and Stage 2**

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

### Learn Python's `ast` module and code execution

- Python `subprocess` docs: https://docs.python.org/3/library/subprocess.html
- Python `ast` module docs: https://docs.python.org/3/library/ast.html
- Real Python guide on AST: https://realpython.com/python-ast/

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
**Duration: 1–2 weeks | Build: Stage 3 — now includes semantic embeddings**

This phase is larger than the previous roadmap because you are building the combined tabular + embedding feature vector. This is the single biggest upgrade from the earlier version and the one that most affects classifier quality.

### Part A — Tabular Features (same as before)

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
One-hot or label-encode the top 15 most common keywords in your dataset. This captures what went wrong semantically, not just structurally.

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

### Part B — Semantic Embedding Features (new, required)

This is what separates the current version from the earlier roadmap. Two `AssertionError` failures can look identical in tabular features but require completely different repair costs. Embeddings let the classifier read the actual content of the error.

**What to learn first:**

**3Blue1Brown — Neural Networks series (Chapters 1–3 only)**
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Understand what an embedding vector is and why similar inputs produce similar vectors.
- Chapters 1–3 only. You do not need backprop for this project.

**Hugging Face — Feature Extraction pipeline**
- https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FeatureExtractionPipeline
- 15 minutes. Shows you how to get a dense vector from any text.

**Model to use — microsoft/codebert-base:**
- https://huggingface.co/microsoft/codebert-base
- 125M parameters, runs on any GPU with 4GB VRAM, produces 768-dimensional vectors
- Pre-trained on code and natural language — designed exactly for code + traceback text

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

**Combining tabular + embedding features:**
```python
import numpy as np

tabular_vec = np.array(list(tabular_features.values()))         # ~20 features
traceback_emb = np.array(traceback_embedding)                   # 768 features
code_emb = np.array(code_embedding)                             # 768 features

combined = np.concatenate([tabular_vec, traceback_emb, code_emb])  # ~1556 features
```

This combined vector is what gets fed to XGBoost in Phase 6.

> **⚠️ NOTE — Dimensionality and XGBoost**
>
> XGBoost handles high-dimensional input well — 1556 features is not a problem. However, the embedding dimensions will dominate any per-feature importance plot. Before running SHAP, use PCA to compress each 768-dim embedding to 32–64 dimensions for the interpretability analysis only (keep full dimensions for training):
> ```python
> from sklearn.decomposition import PCA
> pca = PCA(n_components=50)
> traceback_emb_reduced = pca.fit_transform(all_traceback_embeddings)
> ```
> Report both: full-dimension model performance, and PCA-reduced SHAP analysis for interpretability. This is standard practice and reviewers will expect it.

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
> 500 problems × 2 models × up to 3 repair attempts = up to 3,000 LLM generations + 3,000 sandboxed code executions. This will take hours. CUDA OOM errors, Docker timeouts, and network blips will crash it partway through.
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
> When the script crashes at problem 347, restart it. It reads the checkpoint, skips 1–346, and resumes at 347.

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

## Phase 6: Learn XGBoost + Train the Classifiers
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
- `mord` library for ordinal regression: https://pythonhosted.org/mord/

> **⚠️ CRITICAL — Data Leakage via problem_id**
>
> You run both Qwen and DeepSeek on the same LiveCodeBench problems. If you use a random `train_test_split`, Problem #42 (Qwen attempt) might land in the training set while Problem #42 (DeepSeek attempt) lands in the test set. Your XGBoost model will "cheat" — it will learn the difficulty of the problem itself rather than the failure signature.
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

> **⚠️ UPGRADE — Handle class imbalance**
>
> Your dataset will likely be imbalanced. XGBoost will learn to ignore rare classes.
> ```python
> from sklearn.utils.class_weight import compute_class_weight
> weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
> # Pass to XGBoost via sample_weight parameter in .fit()
> ```

### Run the ablation study — this is required, not optional

You must train four versions of each classifier and report all four:

| Model version | Features used | Purpose |
|---|---|---|
| Tabular only | ~20 handcrafted features | Baseline — what the old approach achieves |
| Traceback embedding only | 768-dim CodeBERT vector of traceback | How much does reading the error text help? |
| Code embedding only | 768-dim CodeBERT vector of failed code | How much does reading the code help? |
| Combined (tabular + both embeddings) | ~1556 features | Full model — your main result |

The gap between "tabular only" and "combined" on AssertionError cases specifically is your key finding. Filter your test set to only AssertionError failures and report AUROC for each model version on that subset. This is the single most anticipated result from reviewer feedback.

### Build Stage 5: Recoverability Classifier

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

X = combined_feature_matrix   # tabular + embeddings concatenated
y = binary_labels              # 0 = unrecoverable, 1 = recoverable
groups = problem_id_array

sgkf = StratifiedGroupKFold(n_splits=5)
auroc_scores = []

for train_idx, test_idx in sgkf.split(X, y, groups):
    model = xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight)
    model.fit(X[train_idx], y[train_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    auroc_scores.append(roc_auc_score(y[test_idx], probs))

print("Mean AUROC:", np.mean(auroc_scores))
```

### Build Stage 6: Budget Router

For recoverable failures only (labels 1, 2, 3):
```python
import mord

router = mord.OrdinalRidge(alpha=1.0)
router.fit(X_train, y_train)
preds = router.predict(X_test)
mae = np.mean(np.abs(preds - y_test))
print("Mean Absolute Level Error:", mae)
```

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
shap_values = explainer(X_test_pca)  # use PCA-reduced embeddings here for readability

shap.plots.beeswarm(shap_values)          # global feature importance
shap.plots.waterfall(shap_values[0])      # single prediction explanation
```

> **⚠️ UPGRADE — Use SHAP Dependence Plots for your paper figures**
>
> ```python
> shap.plots.scatter(shap_values[:, "cyclomatic_complexity"])
> shap.plots.scatter(shap_values[:, "exception_type_timeouterror"])
> ```
> Each plot shows one feature on the X-axis and its SHAP contribution on the Y-axis. A clear upward trend proves your hypothesis visually. These become your best paper figures.

### What to answer with SHAP (write these as numbered observations in your paper)

1. What is the single most important feature for recoverability prediction?
2. Does `exception_type_timeouterror` push predictions toward Unrecoverable?
3. Does `exception_type_syntaxerror` push predictions toward Level 1?
4. Does high `cyclomatic_complexity` correlate with Level 3?
5. Does `runtime_ms` close to the timeout limit predict unrecoverability?
6. Are the top 3 features the same for both Qwen failures and DeepSeek failures?
7. **For AssertionError failures specifically:** which embedding dimensions drive predictions? (This is the key finding that justifies embeddings over tabular features.)

### AssertionError misprediction analysis — write this as a paper section

Filter your test set to only AssertionError failures. For this subset:
- Compare tabular-only AUROC vs. combined AUROC
- Print the failure signatures of every case where tabular-only predicted RECOVERABLE but it was UNRECOVERABLE (the hard cases the tabular model misses)
- Check if embeddings correctly classify those same cases

Write 3–5 observations. This turns a classification result into a finding about what information actually determines repair cost when the exception type is ambiguous.

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
evaluate_policy("adaptive",    lambda x: router.predict([x])[0], test_set)
evaluate_policy("best_of_n",   best_of_n_policy,                 test_set)
```

### Best-of-N baseline — how to implement it

This is the most important new baseline. For a given token budget, instead of routing a single repair, generate N independent repair attempts (with temperature > 0) and pick the first one that passes tests.

```python
def best_of_n_policy(failure, n=3, tokens_per_attempt=500):
    """
    Budget equivalent to Level 2 (1500 tokens): 3 attempts × 500 tokens each.
    Uses temperature=0.7 so each attempt is different.
    """
    for _ in range(n):
        repair = call_model(
            prompt=REPAIR_PROMPT.format(**failure),
            max_new_tokens=tokens_per_attempt,
            temperature=0.7      # must be > 0 or all attempts are identical
        )
        result = safe_execute(repair)
        if result["status"] == "pass":
            return "pass", _ + 1  # passed on attempt number _+1
    return "fail", n

# Report: for the same total token budget (1500), does Best-of-3 at 500 tokens each
# beat your adaptive router's Level 2 assignment?
```

The comparison you are making: for problems your router sends to Level 2 (1500 tokens), does spending those 1500 tokens as one single repair vs. three 500-token parallel attempts change the pass rate? If Best-of-N wins, your core claim needs reframing. If your router wins, you have proved that execution-guided routing is better than brute-force sampling.

### Sandbox latency measurement — report this honestly

Your pipeline has overhead beyond token generation:
1. Sandbox spin-up + code execution time
2. Traceback extraction
3. Embedding forward pass (CodeBERT)
4. XGBoost inference

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
predicted_level = router.predict([combined_features])
timings["routing_ms"] = (time.perf_counter() - t0) * 1000
```

In your paper, report the mean values. The expected finding: XGBoost routing is microseconds, CodeBERT embedding is ~5–20ms per sample on GPU, sandbox execution is the dominant cost (100–500ms per run) and is unavoidable in any execution-guided system. Acknowledge in the limitations section that total wall-clock time per problem is higher than token cost alone suggests, and that this overhead must be weighed against token savings in latency-sensitive deployments.

### DeepSeek-R1 thinking token analysis

By this phase you have stored `think_tokens` and `code_tokens` for every DeepSeek generation. Now analyze them:

```python
import pandas as pd

df_deepseek = df[df["model"] == "deepseek-r1-distill-qwen-7b"].copy()

# Group by exception type, compute mean thinking tokens
print(df_deepseek.groupby("exception_type")[["think_tokens", "code_tokens"]].mean())

# Expected: SyntaxError → low think_tokens, TimeoutError → high think_tokens
```

Train a secondary XGBoost model to predict `think_tokens` (continuous, treat as regression) from the failure signature features. Run SHAP on this secondary model to identify which failure types drive thinking budget.

**Key questions to answer:**
1. Which exception types require the most thinking tokens on average?
2. Does `think_tokens` correlate with repair label (L1/L2/L3/Unrecoverable)?
3. When DeepSeek is given a 500-token cap for a Level 1 repair, what fraction of that is spent in `<think>` vs. actual code? Is the code output being truncated?

Write these up as a standalone subsection. A reviewer who called this a potential standalone contribution was not exaggerating — it is a novel empirical finding about how reasoning models allocate internal computation across failure types.

### Calibration (Brier Score)

```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, predicted_probs)
print("Brier Score:", brier)
# 0.0 = perfect | 0.25 = no skill | lower is better
```

### Generalization experiments

**Cross-model:** Train on Qwen data only. Test on DeepSeek data only. Report AUROC and MAE.

**Cross-difficulty:** Split LiveCodeBench by difficulty. Train on easy + medium. Test on hard only.

> **⚠️ UPGRADE — Use Weights & Biases for experiment tracking**
>
> Do not track results in text files. Use **Weights & Biases** (free for academics): https://wandb.ai
>
> ```python
> import wandb
> wandb.init(project="compute-allocation", config={"model": "qwen2.5", "policy": "adaptive"})
> wandb.log({"pass@1": 0.73, "total_tokens": 142000, "CNA": 5.14e-6, "AUROC": 0.81})
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
| Phase 5 | "Can LLMs Debug Their Own Code?" — search arxiv 2024 | What has already been tried |
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
| `xgboost` | Both classifiers | `pip install xgboost` |
| `shap` | Feature importance + dependence plots | `pip install shap` |
| `mord` | Ordinal regression for budget router | `pip install mord` |
| `scikit-learn` | GroupKFold, PCA, class weights, metrics | `pip install scikit-learn` |
| `pandas` | Managing the dataset | `pip install pandas` |
| `numpy` | Feature matrix operations | `pip install numpy` |
| `jsonlines` | Checkpointed experiment data | `pip install jsonlines` |
| `wandb` | Experiment tracking and dashboards | `pip install wandb` |
| LiveCodeBench | Benchmark + evaluation harness | `git clone` + `poetry install` |

---

## Timeline

| Phase | Duration | Output |
|---|---|---|
| 0 — Foundations check | 1 week | Python, ML, Git solid |
| 1 — Read the papers | 1 week | Understand the problem space |
| 2 — LLM basics + local inference + DeepSeek parser | 1 week | Model running locally, batch inference working, think/code parser built |
| 3 — LiveCodeBench + Stage 1+2 | 1–2 weeks | Raw failure dataset with safe execution |
| 4 — Failure signature extractor (tabular + embeddings) | 1–2 weeks | Stage 3 complete with CodeBERT embeddings |
| 5 — Repair ladder | 2 weeks | Full training dataset with checkpointing |
| 6 — XGBoost classifiers + ablation | 1 week | Stage 5+6, all four model versions, leak-free GroupKFold |
| 7 — SHAP + AssertionError analysis | 4 days | Dependence plots + written observations |
| 8 — Baselines + Best-of-N + DeepSeek analysis + latency | 1–2 weeks | All results in W&B dashboard |
| **Total** | **~12 weeks** | **Complete, reproducible project** |

---

## Final Checklist: What Your Repository Must Have

Before emailing any professor or submitting anywhere, verify all of these:

- [ ] **`pyproject.toml` or `requirements.txt`** with exact pinned versions of all libraries
- [ ] **E2B or Docker sandbox** for all code execution — no raw `subprocess` on untrusted code
- [ ] **`StratifiedGroupKFold` by `problem_id`** — documented in methodology, no data leakage
- [ ] **CodeBERT embeddings** of traceback + failed code stored as `.npy` files, loadable without re-running
- [ ] **Ablation table** comparing tabular-only, embedding-only, and combined models
- [ ] **AssertionError subset analysis** — AUROC breakdown for this class specifically
- [ ] **Best-of-N baseline results** — at matched token budgets
- [ ] **Sandbox latency measurements** — per-component timing reported honestly
- [ ] **DeepSeek thinking token breakdown** — per exception type, with secondary SHAP analysis
- [ ] **W&B dashboard link** in README showing CNA curves for all seven baselines
- [ ] **"Unrecoverable" definition** written as a docstring in label assignment code and in the paper

---

## The Most Important Advice

**Learn by building, not by watching.**
Every phase has a concrete artifact to produce. Do not spend three weeks watching lectures. Watch one video, build the thing, then move on.

**Your dataset is the project.**
The repair ladder data — failure signatures + CodeBERT embeddings mapped to minimum repair levels — is the novel contribution. Even if your classifiers perform poorly, you have built a dataset that other researchers can use. Protect it. Back it up in two places. Document every field.

**The six traps that kill research projects:**
1. Using server-mode inference in a loop for 3,000 generations → use offline batch
2. Running untrusted code without sandboxing → use E2B or Docker
3. Random train/test split across same problem IDs → use GroupKFold
4. Script crashes at generation 2,847 with no checkpoint → write checkpoint from day one
5. Tracking results in text files → use W&B from the first experiment
6. Forgetting to store `think_tokens` / `code_tokens` on DeepSeek runs → parse and save them in Phase 2, before Phase 5 begins
