# RCA LoRA Scaling Study

Does scaling a QLoRA fine-tune of Mistral-7B from 20 to 474 training examples
improve structured root-cause analysis of crash logs? On 53 held-out examples:

- **Reliability improves, significantly.** FT-500 emits schema-valid JSON on
  0.91 of examples vs. 0.74 for FT-20, and names the correct exception on
  0.70 vs. 0.49. Paired sign tests: p = 0.012 and p = 0.001.
- **Diagnosis quality improves less, and not significantly.** LLM-judge mean
  2.13 vs. 1.85 (1–5 scale), p = 0.099. On the 38 examples where both models
  emit valid JSON, the gap shrinks to 2.05 vs. 1.92.
- **Both models still invent identifiers** that appear nowhere in their input.

> **Correction, October 2026.** The first version of this study (September
> 2026) reported the opposite conclusion: no gain from more data, and more
> fabrication. Two bugs caused it — FT-500 was served with FT-20's weights,
> and the FT-20 checkpoint had seen 5 of the 7 seed-holdout examples. Both are
> fixed and every output, score and chart was regenerated.
> See [Correction history](#correction-history).

## Results

Same 53 held-out examples for all three checkpoints, greedy decoding.

| Metric | Base | FT-20 | FT-500 |
|---|---|---|---|
| JSON validity (Tier 1) | 0.00 | 0.74 | **0.91** |
| `error_code` exact match (Tier 1) | 0.00 | 0.49 | **0.70** |
| Root-cause similarity, cosine (Tier 2) | 0.00 | 0.49 | **0.63** |
| Remediation similarity, cosine (Tier 2) | 0.00 | 0.38 | **0.49** |
| LLM-judge mean, 1–5 (Tier 3) | 1.21 | 1.85 | **2.13** |

![Benchmark comparison chart](plots/tier_comparison_chart.png)

Tiers 1–2 and the chart: `Model_Evaluation/performance_comparision.py`.
Tier 3: `logs/tier3_{base,ft20,ft500}.jsonl`.

### Paired comparison, FT-500 vs. FT-20

Both adapters answer the same 53 inputs, so they are compared example by
example. p is an exact two-sided sign test over the examples where they differ
(for the binary metrics, this is the exact McNemar test).

| Metric | FT-500 better | FT-20 better | Tie | p |
|---|---|---|---|---|
| JSON validity | 10 | 1 | 42 | 0.012 |
| `error_code` match | 11 | 0 | 42 | 0.001 |
| Judge score | 20 | 10 | 23 | 0.099 |

On the 38 examples where both emit valid JSON, the judge means are 1.92
(FT-20) and 2.05 (FT-500), and `error_code` matches are 26 and 29. Most of
FT-500's advantage is producing usable output more often, not diagnosing
better once both produce output. Tier 2 shows the same effect, because invalid
outputs score 0 there.

Source: `Model_Evaluation/paired_analysis.py`, which writes
`logs/paired_analysis.txt`: the paired tests, the both-valid subset, the
failure modes, the subset split, the few-shot exclusion and the contamination
checks.

### How each model fails

Outputs that fail Tier 1:

| Checkpoint | Valid JSON, then extra text | Not parseable | Empty |
|---|---|---|---|
| Base | 0 | 43 | 10 |
| FT-20 | 8 | 6 | 0 |
| FT-500 | 4 | 1 | 0 |

FT-20 often fails to stop after the JSON object and appends invented sections
such as `USER_CONTEXT:`. FT-500's four trailing-text failures are short: a
stray ` [/INST]` (examples 14 and 52) or an opened code fence (26 and 45).

### Invented identifiers the judge did and didn't catch

The judge scores differ by 2 points on 11 examples. In five of them, at least
one output names an identifier that does not occur in its input. Each can be
checked by string search over the `input` field of
`logs/{ft20,ft500}_model_outputs.jsonl` at that `example_index`.

| # | Judge (FT-20 / FT-500) | Not in the input |
|---|---|---|
| 40 | 2 / 4 | FT-20: a `tables` argument |
| 36 | 1 / 3 | FT-20: `accelerate.vision`. FT-500: `CompiledModel` |
| 28 | 2 / 4 | FT-500: `_print_checkpoint_tip`, `fit.py` — scored 4 anyway |
| 16 | 4 / 2 | FT-20: `poetry.publisher.pypi` — scored 4 anyway |
| 7 | 4 / 2 | FT-500: `not is subclass(...)`, which is not valid Python |

Examples 28 and 16 show the judge's main blind spot: a plausible-sounding
invented identifier can still earn a 4.

## Task

Input: a crash log — a user bug report plus a traceback. Output: a JSON object
with exactly five keys: `severity`, `failing_service`, `error_code`,
`root_cause_analysis`, `recommended_remediation`. The raw output must pass
`json.loads()` as is, with no markdown fences or surrounding text.

## Data

Examples are synthesized from real bug reports in public GitHub issues and
pull requests.
`Data_Collection/generate_dataset.py` searches 82 Python repositories
(`TARGET_REPOS`), filters candidates, and deduplicates them by embedding
similarity. Gemini converts each accepted report into an
`(input_log, ground_truth_json)` pair. 64 of the 82 repositories contributed
at least one accepted example.

Pipeline yield, from `Data/pipeline_state.json`:

| Outcome | Count |
|---|---|
| Candidates evaluated | 1,549 |
| Rejected: no traceback | 637 |
| Rejected: insufficient context | 330 |
| Rejected: pull request, not a bug report | 63 |
| Rejected: near-duplicate | 18 |
| Accepted | 501 (454 train / 47 eval) |
| Seed set, from an earlier run of the same pipeline | 27 (20 train / 7 eval) |

| File | Rows | Used for |
|---|---|---|
| `Data/ft20_train_dataset.jsonl` | 20 | FT-20 training (seed train) |
| `Data/train_dataset_full.jsonl` | 474 | FT-500 training (the same 20 plus 454 pipeline rows) |
| `Data/eval_dataset_full.jsonl` | 54 | Evaluation (7 seed holdout, then 47 pipeline) |

One eval example (5,071 prompt tokens) does not fit the serving context
window and is excluded, which leaves 53 (`logs/excluded_examples.log`). No
eval input appears in either training set.

## Training

The two runs differ in training data and in training length.

| | FT-20 | FT-500 |
|---|---|---|
| Base model | `mistralai/Mistral-7B-v0.1` | same |
| Quantization | 4-bit NF4, double quantization, bf16 compute | same |
| LoRA | `r=8`, `alpha=32`, `dropout=0.05`, `q_proj` + `v_proj`, `bias=none` | same |
| Prompt | `<s>[INST] {system}\n\n{user} [/INST] {assistant}</s>`, loss on the full sequence | same |
| Optimizer | `paged_adamw_32bit`, lr `2e-4`, effective batch size 4 | same |
| Length | `max_steps=100`: 400 example passes, 20 epochs | `num_train_epochs=3`: 1,422 example passes |

Hardware: RTX 5060 Laptop GPU (8 GB), WSL2 Ubuntu, CUDA 13.0.

Serving: vLLM 0.29.0. One base-model instance serves all three checkpoints;
each adapter is attached through a `LoRARequest` with its own `lora_int_id`.
Decoding is greedy (`temperature=0`), so a difference between checkpoints
cannot come from sampling.

## Evaluation

**Tier 1: format.** `json.loads()` on the raw output, which must yield exactly
the five keys. `error_code` is then compared to the ground truth by exact
string match. The denominator is 53 for every checkpoint, so invalid outputs
count as failures instead of shrinking the sample.

**Tier 2: semantic similarity.** Cosine similarity of `all-MiniLM-L6-v2`
embeddings between predicted and ground-truth `root_cause_analysis`, and
separately `recommended_remediation`. Outputs that fail Tier 1 score 0.

**Tier 3: LLM judge.** `gemini-3.8-flash` scores root cause and remediation
together on a 1–5 rubric, given the input log and the ground truth. It scores
the raw text of every output, including outputs that fail Tier 1 — which is
why Base scores 1.21 rather than 0. The rubric was calibrated against my own
hand scores on a pilot set. Calibration added one rule: an identifier that
actually appears in the traceback earns credit; restating the user's report
does not.

## Correction history

The first version (September 2026, commit `c35d1f1`) concluded that FT-500 was
no better than FT-20 and fabricated more. Two bugs produced that result.

1. **FT-500 was served with FT-20's weights.** `output_generation.py` gave both
   adapters `lora_int_id=1`. vLLM treats the ID as the adapter's identity and
   reuses an already-loaded adapter with the same ID, so the "FT-500" run used
   FT-20. The reported judge gap (2.21 vs. 2.00) was the difference between two
   sampled runs of the same adapter at `temperature=0.1`. Fixed in `58ab2ea`.
2. **The old FT-20 checkpoint had seen the seed holdout.** On 5 of the 7
   seed-holdout examples, its outputs at `c35d1f1` reproduce the ground truth
   almost verbatim (difflib similarity ≥ 0.98). The committed FT-20 split
   does not contain those examples, so that checkpoint was not trained on it.
   FT-20 was retrained on `Data/ft20_train_dataset.jsonl`
   (`59eea05`). After retraining, no output from any checkpoint exceeds 0.6
   similarity to its ground truth (contamination check in
   `logs/paired_analysis.txt`).

The original outputs, scores and README remain in git history at `c35d1f1`.

## Limitations

- **n = 53.** Enough for the Tier 1 differences, not for the judge difference
  (p = 0.099).
- **Volume is confounded with training length and data source.** FT-500 saw
  3.5× more example passes. And 46 of the 53 eval examples come from the same
  pipeline run as FT-500's 454 new rows, while FT-20's 20 rows come from an
  earlier run. The judge gap follows the source: on the 46 pipeline examples
  FT-500 leads, 2.11 vs. 1.65 (20 wins to 6, p = 0.009); on the 7 seed-holdout
  examples FT-20 leads, 3.14 vs. 2.29 (4 wins to 0, p = 0.125). This split was
  examined after seeing the results, so it is an observation, not a finding —
  but part of FT-500's advantage may be distribution match, not volume.
- **One training run per checkpoint,** default seed. Run-to-run variance is
  not measured.
- **Adapter capacity is fixed** (`r=8`, `q_proj` and `v_proj` only). The
  results hold for this capacity only.
- **The judge has known weaknesses.** It misses plausible invented identifiers
  (examples 28 and 16). Its few-shot example is eval example 7, ground truth
  included; dropping example 7 gives 1.21 / 1.81 / 2.13 (n = 52) and changes
  no conclusion. It was run once, at `temperature=0.1`.
- **Gemini writes the references and judges against them,** so a shared
  stylistic preference cannot be ruled out.
- **FT-500 logged eval loss every 30 steps** on the eval set. The final weights
  are used, with no checkpoint selection or early stopping, so the eval set
  does not shape the reported model.
- **One quantization and serving configuration.**

## Next experiments

A rank and target-module sweep at fixed data. A matched-compute control (FT-20
trained for 1,422 example passes). Three seeds per checkpoint. An automatic
invented-identifier rate, so the judge's blind spot is measured instead of
spot-checked.

## Reproduction

```bash
pip install -r requirements.txt
pip install datasets wandb          # training only; or set WANDB_MODE=disabled
```

`.env` in the repo root:

```
GEMINI_API_KEY=<key>
GEMINI_MODEL=gemini-3.8-flash
GitHub_PAT=<token>                  # data collection only
```

From the repo root:

```bash
python Data_Collection/generate_dataset.py         # optional; data is committed
python Model_Training/train_ft20.py                # GPU
python Model_Training/train_ft500.py               # GPU
python Model_Evaluation/output_generation.py       # GPU; all three checkpoints
python Model_Evaluation/LLM_judge_evaluation.py    # Tier 3
python Model_Evaluation/performance_comparision.py # Tiers 1–2 and the chart
python Model_Evaluation/paired_analysis.py         # paired tests, failure modes, contamination checks
```

## Repository layout

```
Data/              splits and pipeline ledger
Data_Collection/   GitHub collection, filtering, deduplication, synthesis
Model_Training/    QLoRA training (FT-20, FT-500)
Checkpoints/       LoRA adapters (~14 MB each)
Model_Evaluation/  generation, the three scoring tiers, paired analysis
logs/              raw outputs, judge scores, exclusions, analysis report
plots/             benchmark chart
```

## License

Code: MIT ([LICENSE](LICENSE)). Data is synthesized from public GitHub issues
and pull requests via the Gemini API; no private repositories or proprietary code were used.