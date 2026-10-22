"""Paired FT-20 vs FT-500 analysis on the committed logs.

Reproduces every derived number in the README that the tier scripts do not:
paired sign tests, the both-valid subset, failure-mode counts, the judge
few-shot exclusion, the seed-holdout split and the contamination check.

Run from the repo root:  python Model_Evaluation/paired_analysis.py
"""
import json
import difflib
from math import comb
from validity_evaluation import is_strictly_valid, is_correct_error_code

LOGS = './logs'
REPORT_FILE = './logs/paired_analysis.txt'
JUDGE_SOURCE = './Model_Evaluation/LLM_judge.py'
SEED_HOLDOUT = './Data/ft20_eval_holdout.jsonl'
TRAIN_FULL = './Data/train_dataset_full.jsonl'
EVAL_FULL = './Data/eval_dataset_full.jsonl'
MEMORIZATION_THRESHOLD = 0.6   # string similarity to ground truth above which an output is flagged

lines = []
def report(text=''):
    print(text)
    lines.append(text)


def load_by_index(path: str) -> dict:
    with open(path, 'r') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row['example_index']: row for row in rows}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test on discordant pairs (exact McNemar for binary outcomes)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def error_code_hit(row: dict) -> bool:
    output = row['generated_output']
    return is_strictly_valid(output) and is_correct_error_code(row['ground_truth'], output)


def failure_mode(output: str) -> str:
    text = output.strip()
    if not text:
        return 'empty'
    try:
        json.loads(text)
        return 'parses, wrong schema'
    except json.JSONDecodeError:
        pass
    try:
        json.JSONDecoder().raw_decode(text)
        return 'valid JSON + trailing text'
    except json.JSONDecodeError:
        return 'not parseable'


def user_inputs(path: str) -> set:
    with open(path, 'r') as f:
        return {json.loads(line)['messages'][1]['content'] for line in f if line.strip()}


outputs = {name: load_by_index(f'{LOGS}/{name}_model_outputs.jsonl') for name in ('base', 'ft20', 'ft500')}
judge = {name: {i: r['score'] for i, r in load_by_index(f'{LOGS}/tier3_{name}.jsonl').items()} for name in ('base', 'ft20', 'ft500')}
indices = sorted(outputs['ft20'])
assert indices == sorted(outputs['ft500']) == sorted(judge['ft20']) == sorted(judge['ft500']), 'logs are not aligned'
n = len(indices)


# 1. paired comparison, FT-500 vs FT-20
report(f'== Paired comparison, FT-500 vs FT-20 (n={n}, exact two-sided sign test) ==')
metrics = {
    'JSON validity': lambda name, i: int(is_strictly_valid(outputs[name][i]['generated_output'])),
    'error_code match': lambda name, i: int(error_code_hit(outputs[name][i])),
    'judge score': lambda name, i: judge[name][i],
}
for metric, value in metrics.items():
    diffs = [value('ft500', i) - value('ft20', i) for i in indices]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    report(f'{metric:18s} FT-500 better: {wins:2d}  FT-20 better: {losses:2d}  ties: {n - wins - losses:2d}  p = {sign_test(wins, losses):.3f}')


# 2. both-valid subset
both_valid = [i for i in indices if all(is_strictly_valid(outputs[m][i]['generated_output']) for m in ('ft20', 'ft500'))]
report(f'\n== Examples where both FT-20 and FT-500 emit valid JSON (n={len(both_valid)}) ==')
for name in ('ft20', 'ft500'):
    mean_judge = sum(judge[name][i] for i in both_valid) / len(both_valid)
    hits = sum(error_code_hit(outputs[name][i]) for i in both_valid)
    report(f'{name:6s} judge mean: {mean_judge:.2f}  error_code matches: {hits}/{len(both_valid)}')


# 3. failure modes
report('\n== Failure modes (outputs failing Tier 1) ==')
for name in ('base', 'ft20', 'ft500'):
    modes = {}
    for i in indices:
        output = outputs[name][i]['generated_output']
        if not is_strictly_valid(output):
            modes.setdefault(failure_mode(output), []).append(i)
    summary = ', '.join(f'{mode}: {len(ids)} {ids}' for mode, ids in modes.items())
    report(f'{name:6s} {summary}')


# 4. judge means with the judge's own few-shot example removed
with open(JUDGE_SOURCE, 'r', encoding='utf-8') as f:
    judge_prompt = f.read()
leaked = [i for i in indices
          if json.loads(outputs['ft20'][i]['ground_truth'])['root_cause_analysis'][:80] in judge_prompt]
kept = [i for i in indices if i not in leaked]
report(f'\n== Judge means excluding the judge few-shot example {leaked} (n={len(kept)}) ==')
for name in ('base', 'ft20', 'ft500'):
    report(f'{name:6s} {sum(judge[name][i] for i in kept) / len(kept):.2f}')


# 5. seed-holdout vs pipeline examples
seed_inputs = user_inputs(SEED_HOLDOUT)
seed = [i for i in indices if any(s in outputs['ft20'][i]['input'] for s in seed_inputs)]
pipeline = [i for i in indices if i not in seed]
report('\n== Judge means by eval subset ==')
for label, subset in (('seed holdout', seed), ('pipeline', pipeline)):
    diffs = [judge['ft500'][i] - judge['ft20'][i] for i in subset]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    report(f'{label:12s} n={len(subset):2d}  FT-20: {sum(judge["ft20"][i] for i in subset) / len(subset):.2f}  '
           f'FT-500: {sum(judge["ft500"][i] for i in subset) / len(subset):.2f}  '
           f'FT-500 better: {wins}  FT-20 better: {losses}  p = {sign_test(wins, losses):.3f}')


# 6. contamination checks
report('\n== Contamination checks ==')
report(f'train/eval input overlap: {len(user_inputs(TRAIN_FULL) & user_inputs(EVAL_FULL))}')
for name in ('base', 'ft20', 'ft500'):
    flagged = []
    for i in indices:
        row = outputs[name][i]
        similarity = difflib.SequenceMatcher(None, row['generated_output'].strip(), row['ground_truth']).ratio()
        if similarity > MEMORIZATION_THRESHOLD:
            flagged.append((i, round(similarity, 2)))
    report(f'{name:6s} outputs with similarity to ground truth > {MEMORIZATION_THRESHOLD}: {flagged if flagged else "none"}')


with open(REPORT_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')