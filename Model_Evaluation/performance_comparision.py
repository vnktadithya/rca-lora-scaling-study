import matplotlib.pyplot as plt
from validity_evaluation import compute_validty_rate_and_error_rate
from similarity_evaluation import compute_similarity
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#Tier 1
base_model_validity_rate, base_model_error_code_rate, base_invalid = compute_validty_rate_and_error_rate('./logs/base_model_outputs.jsonl')
ft20_model_validity_rate, ft20_model_error_code_rate, ft20_invalid = compute_validty_rate_and_error_rate('./logs/ft20_model_outputs.jsonl')
ft500_model_validity_rate, ft500_model_error_code_rate, ft500_invalid = compute_validty_rate_and_error_rate('./logs/ft500_model_outputs.jsonl')

#Tier 2
base_model_rca_scores, base_model_remedy_scores = compute_similarity('./logs/base_model_outputs.jsonl')
ft20_model_rca_scores, ft20_model_remedy_scores = compute_similarity('./logs/ft20_model_outputs.jsonl')
ft500_model_rca_scores, ft500_model_remedy_scores = compute_similarity('./logs/ft500_model_outputs.jsonl')

base_model_mean_rca_score = sum(base_model_rca_scores) / len(base_model_rca_scores)
base_model_mean_remedy_score = sum(base_model_remedy_scores) / len(base_model_remedy_scores)
ft20_model_mean_rca_score = sum(ft20_model_rca_scores) / len(ft20_model_rca_scores)
ft20_model_mean_remedy_score = sum(ft20_model_remedy_scores) / len(ft20_model_remedy_scores)
ft500_model_mean_rca_score = sum(ft500_model_rca_scores) / len(ft500_model_rca_scores)
ft500_model_mean_remedy_score = sum(ft500_model_remedy_scores) / len(ft500_model_remedy_scores)

#Tier 3
def get_mean_evaluation_score(file: str):
    with open(file, 'r') as f:
        scores = []
        for line in f:
            data = json.loads(line.strip())
            scores.append(data['score'])

    return sum(scores) / len(scores)

base_model_evaluation_mean_score = get_mean_evaluation_score('./logs/tier3_base.jsonl')
ft20_model_evaluation_mean_score = get_mean_evaluation_score('./logs/tier3_ft20.jsonl')
ft500_model_evaluation_mean_score = get_mean_evaluation_score('./logs/tier3_ft500.jsonl')

#plot graphs
models = ['Base', 'ft20', 'ft500']
colors = ['red', 'blue', 'green']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

def plot_panel(ax, title, values, ylim):
    bars = ax.bar(models, values, color=colors)
    ax.set_title(title)
    ax.set_ylim(0, ylim)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ylim * 0.02,
                 f'{v:.2f}', ha='center', va='bottom')

plot_panel(axes[0], 'JSON Validity Rate',
           [base_model_validity_rate, ft20_model_validity_rate, ft500_model_validity_rate], 1)

plot_panel(axes[1], 'Error Code Exact-Match Rate',
           [base_model_error_code_rate, ft20_model_error_code_rate, ft500_model_error_code_rate], 1)

plot_panel(axes[2], 'RCA Similarity (cosine)',
           [base_model_mean_rca_score, ft20_model_mean_rca_score, ft500_model_mean_rca_score], 1)

plot_panel(axes[3], 'Remedy Similarity (cosine)',
           [base_model_mean_remedy_score, ft20_model_mean_remedy_score, ft500_model_mean_remedy_score], 1)

plot_panel(axes[4], 'LLM-Judge Mean Score (1-5)',
           [base_model_evaluation_mean_score, ft20_model_evaluation_mean_score, ft500_model_evaluation_mean_score], 5)

axes[5].axis('off')  # unused 6th slot in the 2x3 grid

fig.suptitle('Base vs FT-20 vs FT-500 — RCA Fine-Tune Evaluation (n=53)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/tier_comparison_chart.png', dpi=200, bbox_inches='tight')
print("Saved chart to plots/tier_comparison_chart.png")