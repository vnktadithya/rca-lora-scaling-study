from LLM_judge import evaluate_analysis
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def llm_judge_analysis(read_file: str, write_file):
    with open(read_file, 'r') as f, open(write_file, 'w') as w_f:
        for line in f:
            example = json.loads(line.strip())
            raw_input = example['input'].split('\n\n', 1)[1].split('[/INST]')[0].strip()
            ground_truth = example['ground_truth']
            model_prediction = example['generated_output']
            example_index = example['example_index']
            result = evaluate_analysis(raw_input, ground_truth, model_prediction)

            if result is None:
                print(f"[WARN] example_index {example_index} failed after retries — skipping")
                continue
            
            data = {'example_index': example_index,
                    'score': result['score'], 
                    'reasoning': result['reasoning'],
                    'prompt_tokens': result['prompt_tokens'],
                    'output_tokens': result['output_tokens']
                }
            w_f.write(json.dumps(data) + '\n')
            w_f.flush()
            print(f"write_file: {write_file} ---> example_index {example_index} -> score {result['score']}")

llm_judge_analysis('./logs/base_model_outputs.jsonl','./logs/tier3_base.jsonl')
llm_judge_analysis('./logs/ft20_model_outputs.jsonl','./logs/tier3_ft20.jsonl')
llm_judge_analysis('./logs/ft500_model_outputs.jsonl','./logs/tier3_ft500.jsonl')