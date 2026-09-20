import json
from pipeline_state import load_state, mark_processed

current_state = load_state()
count = 1
with open('./Data/Log_telemetry_dataset.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        example = json.loads(line)
        raw_log = example['messages'][1]['content']
        key = f'_seed#{count}'
        mark_processed(current_state, key, "accepted", raw_log)
        count += 1