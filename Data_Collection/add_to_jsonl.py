import json

def append_to_jsonl(filename: str, raw_log: str, rca_json: dict):
    """
    Constructs the final OpenAI/HuggingFace conversational format and appends to disk.
    """
    # The exact target schema required for fine-tuning
    finetune_row = {
        "messages": [
            {
                "role": "system", 
                "content": "You are an elite Site Reliability Engineer. Extract the root cause from the provided crash log. You must output ONLY a valid JSON object with the keys: severity, failing_service, error_code, root_cause_analysis, recommended_remediation."
            },
            {
                "role": "user", 
                "content": raw_log
            },
            {
                "role": "assistant", 
                "content": json.dumps(rca_json) # Assistant target MUST be a stringified JSON
            }
        ]
    }
    
    # append data to file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(finetune_row) + "\n")