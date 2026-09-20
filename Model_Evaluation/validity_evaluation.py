import json

EXPECTED_KEYS = {'severity', 'failing_service', 'error_code', 'root_cause_analysis', 'recommended_remediation'}

def is_strictly_valid(output: str)-> bool:
    try:
        obj = json.loads(output)
    except json.JSONDecodeError as e:
        return False

    return isinstance(obj, dict) and set(obj.keys()) == EXPECTED_KEYS
    
def is_correct_error_code(ground_truth: str, model_output: str)-> bool:
    ground_truth_error_code = json.loads(ground_truth.strip())['error_code']
    model_output_error_code = json.loads(model_output.strip())['error_code']
    if str(ground_truth_error_code).strip() == str(model_output_error_code).strip():
        return True
    
    return False

def compute_validty_rate_and_error_rate(file: str):
    with open(file, 'r') as f:
        validity_count = 0
        total_count = 0
        error_code_count = 0
        invalid_indices = []
        for line in f:
            total_count += 1
            data = json.loads(line.strip())
            if is_strictly_valid(data['generated_output']):
                validity_count += 1
                if is_correct_error_code(data['ground_truth'], data['generated_output']):
                    error_code_count += 1
            else:
                invalid_indices.append(data['example_index'])

    model_validity_rate = validity_count / total_count
    model_error_code_rate = error_code_count / total_count

    return model_validity_rate, model_error_code_rate, invalid_indices
