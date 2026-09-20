import json
from sentence_transformers import SentenceTransformer, util

_model = None

def _get_model(): #helper function to load the model with a global variable.
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_texts(text: str):
    return _get_model().encode(text)


EXPECTED_KEYS = {'severity', 'failing_service', 'error_code', 'root_cause_analysis', 'recommended_remediation'}

def is_strictly_valid(output: str)-> bool:
    try:
        obj = json.loads(output)
    except json.JSONDecodeError as e:
        return False

    return isinstance(obj, dict) and set(obj.keys()) == EXPECTED_KEYS

def compute_similarity(file: str):
    rca_scores = []
    remedy_scores = []
    total_count = 0
    validity_count = 0
    with open(file, 'r') as f:
        for line in f:
            total_count += 1
            data = json.loads(line.strip())
            generated_output = data['generated_output']
            ground_truth = data['ground_truth']

            if is_strictly_valid(generated_output):
                validity_count += 1
                model_output_json = json.loads(generated_output.strip())
                ground_truth_json = json.loads(ground_truth.strip())

                #rca similarity score
                model_rca = model_output_json['root_cause_analysis']
                model_output_rca_embedding = embed_texts(model_rca)
                ground_truth_rca = ground_truth_json['root_cause_analysis']
                ground_truth_rca_embedding = embed_texts(ground_truth_rca)
                rca_scores.append((util.cos_sim(model_output_rca_embedding, ground_truth_rca_embedding)).item())

                #remedy similarity score
                model_output_remedy = model_output_json['recommended_remediation']
                model_output_remedy_embedding = embed_texts(model_output_remedy)
                ground_truth_remedy = ground_truth_json['recommended_remediation']
                ground_truth_remedy_embedding = embed_texts(ground_truth_remedy)
                remedy_scores.append((util.cos_sim(model_output_remedy_embedding, ground_truth_remedy_embedding)).item())

            else:
                rca_scores.append(0.0)
                remedy_scores.append(0.0)


    return rca_scores, remedy_scores
