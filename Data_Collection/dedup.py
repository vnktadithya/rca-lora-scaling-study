from sentence_transformers import SentenceTransformer, util
import re

def extract_signature(raw_log: str) -> str | None:
    """Take the last match in the text and normalize whitespace
    before returning. Returns the pattern if exists and None if 
    no match"""
    error_pattern = (r"\b(?:[A-Z][a-zA-Z]+(?:Error|Exception|Fail)):\s+.+")
    matches = re.findall(error_pattern, raw_log)
    if not matches:
        return None

    return " ".join(matches[-1].split())

_model = None

def _get_model(): #helper function to load the model with a global variable.
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_texts(texts: list[str]):
    """return the embeddings of the raw_logs"""
    return _get_model().encode(texts)

def is_duplicate(candidate_raw_log: str, corpus: list[dict]):
    """corpus is a list of {"raw_log": str, "embedding": ..., "signature": str}.
    Check candidate's signature against every corpus signature first — if any 
    hits, return True. Otherwise fall back to max cosine similarity against
     corpus embeddings, with a threshold value."""

    candidate_signature = extract_signature(candidate_raw_log)
    
    for i, item in enumerate(corpus):
        if candidate_signature is None or item["signature"] is None:
            continue
        if item["signature"] in candidate_signature or candidate_signature in item["signature"]:
            print('Signature found')
            return True, 1.0

    candidate_embedding = embed_texts([candidate_raw_log])[0]
    max_score = 0.0
    for i, item in enumerate(corpus):
        similarity_score = float(util.cos_sim(candidate_embedding, item['embedding']))
        max_score = max(max_score, similarity_score)
        if similarity_score > 0.90:
            print(f'Score: {similarity_score}. Duplicate found')
            return True, similarity_score

    print('Not a duplicate')
    return False, max_score
