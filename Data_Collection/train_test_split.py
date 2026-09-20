import hashlib

def train_test_split(url):
    hash_code = hashlib.sha256(url.encode('utf-8'))
    hash_int = int(hash_code.hexdigest(), 16) % 100
    file = ''

    if hash_int < 10:
        return './Data/eval_dataset.jsonl'
    else:
        return './Data/train_dataset.jsonl'
