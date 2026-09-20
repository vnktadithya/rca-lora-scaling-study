import json
from sentence_transformers import SentenceTransformer, util

raw_logs = []
with open('./Data/Log_telemetry_dataset.jsonl', "r") as f:
    for line in f:
        line = line.strip()
        example = json.loads(line)
        raw_logs.append(example['messages'][1]['content'])

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(raw_logs)
similarity_matrix = util.cos_sim(embeddings, embeddings)

similarity_scores = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix[0])):
        similarity_scores.append([i, j, similarity_matrix[i][j]])

similarity_scores.sort(key = lambda x: x[2], reverse=True)

for i,j,score in similarity_scores:
    print(f'Score: {score}')
    print(f'Log 1: {raw_logs[i][:150]}')
    print(f'Log 2: {raw_logs[j][:150]}')
    print('-------------------------------------------------------------------')
