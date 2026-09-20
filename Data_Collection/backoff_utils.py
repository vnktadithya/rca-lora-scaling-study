# Handle API-Rate limits for Gemini and GitHub
import random

def compute_backoff(attempt, base=2, cap=120):
    return min(cap, base * (2 ** attempt)) + random.uniform(0, 1)