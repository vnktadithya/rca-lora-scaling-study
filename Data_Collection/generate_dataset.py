from dotenv import load_dotenv
import requests
import re
import os
from pathlib import Path
import time
from LLM_Generator import generate_synthetic_telemetry, QuotaExhaustedError
from check_error_logs import contains_stack_trace_or_log
from add_to_jsonl import append_to_jsonl
from train_test_split import train_test_split
from backoff_utils import compute_backoff
from pipeline_state import load_state, is_processed, mark_processed, get_accepted_raw_logs
from dedup import extract_signature, embed_texts, is_duplicate

base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

GITHUB_TOKEN = os.getenv("GitHub_PAT")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN. Get one from GitHub Developer Settings.")

url = "https://api.github.com/search/issues"

headers = {
    "User-Agent": "InfrastructureTelemetryFetcher/1.0",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

TARGET_REPOS = [        # Repositories written in python
    "BerriAI/litellm",
    "run-llama/llama_index",
    "vibrantlabsai/ragas",
    "langchain-ai/langgraph",
    "stanfordnlp/dspy",
    "docling-project/docling",
    "deepset-ai/haystack",
    "crewaiinc/crewai",
    "psf/requests",
    "pallets/flask",
    "sqlalchemy/sqlalchemy",
    "scrapy/scrapy",
    "python-poetry/poetry",
    "huggingface/transformers",
    "pandas-dev/pandas",
    "dagster-io/dagster",
    "encode/uvicorn",
    "pytest-dev/pytest",
    "ansible/ansible",
    "encode/httpx",
    "Aider-AI/aider",
    "vllm-project/vllm",
    "microsoft/autogen",
    "pydantic/pydantic-ai",
    "dottxt-ai/outlines",
    "Unstructured-IO/unstructured",
    "getsentry/sentry-python",
    "boto/boto3",
    "getmoto/moto",
    "Kludex/starlette",
    "docker/docker-py",
    "lightning-ai/pytorch-lightning",
    "open-telemetry/opentelemetry-python",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "keras-team/keras",
    "scikit-learn/scikit-learn",
    "huggingface/accelerate",
    "huggingface/diffusers",
    "huggingface/datasets",
    "huggingface/peft",         
    "huggingface/tokenizers",
    "ray-project/ray",
    "optuna/optuna",
    "qdrant/qdrant-client",
    "weaviate/weaviate-python-client",
    "aio-libs/aiohttp",
    "python-attrs/attrs",
    "textualize/rich",
    "textualize/textual",
    "cookiecutter/cookiecutter",
    "strawberry-graphql/strawberry",
    "encode/django-rest-framework",
    "langchain-ai/langchain",
    "celery/celery",          
    "apache/airflow",         
    "dask/dask",              
    "fastapi/fastapi",       
    "pydantic/pydantic",
    "django/django",
    "PrefectHQ/prefect",
    "mlflow/mlflow",
    "gradio-app/gradio",
    "streamlit/streamlit",
    "pyca/cryptography",
    "paramiko/paramiko",
    "guidance-ai/guidance",
    "home-assistant/core",
    "ipython/ipython",
    "jupyterlab/jupyterlab",
    "scipy/scipy",
    "numpy/numpy",
    "statsmodels/statsmodels",
    "pydata/xarray",
    "fivetran/great_expectations",
    "apache/superset",
    "saltstack/salt",
    "HypothesisWorks/hypothesis",
    "PrefectHQ/marvin",
    "pypa/pip",
    "tornadoweb/tornado",
    "psycopg/psycopg2",
]

def fetch_real_rca_data(target_count):
    collected_examples = 0

    state = load_state()
    corpus = []
    for raw_log in get_accepted_raw_logs(state):
        corpus.append({
            "raw_log": raw_log,
            "embedding": embed_texts([raw_log])[0],
            "signature": extract_signature(raw_log),
        })
    print(f"Seeded corpus with {len(corpus)} existing examples.")
    
    # We only want merged PRs
    for repo in TARGET_REPOS:
        if collected_examples >= target_count:
            break

        page = 1
        retry_count = 0

        params = {
                "q": f'repo:{repo} is:pr is:merged',
                "per_page": 50,  
                "page": page,  
            }

        print(f'Initiating scan on {repo}:')

        while collected_examples < target_count:
            print(f"Fetching PR page {page} in repo {repo}")


            try:
                pr_response = requests.get(url=url, headers=headers, params=params, timeout=10)
            except requests.exceptions.RequestException as e:
                print(f"Network error fetching page {page}: {e}. Retrying in 10s...")
                time.sleep(10)
                continue
            
            if pr_response.status_code != 200:
                if pr_response.status_code in (403, 429):
                    retry_after = pr_response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else compute_backoff(retry_count)
                    print(f"Rate limited (status {pr_response.status_code}). Sleeping {wait:.1f}s.")
                    time.sleep(wait)
                    retry_count += 1
                    if retry_count > 6:
                        print("Too many retries on this page — moving on.")
                        break
                    continue
                print(f"GitHub API Error: {pr_response.text}")
                break
                
            retry_count = 0
            response_data = pr_response.json()
            prs = response_data.get("items", [])

            if not prs:
                print(f"   -> Exhausted search results for {repo}. Moving to next repository.")
                break
                
            for pr in prs:

                if "pull_request" not in pr:
                    print('No pull request in PR body.')
                    continue

                body = 'PULL REQUEST AND RELATED ISSUES:\n'  
                body += pr.get("body") or ""
                
                # Look for the exact issue link syntax (e.g., Fixes #12345, Resolves #12345)
                pattern = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)"
                matches = re.finditer(pattern, body, re.IGNORECASE)
                issue_numbers = {m.group(1) for m in matches}
                if not issue_numbers:
                    print('No matches')
                    continue

                for issue_number in issue_numbers:
                    key = f"{repo}#{issue_number}"

                    #check if the issue already exists in the data
                    if is_processed(state, key):
                        continue

                    # Fetch the original issue
                    issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

                    try:
                        issue_response = requests.get(issue_url, headers=headers, timeout=10)
                    except requests.exceptions.RequestException:
                        continue
                    
                    if issue_response.status_code != 200:
                        continue
                        
                    issue_data = issue_response.json()
                    if "pull_request" in issue_data:
                        mark_processed(state, key, "rejected_is_pr") #Record the decision
                        print(f"#{issue_number} is a PR, not an issue. Skipping.")
                        continue

                    issue_text = issue_data.get("body") or ""
                    body += f'#{issue_number}:\n{issue_text}\n'
                    time.sleep(2)
                    
                    # FILTER: Does the issue actually contain a stack trace or raw log?
                    if not contains_stack_trace_or_log(body):
                        mark_processed(state, key, "rejected_no_traceback")
                        print("The related issue in the PR doesn't contain any stack trace or error log.")
                        continue

                    # dedup check, on the pre-LLM body
                    is_dup, max_score = is_duplicate(body, corpus)
                    print(f"[DEDUP] {key} max_sim={max_score}")
                    if is_dup:
                        mark_processed(state, key, "rejected_duplicate")
                        continue
                        
                    # call LLM to format the data and generate the response for the dataset.
                    try:
                        log_telemetry = generate_synthetic_telemetry(body)
                    except QuotaExhaustedError as e:
                        print(f"Stopping run — {e}")
                        return

                    # defensive check against none type output
                    if not log_telemetry:
                        print(f"LLM failed to generate schema for PR: {pr.get('number', 'Unknown')}. Skipping.")
                        continue

                    rca = log_telemetry.get('rca_schema', {}).get('root_cause_analysis', '')
                    if rca.lower().replace(" ", "_") == 'insufficient_data':
                        mark_processed(state, key, "rejected_insufficient_data")
                        print(f"LLM rejected PR {pr.get('number', 'Unknown')} due to insufficient context. Skipping.")
                        continue
                    
                    #append data to respective train or test files
                    file_name = train_test_split(issue_url)
                    try:
                        append_to_jsonl(
                            file_name, 
                            log_telemetry['raw_log'],
                            log_telemetry['rca_schema']
                        )
                        mark_processed(state, key, "accepted", raw_log=log_telemetry['raw_log'], split=file_name)
                        corpus.append({
                            "raw_log": log_telemetry['raw_log'],
                            "embedding": embed_texts([log_telemetry['raw_log']])[0],
                            "signature": extract_signature(log_telemetry['raw_log']),
                        })
                        collected_examples += 1
                        print(f"Collected {collected_examples}/{target_count} examples.")

                    except KeyError as e:
                        print(f"Malformed LLM output missing key {e}. Skipping.")
                        continue

                    if collected_examples >= target_count:
                        print("Dataset collected.")
                        return

            if page >= 3:
                break                  
            page += 1
            params["page"] = page
            time.sleep(2)

fetch_real_rca_data(67)
