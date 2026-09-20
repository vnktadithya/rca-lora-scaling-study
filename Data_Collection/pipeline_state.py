import tempfile
import os
import json
from pathlib import Path

def load_state(path="./Data/pipeline_state.json") -> dict:
    # Returns the current state. If the file doesn't exist yet, 
    # returns an empty skeleton
    file_path = Path(path)

    if file_path.is_file():
        content = file_path.read_text()
        return json.loads(content)

    return {"processed_issues": {}}

def save_state(state: dict, path="./Data/pipeline_state.json") -> None:
    # Writes state to disk atomically
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name)
    with os.fdopen(fd, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)

def is_processed(state: dict, key: str) -> bool:
    # True if this issue key has a terminal decision recorded already
    return key in state["processed_issues"]

def mark_processed(state: dict, key: str, status: str, raw_log: str = None, split: str = None) -> None:
    # Records the decision for this issue and persists immediately
    if not is_processed(state, key):
        state["processed_issues"][key] ={"status": status, "split": split, "raw_log": raw_log}
        save_state(state)

def get_accepted_raw_logs(state: dict) -> list[str]:
    # Return every raw_log for issues whose status is 'accepted' — 
    # this is what rebuilds the dedup corpus after a restart
    accepted_raw_logs = []
    for key, value in state["processed_issues"].items():
        if value["status"] == "accepted":
            accepted_raw_logs.append(value["raw_log"])
    return accepted_raw_logs
