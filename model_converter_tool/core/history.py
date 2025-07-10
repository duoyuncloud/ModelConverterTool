import os
import json
from pathlib import Path
from typing import Dict, List, Any
import threading

HISTORY_FILE = os.path.expanduser("~/.model_converter_tool_history.json")
HISTORY_LOCK = threading.Lock()

# Helper to read the history file
def _read_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# Helper to write a new record to the history file
# record: dict with keys: model_path, output_format, output_path, status, timestamp, error (optional)
def append_history_record(record: Dict[str, Any]):
    with HISTORY_LOCK:
        history = _read_history()
        history.append(record)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

# Main function for CLI
# Returns: {"completed": [...], "failed": [...], "active": [...]} (lists of str or dict)
def get_history() -> Dict[str, List[Any]]:
    history = _read_history()
    completed = []
    failed = []
    active = []
    for entry in history:
        status = entry.get("status")
        summary = (
            f"{entry.get('model_path', '?')} -> {entry.get('output_format', '?')} | "
            f"Output: {entry.get('output_path', '?')} | Time: {entry.get('timestamp', '?')}"
        )
        if status == "completed":
            completed.append(summary)
        elif status == "failed":
            summary += f" | Error: {entry.get('error', '')}"
            failed.append(summary)
        elif status == "active":
            active.append(summary)
    return {"completed": completed, "failed": failed, "active": active} 