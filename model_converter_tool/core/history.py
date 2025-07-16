import os
import json
from pathlib import Path
from typing import Dict, List, Any
import threading

HISTORY_FILE = os.path.expanduser("~/.model_converter_tool_history.json")
HISTORY_LOCK = threading.Lock()

def _read_history() -> List[Dict[str, Any]]:
    """
    Read the history file and return a list of history records.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def append_history_record(record: Dict[str, Any]):
    """
    Append a new record to the history file.
    """
    with HISTORY_LOCK:
        history = _read_history()
        history.append(record)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

def get_history() -> Dict[str, List[Any]]:
    """
    Get the conversion history, separated into completed, failed, and active tasks.
    Returns a dictionary with keys: completed, failed, active.
    """
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