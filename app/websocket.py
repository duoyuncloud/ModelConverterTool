"""
websocket.py
Progress tracking and WebSocket communication.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any, Callable
import json
import asyncio
import logging
from datetime import datetime
import redis
import os

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks task progress and status in Redis."""
    def __init__(self, redis_url: str = None):
        if redis_url is None:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(redis_url)
        self.expire_time = 3600

    def update_progress(self, task_id: str, progress: int, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        progress_data = {
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        key = f"progress:{task_id}"
        self.redis_client.setex(key, self.expire_time, json.dumps(progress_data))
        self.redis_client.publish(f"progress_updates:{task_id}", json.dumps(progress_data))

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        key = f"progress:{task_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

    def set_task_info(self, task_id: str, model_name: str, target_format: str, model_type: str, conversion_params: Dict[str, Any]) -> None:
        task_info = {
            "task_id": task_id,
            "model_name": model_name,
            "target_format": target_format,
            "model_type": model_type,
            "conversion_params": conversion_params,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        key = f"task_info:{task_id}"
        self.redis_client.setex(key, self.expire_time, json.dumps(task_info))

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        key = f"task_info:{task_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

    def set_task_status(self, task_id: str, status: str, result: Optional[str] = None) -> None:
        task_info = self.get_task_info(task_id)
        if task_info:
            task_info["status"] = status
            task_info["result"] = result
            task_info["completed_at"] = datetime.now().isoformat()
            key = f"task_info:{task_id}"
            self.redis_client.setex(key, self.expire_time, json.dumps(task_info))

    def cleanup_task(self, task_id: str) -> None:
        for key in [f"progress:{task_id}", f"task_info:{task_id}"]:
            self.redis_client.delete(key)

    def set_task_error_log(self, task_id: str, error_message: str) -> None:
        key = f"error_log:{task_id}"
        self.redis_client.setex(key, self.expire_time, error_message)

    def get_task_error_log(self, task_id: str) -> Optional[str]:
        key = f"error_log:{task_id}"
        data = self.redis_client.get(key)
        return data.decode() if data else None

    def add_progress_update(self, task_id: str, progress: int, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        key = f"progress_timeline:{task_id}"
        entry = {
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.redis_client.rpush(key, json.dumps(entry))
        self.redis_client.expire(key, self.expire_time)

    def get_progress_timeline(self, task_id: str) -> list:
        key = f"progress_timeline:{task_id}"
        entries = self.redis_client.lrange(key, 0, -1)
        return [json.loads(e) for e in entries] if entries else []

    def get_full_task_history(self, task_id: str) -> dict:
        return {
            "task_info": self.get_task_info(task_id),
            "progress_timeline": self.get_progress_timeline(task_id),
            "error_log": self.get_task_error_log(task_id)
        }

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.task_subscriptions: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, task_id: Optional[str] = None) -> int:
        await websocket.accept()
        connection_id = id(websocket)
        if task_id:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            self.active_connections[task_id].append(websocket)
            if connection_id not in self.task_subscriptions:
                self.task_subscriptions[connection_id] = []
            self.task_subscriptions[connection_id].append(task_id)
        logger.info(f"WebSocket connected: {connection_id}, task: {task_id}")
        return connection_id

    def disconnect(self, websocket: WebSocket) -> None:
        connection_id = id(websocket)
        if connection_id in self.task_subscriptions:
            for task_id in self.task_subscriptions[connection_id]:
                if task_id in self.active_connections:
                    self.active_connections[task_id] = [
                        conn for conn in self.active_connections[task_id] 
                        if conn != websocket
                    ]
                    if not self.active_connections[task_id]:
                        del self.active_connections[task_id]
            del self.task_subscriptions[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_progress_update(self, task_id: str, progress_data: dict) -> None:
        if task_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[task_id]:
                try:
                    await websocket.send_text(json.dumps(progress_data))
                except Exception as e:
                    logger.error(f"Send progress update failed: {e}")
                    disconnected.append(websocket)
            for websocket in disconnected:
                self.disconnect(websocket)

    async def broadcast(self, message: dict) -> None:
        for task_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Broadcast message failed: {e}")
                    self.disconnect(websocket)

progress_tracker = ProgressTracker()
manager = ConnectionManager()

def progress_callback(task_id: str) -> Callable[[int, str, Optional[Dict[str, Any]]], None]:
    def callback(progress: int, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        progress_tracker.update_progress(task_id, progress, message, details)
    return callback

async def websocket_endpoint(websocket: WebSocket, task_id: Optional[str] = None) -> None:
    connection_id = await manager.connect(websocket, task_id)
    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "connection_id": connection_id,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }))
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "subscribe" and message.get("task_id"):
                    new_task_id = message["task_id"]
                    if new_task_id not in manager.active_connections:
                        manager.active_connections[new_task_id] = []
                    manager.active_connections[new_task_id].append(websocket)
                    if connection_id not in manager.task_subscriptions:
                        manager.task_subscriptions[connection_id] = []
                    manager.task_subscriptions[connection_id].append(new_task_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "task_id": new_task_id,
                        "timestamp": datetime.now().isoformat()
                    }))
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    finally:
        manager.disconnect(websocket) 