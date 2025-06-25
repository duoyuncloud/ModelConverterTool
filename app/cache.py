"""
cache.py
Unified cache and system monitoring (Redis-based).
- SystemMonitor: collects system metrics and alerts
- ModelCache/ConversionCache: model/result cache
- CacheManager: unified interface
- cached: decorator for API caching
"""

import redis
import json
import hashlib
import psutil
import threading
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from functools import wraps
from collections import deque
import os

# --- System Monitoring ---
class SystemMonitor:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.system_metrics_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=50)
        self.lock = threading.Lock()
        self.thresholds = {
            "cpu_warning": 80.0, "cpu_critical": 95.0,
            "memory_warning": 85.0, "memory_critical": 95.0,
            "disk_warning": 90.0, "disk_critical": 95.0
        }

    def collect_system_metrics(self) -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_usage_percent": disk.percent
            }
            with self.lock:
                self.system_metrics_history.append(metrics)
                self.check_alerts(metrics)
            return metrics
        except Exception as e:
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def check_alerts(self, metrics: Dict[str, Any]):
        alerts = []
        if metrics.get("cpu_percent", 0) >= self.thresholds["cpu_critical"]:
            alerts.append({"level": "CRITICAL", "metric": "CPU", "value": metrics["cpu_percent"]})
        elif metrics.get("cpu_percent", 0) >= self.thresholds["cpu_warning"]:
            alerts.append({"level": "WARNING", "metric": "CPU", "value": metrics["cpu_percent"]})
        if metrics.get("memory_percent", 0) >= self.thresholds["memory_critical"]:
            alerts.append({"level": "CRITICAL", "metric": "Memory", "value": metrics["memory_percent"]})
        elif metrics.get("memory_percent", 0) >= self.thresholds["memory_warning"]:
            alerts.append({"level": "WARNING", "metric": "Memory", "value": metrics["memory_percent"]})
        if metrics.get("disk_usage_percent", 0) >= self.thresholds["disk_critical"]:
            alerts.append({"level": "CRITICAL", "metric": "Disk", "value": metrics["disk_usage_percent"]})
        elif metrics.get("disk_usage_percent", 0) >= self.thresholds["disk_warning"]:
            alerts.append({"level": "WARNING", "metric": "Disk", "value": metrics["disk_usage_percent"]})
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            self.alerts_history.append(alert)

    def get_system_stats(self) -> Dict[str, Any]:
        with self.lock:
            recent_metrics = list(self.system_metrics_history)
        return {"metrics": recent_metrics, "alerts": list(self.alerts_history)}

# --- Model and Conversion Cache ---
class ModelCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.model_cache_ttl = 3600 * 24

    def cache_model_info(self, model_name: str, model_info: Dict[str, Any]):
        key = f"model_cache:{hashlib.md5(model_name.encode()).hexdigest()}"
        self.redis_client.setex(key, self.model_cache_ttl, json.dumps(model_info))

    def get_cached_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        key = f"model_cache:{hashlib.md5(model_name.encode()).hexdigest()}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

class ConversionCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.conversion_cache_ttl = 3600 * 24 * 7

    def cache_conversion_result(self, model_name: str, target_format: str, params: Dict[str, Any], output_path: str):
        key = f"conversion_cache:{hashlib.md5((model_name+target_format+json.dumps(params,sort_keys=True)).encode()).hexdigest()}"
        cache_data = {"model_name": model_name, "target_format": target_format, "params": params, "output_path": output_path}
        self.redis_client.setex(key, self.conversion_cache_ttl, json.dumps(cache_data))

    def get_cached_conversion(self, model_name: str, target_format: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = f"conversion_cache:{hashlib.md5((model_name+target_format+json.dumps(params,sort_keys=True)).encode()).hexdigest()}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

# --- Unified Cache Manager ---
class CacheManager:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.monitor = SystemMonitor()
        self.model_cache = ModelCache(redis_url)
        self.conversion_cache = ConversionCache(redis_url)

    def get_cache_stats(self) -> Dict[str, Any]:
        info = self.redis_client.info()
        return {
            "redis_info": {
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0)
            }
        }

    def clear_cache(self, prefix: Optional[str] = None):
        if prefix:
            pattern = f"{prefix}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        else:
            keys = self.redis_client.keys("*")
            if keys:
                self.redis_client.delete(*keys)

    def get_system_monitoring(self) -> Dict[str, Any]:
        return self.monitor.get_system_stats()

def cached(ttl: int = 3600, key_prefix: str = "cache"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            cached_result = CacheManager(redis_url).redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            result = func(*args, **kwargs)
            CacheManager(redis_url).redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
cache_manager = CacheManager(redis_url) 