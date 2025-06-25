#!/usr/bin/env python3
"""
Model Converter Tool startup script
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

def check_dependencies():
    """Check required dependencies."""
    required_packages = ["fastapi", "uvicorn", "celery", "redis", "torch", "transformers"]
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    print("All required packages are installed.")
    return True

def check_redis():
    """Check if Redis service is running."""
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        parsed = urlparse(redis_url)
        r = redis.Redis(host=parsed.hostname, port=parsed.port, db=int(parsed.path.lstrip('/')))
        r.ping()
        print("Redis service is running.")
        return True
    except Exception as e:
        print(f"Redis service is not running: {e}")
        print("Please start Redis: redis-server")
        return False

def create_directories():
    """Create required directories."""
    directories = ["uploads", "outputs", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Directory structure created.")

def start_development_mode():
    """Start in development mode."""
    print("Starting in development mode...")
    if not check_dependencies():
        return False
    create_directories()
    if not check_redis():
        return False
    print("Development mode started.")
    print("API docs: http://localhost:8000/docs")
    print("Please manually start the following services:")
    print("   1. Celery Worker: python -m celery -A app.tasks worker --loglevel=info")
    print("   2. FastAPI Server: python -m uvicorn app.main:app --reload")
    return True

def start_production_mode():
    """Start in production mode."""
    print("Starting in production mode...")
    if not check_dependencies():
        return False
    create_directories()
    if not check_redis():
        return False
    print("Production mode started.")
    print("API docs: http://localhost:8000/docs")
    print("Please manually start the following services:")
    print("   1. Celery Worker: python -m celery -A app.tasks worker --loglevel=info --concurrency=4")
    print("   2. FastAPI Server: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Converter Tool startup script")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev",
                       help="Startup mode: dev (development) or prod (production)")
    args = parser.parse_args()
    print("=" * 50)
    print("Model Converter Tool")
    print("=" * 50)
    if args.mode == "dev":
        start_development_mode()
    elif args.mode == "prod":
        start_production_mode()
    print("=" * 50)

if __name__ == "__main__":
    main() 