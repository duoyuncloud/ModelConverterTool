"""
main.py
FastAPI application entry point.
- App setup, CORS, error handling, API routing, health check.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from app.api import router
from app.config import settings, validate_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Model conversion tool",
    version=settings.version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    validate_config()
    logger.info("Startup complete")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs",
        "supported_formats": settings.supported_formats,
        "supported_model_types": settings.supported_model_types
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.version}

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 