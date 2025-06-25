# Model Converter Tool

A robust, extensible tool for converting, validating, and managing machine learning models across multiple formats. Built for efficiency, automation, and integration with modern ML workflows.

## Features
- **Real model conversion** between formats: ONNX, GPTQ, AWQ, GGUF, MLX, TorchScript, FP16, HuggingFace, and more
- Supports both HuggingFace model names and local file uploads
- Batch conversion and validation
- Real-time progress tracking via WebSocket
- Caching and model preloading for performance
- System and cache monitoring endpoints
- RESTful API with OpenAPI documentation
- Docker and Docker Compose support

## Technology Stack
- Python 3.9+
- FastAPI (REST API)
- Celery (task queue)
- Redis (cache, broker, progress tracking)
- HuggingFace Transformers
- ONNX, TensorFlow, CoreML, OpenVINO, and other conversion libraries

## Quick Start

### Prerequisites
- Python 3.9+
- Redis server (for caching and Celery broker)
- (Optional) Docker & Docker Compose

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis server (if not running)
redis-server

# Create required directories
mkdir -p uploads outputs

# Start FastAPI server
python -m uvicorn app.main:app --reload

# Start Celery worker (required for real conversion)
python -m celery -A app.tasks worker --loglevel=info

# Start Streamlit UI
python3 -m streamlit run streamlit_app.py
```

### Production (Recommended)
```bash
# Build and run with Docker Compose
# (Ensure Docker and Docker Compose are installed)
docker-compose up --build
```

## Usage

### 1. Convert HuggingFace Models
- Enter a valid HuggingFace model name (e.g., `gpt2`, `bert-base-uncased`, `meta-llama/Llama-2-7b-hf`).
- Select the target format and model type.
- Click "Start Conversion" to perform a real conversion (not a dummy/test file).

### 2. Convert Local Files
- Select "Local File" as the model source.
- Upload a HuggingFace-compatible model file or directory (preferably as a zip/tar.gz, or upload all required files).
- Select the target format and model type.
- Click "Start Conversion".

> **Note:** For best results, upload the full HuggingFace model directory (including config, weights, and tokenizer files).

### 3. Download Results
- After conversion, download the output from the UI or find it in the `outputs/` directory.

## API Usage
- Access API docs at: `http://localhost:8000/docs`
- Key endpoints:
  - `/convert` : Submit model conversion task
  - `/status/{task_id}` : Get task status
  - `/download/{task_id}` : Download converted model
  - `/validate/model` : Validate model
  - `/cache/stats` : Cache statistics
  - `/ws/{task_id}` : WebSocket for progress

## Environment Variables (see `env.example`)
- `REDIS_URL` : Redis connection string
- `UPLOAD_DIR`, `OUTPUT_DIR` : File storage paths
- `SUPPORTED_FORMATS`, `SUPPORTED_MODEL_TYPES` : Supported model formats/types
- `CACHE_TTL`, `MAX_CACHE_SIZE` : Cache settings
- `ONNX_OPSET_VERSION`, `GPTQ_BITS`, etc.: Conversion parameters

## Real Quantization Requirements (GPTQ/AWQ)
- **OS:** Linux (x86_64)
- **GPU:** CUDA-enabled NVIDIA GPU
- **Python Packages:**
  - `auto-gptq` for GPTQ quantization
  - `awq` for AWQ quantization

```bash
pip install auto-gptq awq
```

- These quantization methods will **not** work on macOS或CPU-only系统。
- 如果环境不满足，工具会给出清晰的报错。
- 你仍然可以在任何系统上将已量化模型（如 HuggingFace 上下载的）转换为其他格式。

## Troubleshooting
- 确保 FastAPI、Celery worker、Redis、Streamlit UI 均已启动。
- 检查 `outputs/` 目录，确认输出文件大小和内容是否为真实模型。
- 如遇报错，请查看终端日志或 UI 错误提示，并根据提示修复。

---

**Enjoy real, production-grade model conversion!**