# Model Converter Engine

This document provides technical details about the model conversion engine, supported formats, and the internal workflow.

## Overview
The conversion engine transforms machine learning models between different formats, applies quantization, and optimizes for various hardware backends.

- Supports muP-to-LLaMA scaling and config adaptation (`--mup2llama`)
- Supports fake weight generation for testing (`--fake-weight`, `--fake-weight-config`)

## Supported Formats
- GGUF
- ONNX
- FP16 (SafeTensors)
- AWQ
- MLX
- GPTQ
- Safetensors
- TorchScript
- HuggingFace (re-save)

## Supported Conversion Matrix

| Input Format | Supported Output Formats |
|--------------|-------------------------|
| huggingface  | huggingface, hf, safetensors, torchscript, onnx, gguf, mlx, gptq, awq, mtk, rk, ax, qnn |
| hf           | huggingface, hf, safetensors, torchscript, onnx, gguf, mlx, gptq, awq, mtk, rk, ax, qnn |
| megatron     | hf, megatron2hf         |
| safetensors  | huggingface, hf, safetensors |
| torchscript  | torchscript             |
| onnx         | onnx                    |
| gguf         | gguf                    |
| mlx          | mlx                     |

> **Note:** Formats such as `mtk`, `rk`, `ax`, `qnn`, and `megatron2hf` are planned and will be supported in future releases. If you try to use them now, you will see a clear "NotImplementedError" message.

---

## Conversion Workflow
1. Load the input model
2. Parse and validate model structure
3. Optionally apply quantization, muP scaling, or fake weight logic
4. Convert to the target format
5. Adapt and save the output config (remove muP params, add LLaMA params if needed)
6. Save the output model

## Engine Architecture
- Modular design for easy extension
- Pluggable backends for different formats
- Device-aware conversion (CPU, GPU, Apple Silicon)
- All special options (muP, fake weight, quantization) are handled in the main workflow for consistency

## Extending the Engine
- Add support for new model formats by implementing a new engine module in `model_converter_tool/engine/`
- Follow the KISS/DRY/YAGNI principles for maintainability

## References
- [Core API documentation](../model_converter_tool/api.py)
- [Engine source code](../model_converter_tool/engine/)
- [Test suite](../tests/)
- [Example scripts](../examples/) 