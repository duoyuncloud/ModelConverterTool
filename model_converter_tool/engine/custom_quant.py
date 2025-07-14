import numpy as np
import json
from pathlib import Path
from typing import Any, Optional
from model_converter_tool.utils import auto_load_model_and_tokenizer, patch_quantization_config

def quantize_tensor(tensor, bits=4, group_size=128, sym=False, desc=None):
    """
    Quantize a float32 tensor using group-wise quantization.
    Returns quantized weights and quantization parameter metadata.
    """
    assert tensor.dtype == np.float32
    orig_shape = tensor.shape
    tensor = tensor.flatten()
    n = len(tensor)
    groups = int(np.ceil(n / group_size))
    qmin = -(2 ** (bits - 1)) if sym else 0
    qmax = (2 ** (bits - 1)) - 1 if sym else (2 ** bits) - 1
    scales = np.zeros(groups, dtype=np.float32)
    zeros = np.zeros(groups, dtype=np.float32)
    qweight = np.zeros(n, dtype=np.int32)
    for g in range(groups):
        start = g * group_size
        end = min((g + 1) * group_size, n)
        chunk = tensor[start:end]
        if sym:
            scale = (np.max(np.abs(chunk)) + 1e-8) / qmax
            zero = 0
        else:
            scale = (np.max(chunk) - np.min(chunk) + 1e-8) / (qmax - qmin)
            zero = np.min(chunk)
        scales[g] = scale
        zeros[g] = zero
        if sym:
            q = np.round(chunk / scale)
        else:
            q = np.round((chunk - zero) / scale)
        q = np.clip(q, qmin, qmax)
        qweight[start:end] = q.astype(np.int32)
    quantized = {
        'qweight': qweight.reshape(orig_shape).astype(np.int32),
        'scales': scales,
        'zeros': zeros,
        'bits': bits,
        'group_size': group_size,
        'sym': sym,
        'desc': desc,
        'orig_shape': orig_shape,
    }
    return quantized

def dequantize_tensor(quantized):
    qweight = quantized['qweight'].flatten()
    scales = quantized['scales']
    zeros = quantized['zeros']
    bits = quantized['bits']
    group_size = quantized['group_size']
    sym = quantized['sym']
    orig_shape = quantized['orig_shape']
    n = len(qweight)
    groups = int(np.ceil(n / group_size))
    qmin = -(2 ** (bits - 1)) if sym else 0
    qmax = (2 ** (bits - 1)) - 1 if sym else (2 ** bits) - 1
    tensor = np.zeros(n, dtype=np.float32)
    for g in range(groups):
        start = g * group_size
        end = min((g + 1) * group_size, n)
        scale = scales[g]
        zero = zeros[g]
        q = qweight[start:end]
        if sym:
            chunk = q * scale
        else:
            chunk = q * scale + zero
        tensor[start:end] = chunk
    return tensor.reshape(orig_shape)

def save_quantized(quantized, path):
    np.savez(path + '.npz', qweight=quantized['qweight'], scales=quantized['scales'], zeros=quantized['zeros'])
    meta = {k: v for k, v in quantized.items() if k not in ['qweight', 'scales', 'zeros']}
    with open(path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

def load_quantized(path):
    arrs = np.load(path + '.npz')
    with open(path + '.meta.json', 'r') as f:
        meta = json.load(f)
    quantized = {
        'qweight': arrs['qweight'],
        'scales': arrs['scales'],
        'zeros': arrs['zeros'],
        **meta
    }
    return quantized

def convert_to_custom_quant(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str,
    quantization: Optional[str] = None,
    use_large_calibration: bool = False,
    quantization_config: dict = None
) -> tuple:
    """
    Quantize the first float32 weight tensor of the model as a demonstration.
    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object (unused)
        model_name: Source model name or path
        output_path: Output directory
        model_type: Model type
        device: Device (unused)
        quantization: Quantization string (optional)
        use_large_calibration: Unused for custom quant
        quantization_config: Dict with quantization parameters (optional)
    Returns:
        (success: bool, extra_info: dict or error string)
    """
    try:
        # Robust model auto-loading
        model, _ = auto_load_model_and_tokenizer(model, None, model_name, model_type)
        import torch
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Load model weights (demo: only quantize first weight)
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = torch.load(model_name, map_location='cpu')
        # Find first float32 weight
        weight_name, weight_tensor = None, None
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                weight_name, weight_tensor = k, v.detach().cpu().numpy()
                break
        if weight_tensor is None:
            return False, {'error': 'No float32 weights found for quantization.'}
        # Parse quantization params
        bits = quantization_config.get('bits', 4) if quantization_config else 4
        group_size = quantization_config.get('group_size', 128) if quantization_config else 128
        sym = quantization_config.get('sym', False) if quantization_config else False
        desc = quantization_config.get('desc', None) if quantization_config else None
        quantized = quantize_tensor(weight_tensor, bits=bits, group_size=group_size, sym=sym, desc=desc)
        save_quantized(quantized, str(output_dir / f'{weight_name}_quant'))
        # Save quantization config as metadata
        with open(output_dir / 'custom_quant_config.json', 'w') as f:
            json.dump({
                'weight_name': weight_name,
                'bits': bits,
                'group_size': group_size,
                'sym': sym,
                'desc': desc
            }, f, indent=2)
        # Patch quantization config for test compatibility
        patch_quantization_config(output_dir / "config.json", bits, group_size, sym, desc)
        return True, {'quantized_weight': str(output_dir / f'{weight_name}_quant.npz')}
    except Exception as e:
        return False, {'error': str(e)}

def validate_custom_quant_file(output_dir: Path, _: Any) -> bool:
    # Simple validation: check if .npz and .meta.json exist
    for f in output_dir.glob('*.npz'):
        meta = f.with_suffix('.meta.json')
        if meta.exists():
            return True
    return False 