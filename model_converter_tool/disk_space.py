"""
Disk space checking and user prompt utilities for model conversion CLI.
"""

import os
import shutil
from typing import Any, Optional, Dict, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import logging

logger = logging.getLogger(__name__)
console = Console()


def get_disk_usage(path: str = "/") -> Tuple[int, int, int]:
    """
    Get disk usage information for a path.
    Args:
        path: Path to check disk usage for
    Returns:
        Tuple of (total_bytes, used_bytes, free_bytes)
    """
    try:
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    except Exception as e:
        logger.warning(f"Could not get disk usage for {path}: {e}")
        return 0, 0, 0


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format with GiB/MiB units"""
    if bytes_value == 0:
        return "0B"
    size_names = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while bytes_value >= 1024 and i < len(size_names) - 1:
        bytes_value = bytes_value / 1024.0
        i += 1
    return f"{bytes_value:.1f}{size_names[i]}"


def check_disk_space_safety(
    required_bytes: int, safety_margin_gib: float = 5.0, path: str = "/"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if there's enough disk space with safety margin.
    Args:
        required_bytes: Required space in bytes
        safety_margin_gib: Safety margin in GiB (default: 5.0)
        path: Path to check disk usage for
    Returns:
        Tuple of (has_enough_space, info_dict)
    """
    total, used, free = get_disk_usage(path)
    safety_margin_bytes = int(safety_margin_gib * 1024**3)
    has_enough_for_operation = free >= required_bytes
    remaining_after_operation = free - required_bytes
    has_safety_margin = remaining_after_operation >= safety_margin_bytes
    info = {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "required_bytes": required_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "remaining_after_operation": remaining_after_operation,
        "has_enough_for_operation": has_enough_for_operation,
        "has_safety_margin": has_safety_margin,
        "formatted": {
            "total": format_bytes(total),
            "used": format_bytes(used),
            "free": format_bytes(free),
            "required": format_bytes(required_bytes),
            "safety_margin": format_bytes(safety_margin_bytes),
            "remaining_after": format_bytes(remaining_after_operation),
        },
    }
    return has_enough_for_operation and has_safety_margin, info


def estimate_model_size(model_path: str, output_format: str, quantization: Optional[str] = None) -> int:
    """
    Estimate the size of the output model in bytes.
    Args:
        model_path: Input model path or name
        output_format: Target output format
        quantization: Quantization type if applicable
    Returns:
        Estimated size in bytes
    """
    try:
        if os.path.exists(model_path):
            if os.path.isfile(model_path):
                base_size = os.path.getsize(model_path)
            else:
                base_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        if os.path.exists(file_path):
                            base_size += os.path.getsize(file_path)
        else:
            lower_name = model_path.lower()
            if "bert-base" in lower_name:
                base_size = 420 * 1024**2
            elif "llama-2-7b" in lower_name or "llama2-7b" in lower_name:
                base_size = 13 * 1024**3
            elif "llama-2-13b" in lower_name or "llama2-13b" in lower_name:
                base_size = 24 * 1024**3
            elif "llama-2-70b" in lower_name or "llama2-70b" in lower_name:
                base_size = 130 * 1024**3
            elif "gpt2" in lower_name:
                base_size = 500 * 1024**2
            elif "opt-125m" in lower_name:
                base_size = 250 * 1024**2
            elif "opt-350m" in lower_name:
                base_size = 700 * 1024**2
            elif "opt-1.3b" in lower_name:
                base_size = 2.5 * 1024**3
            elif "opt-6.7b" in lower_name:
                base_size = 13 * 1024**3
            else:
                base_size = 1 * 1024**3
        format_multipliers = {
            "onnx": 1.2,
            "gguf": 0.8,
            "torchscript": 1.0,
            "fp16": 0.5,
            "safetensors": 1.0,
            "mlx": 0.8,
        }
        quant_multipliers = {
            "4bit": 0.25,
            "8bit": 0.5,
            "q4_k_m": 0.25,
            "q8_0": 0.5,
            "q5_k_m": 0.31,
            "q4_0": 0.25,
            "q4_1": 0.25,
        }
        multiplier = format_multipliers.get(output_format, 1.0)
        if quantization and quantization in quant_multipliers:
            multiplier *= quant_multipliers[quantization]
        estimated_size = int(base_size * multiplier)
        conversion_buffer = int(estimated_size * 0.5)
        return estimated_size + conversion_buffer
    except Exception as e:
        logger.warning(f"Could not estimate model size: {e}")
        return 2 * 1024**3


def prompt_user_confirmation_low_space(space_info: Dict[str, Any]) -> bool:
    """
    Prompt user for confirmation when disk space is low.
    Args:
        space_info: Disk space information from check_disk_space_safety
    Returns:
        True if user confirms, False otherwise
    """
    formatted = space_info["formatted"]
    warning_text = Text()
    warning_text.append("⚠️  ", style="bold yellow")
    warning_text.append("LOW DISK SPACE WARNING\n\n", style="bold red")
    warning_text.append("Current disk space:\n", style="bold")
    warning_text.append(f"  Free: {formatted['free']}\n")
    warning_text.append(f"  Required: {formatted['required']}\n")
    warning_text.append(f"  Safety margin: {formatted['safety_margin']}\n")
    warning_text.append(f"  Remaining after operation: {formatted['remaining_after']}\n\n")
    warning_text.append("After this operation, you will have less than the recommended ", style="yellow")
    warning_text.append("5 GiB", style="bold yellow")
    warning_text.append(" safety margin.\n\n", style="yellow")
    warning_text.append("This could cause system instability or prevent other applications from working properly.\n\n")
    warning_text.append("Do you want to continue anyway? Type 'yes' to confirm: ", style="bold")
    panel = Panel(warning_text, title="[bold red]Disk Space Warning[/bold red]", border_style="red")
    console.print(panel)
    try:
        user_input = input().strip().lower()
        return user_input == "yes"
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return False


def check_and_handle_disk_space(
    model_path: str,
    output_format: str,
    quantization: Optional[str] = None,
    safety_margin_gib: float = 5.0,
    path: str = "/",
) -> bool:
    """
    Check disk space and handle low space scenarios.
    Args:
        model_path: Input model path or name
        output_format: Target output format
        quantization: Quantization type if applicable
        safety_margin_gib: Safety margin in GiB
        path: Path to check disk usage for
    Returns:
        True if operation should proceed, False if aborted
    """
    required_bytes = estimate_model_size(model_path, output_format, quantization)
    has_enough_space, space_info = check_disk_space_safety(required_bytes, safety_margin_gib, path)
    if has_enough_space:
        formatted = space_info["formatted"]
        console.print("[green]✓[/green] Sufficient disk space available:")
        console.print(
            f"  Free: {formatted['free']} | Required: {formatted['required']} | Safety margin: {formatted['safety_margin']}"
        )
        return True
    if space_info["has_enough_for_operation"] and not space_info["has_safety_margin"]:
        return prompt_user_confirmation_low_space(space_info)
    formatted = space_info["formatted"]
    error_text = Text()
    error_text.append("❌ ", style="bold red")
    error_text.append("INSUFFICIENT DISK SPACE\n\n", style="bold red")
    error_text.append("Current disk space:\n", style="bold")
    error_text.append(f"  Free: {formatted['free']}\n")
    error_text.append(f"  Required: {formatted['required']}\n")
    error_text.append(f"  Shortage: {format_bytes(required_bytes - space_info['free_bytes'])}\n\n")
    error_text.append("Please free up disk space and try again.", style="yellow")
    panel = Panel(error_text, title="[bold red]Disk Space Error[/bold red]", border_style="red")
    console.print(panel)
    return False
