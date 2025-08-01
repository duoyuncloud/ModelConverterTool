# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys

from megatron.training import dist_signal_handler
from megatron.core import Timers
from megatron.training.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()


def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    # Check if args is already initialized, if so, skip setting it
    if _GLOBAL_ARGS is None:
        _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
        set_args(args)
    else:
        print("[DEBUG] Args already initialized, skipping set_args")

    # Only build these if they haven't been built yet
    if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        _build_num_microbatches_calculator(args)
    else:
        print("[DEBUG] Num microbatches calculator already initialized, skipping")

    if build_tokenizer and _GLOBAL_TOKENIZER is None:
        _ = _build_tokenizer(args)
    elif build_tokenizer:
        print("[DEBUG] Tokenizer already initialized, skipping")

    if _GLOBAL_TENSORBOARD_WRITER is None:
        _set_tensorboard_writer(args)
    else:
        print("[DEBUG] Tensorboard writer already initialized, skipping")

    if _GLOBAL_WANDB_WRITER is None:
        _set_wandb_writer(args)
    else:
        print("[DEBUG] Wandb writer already initialized, skipping")

    if _GLOBAL_ONE_LOGGER is None:
        _set_one_logger(args)
    else:
        print("[DEBUG] One logger already initialized, skipping")

    if _GLOBAL_ADLR_AUTORESUME is None:
        _set_adlr_autoresume(args)
    else:
        print("[DEBUG] ADLR autoresume already initialized, skipping")

    if _GLOBAL_TIMERS is None:
        _set_timers(args)
    else:
        print("[DEBUG] Timers already initialized, skipping")

    if args.exit_signal_handler:
        _set_signal_handler()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, "num microbatches calculator")

        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)
    else:
        print("[DEBUG] Num microbatches calculator already exists, skipping")


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, "tokenizer")
        _GLOBAL_TOKENIZER = build_tokenizer(args)
    else:
        print("[DEBUG] Tokenizer already exists, skipping")
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    if _GLOBAL_TENSORBOARD_WRITER is None:
        _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

        if hasattr(args, "tensorboard_dir") and args.tensorboard_dir and args.rank == (args.world_size - 1):
            try:
                from torch.utils.tensorboard import SummaryWriter

                print("> setting tensorboard ...")
                _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                    log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
                )
            except ModuleNotFoundError:
                print(
                    "WARNING: TensorBoard writing requested but is not "
                    "available (are you using PyTorch 1.1.0 or later?), "
                    "no TensorBoard logs will be written.",
                    flush=True,
                )
    else:
        print("[DEBUG] Tensorboard writer already exists, skipping")


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    if _GLOBAL_WANDB_WRITER is None:
        _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, "wandb writer")
        if getattr(args, "wandb_project", "") and args.rank == (args.world_size - 1):
            if args.wandb_exp_name == "":
                raise ValueError("Please specify the wandb experiment name!")

            import wandb

            if args.wandb_save_dir:
                save_dir = args.wandb_save_dir
            else:
                # Defaults to the save dir.
                save_dir = os.path.join(args.save, "wandb")
            wandb_kwargs = {
                "dir": save_dir,
                "name": args.wandb_exp_name,
                "project": args.wandb_project,
                "config": vars(args),
            }
            os.makedirs(wandb_kwargs["dir"], exist_ok=True)
            wandb.init(**wandb_kwargs)
            _GLOBAL_WANDB_WRITER = wandb
    else:
        print("[DEBUG] Wandb writer already exists, skipping")


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    if _GLOBAL_ONE_LOGGER is None:
        _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, "one logger")

        if args.enable_one_logger and args.rank == (args.world_size - 1):
            try:
                from one_logger.core import OneLogger

                config = {
                    "project": args.one_logger_project,
                    "entity": args.one_logger_entity,
                    "name": args.one_logger_run_name,
                }
                one_logger = OneLogger(config=config)
                _GLOBAL_ONE_LOGGER = one_logger
            except BaseException:
                print(
                    "WARNING: one_logger package is required to enable e2e metrics "
                    "tracking. Try pip install "
                    "--index-url=https://sc-hw-artf.nvidia.com/api/pypi/hwinf-ml-pypi/simple"
                    " one_logger to install it"
                )
    else:
        print("[DEBUG] One logger already exists, skipping")


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    if _GLOBAL_ADLR_AUTORESUME is None:
        _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, "adlr autoresume")

        if args.adlr_autoresume:
            if args.rank == 0:
                print("enabling autoresume ...", flush=True)
            sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
            try:
                from userlib.auto_resume import AutoResume
            except BaseException:
                print("ADLR autoresume is not available, exiting ...")
                sys.exit()

            _GLOBAL_ADLR_AUTORESUME = AutoResume
    else:
        print("[DEBUG] ADLR autoresume already exists, skipping")


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    if _GLOBAL_TIMERS is None:
        _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
        _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)
    else:
        print("[DEBUG] Timers already exists, skipping")


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)
