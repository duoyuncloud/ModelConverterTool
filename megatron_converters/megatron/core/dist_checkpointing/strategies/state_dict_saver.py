# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" State dict saver for PyT Distributed format allowing asynchronous save. """

from logging import getLogger
from time import time
from typing import TYPE_CHECKING, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.utils import _DistWrapper, _get_failure_dict

if TYPE_CHECKING:
    from .filesystem_async import FileSystemWriterAsync


logger = getLogger(__name__)


def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: "FileSystemWriterAsync",
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[SavePlanner] = None,
) -> Tuple["FileSystemWriterAsync", Metadata, _DistWrapper]:
    """
    First stage of saving a state dict to storage.

    This is an async adjustment of torch.distributed.checkpoint.state_dict_saver.
    In order to support async save, saving should be split into three parts:
    1. Planning
    2. Actual saving
    3. Finalization

    Out of these, step (2) *must* happen asynchronously.
    The first step is realized with this function.

    The planning part consists of several steps, described here:
    https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner

    Args:
        state_dict (STATE_DICT_TYPE): state dict to save
        storage_writer (FileSystemWriterAsync): in current version only an instance of
            FileSystemWriterAsync
        process_group (dist.ProcessGroup, optional): process group used for save planning
        coordinator_rank (int, optional): coordinator rank for planning. Defaults to 0.
        planner (SavePlanner, optional): save planner for torch.distributed.checkpoint format

    Returns: Tuple of:
        - storage writer (the one passed as input)
        - metadata from planning
        - distributed wrapper used for planning
    The return value of this function should be passed as an input to
    `save_state_dict_async_finalize`.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    dist_wrapper = _DistWrapper(process_group, True, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None

    def local_step():
        assert planner is not None
        planner.set_up_planner(state_dict, dist_wrapper.is_coordinator)
        storage_writer.set_up_storage_writer(dist_wrapper.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metadata

        assert planner is not None
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    # Execute local and global planning
    start_plan = time()
    central_plan = dist_wrapper.reduce_scatter("plan", local_step, global_step)
    logger.debug(f"rank: {rank}, plan time: {time() - start_plan}")

    # Prepare async writing of tensors.
    # The `storage_writer` will store the information about tensors it needs to save
    start = time()
    final_local_plan = planner.finish_plan(central_plan)
    storage_writer.prepare_write_data(final_local_plan, planner)
    end = time()
    logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")
    return storage_writer, cast(Metadata, global_metadata), dist_wrapper


def save_state_dict_async_finalize(
    storage_writer: "FileSystemWriterAsync",
    global_metadata: Metadata,
    dist_wrapper: _DistWrapper,
) -> None:
    """
    Finalization of save_state_dict_async_plan.

    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer.

    Args:
        storage_writer (FileSystemWriterAsync): storage writer used for planning
        global_metadata (Metadata): metadata created during planning
        dist_wrapper (_DistWrapper): distributed wrapper created during planning

    Returns: None
    """
    write_results = storage_writer.retrieve_write_results()

    # Gather the write results that will be saved to the metadata file.
    gather_start = time()
    all_results = dist_wrapper.gather_object(write_results)
    gather_end = time()
    logger.debug(f"{gather_end}, {torch.distributed.get_rank()}, gather: {gather_end-gather_start}")

    # Store the metadata on coordinator rank
    if dist_wrapper.is_coordinator:
        node_failures = _get_failure_dict(all_results)
        if len(node_failures) == 0:
            assert global_metadata is not None
            write_start = time()
            storage_writer.finish(global_metadata, all_results)
            write_end = time()
            logger.debug(f"{write_end}, metadata_write: {write_end - write_start}")
        else:
            raise CheckpointException("write", node_failures)
