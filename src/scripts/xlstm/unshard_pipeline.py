#!/usr/bin/env python3
"""
Unshard a pipeline-parallel checkpoint into a single-file state dict.

This must be launched with ``torchrun --nproc-per-node=<pp_degree>`` so that each
pipeline stage loads its own shard. Rank 0 merges the per-stage states, fixes the
block indices, and writes a regular PyTorch checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from olmo_core.train.train_module.transformer.config import TransformerPipelineParallelConfig
from olmo_core.distributed.parallel import PipelineScheduleType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.xlstm_large.model import xLSTMLargeConfig
from olmo_core.utils import prepare_cli_environment
from scripts.xlstm.inference import (  # type: ignore
    _build_stage_ranges,
    build_pipeline_stage_model,
    load_config,
    resolve_checkpoint_paths,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unshard a pipeline-parallel checkpoint into a single-file state dict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the original checkpoint directory (step directory or model_and_optim).",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Where to write the merged PyTorch checkpoint (e.g. /path/to/model.pt).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype to load weights with before saving.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def initialize_distributed() -> None:
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _offset_block_key(key: str, offset: int) -> str:
    """Replace the first backbone.blocks.<idx> occurrence with a global offset."""
    slug = "backbone.blocks."
    if slug not in key:
        return key
    before, after = key.split(slug, 1)
    if "." not in after:
        return key
    local_idx_str, rest = after.split(".", 1)
    try:
        local_idx = int(local_idx_str)
    except ValueError:
        return key
    return f"{before}{slug}{offset + local_idx}.{rest}"


def main() -> None:
    args = parse_args()
    prepare_cli_environment()
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config_dir, model_dir = resolve_checkpoint_paths(args.checkpoint_dir, None)
    config_dict = load_config(config_dir)

    train_module_cfg = config_dict.get("train_module", {})
    pp_cfg_dict = train_module_cfg.get("pp_config") or {}
    pp_degree = pp_cfg_dict.get("degree", 1)
    if pp_degree <= 1:
        raise RuntimeError("Checkpoint is not pipeline-parallel; use scripts/unshard.py instead.")
    if world_size != pp_degree:
        raise RuntimeError(
            f"World size ({world_size}) must equal pipeline degree ({pp_degree}). "
            "Launch with torchrun --nproc-per-node=<pp_degree> ..."
        )

    schedule_val = pp_cfg_dict.get("schedule", PipelineScheduleType.interleaved_1F1B)
    if isinstance(schedule_val, str):
        schedule_val = PipelineScheduleType(schedule_val)
    pp_config = TransformerPipelineParallelConfig(
        degree=pp_degree,
        schedule=schedule_val,
        split_points=pp_cfg_dict.get("split_points"),
    )

    model_cfg = xLSTMLargeConfig.from_dict(config_dict["model"])
    model_cfg.mode = "inference"
    model_cfg.return_last_states = True

    base_model = model_cfg.build(init_device="meta")
    total_blocks = len(base_model.backbone.blocks)
    split_points = pp_config.get_split_points(total_blocks)
    num_stages = len(split_points) + 1
    stage_ranges = _build_stage_ranges(split_points, total_blocks)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    stage_ids = pp_config.stage_ids_this_rank(rank, num_stages)
    local_state: Dict[str, Any] = {}
    for stage_idx in stage_ids:
        is_first = stage_idx == 0
        is_last = stage_idx == num_stages - 1
        start_offset, _ = stage_ranges[stage_idx]

        log.info(
            "Rank %d loading stage %d/%d (blocks %s) from '%s'",
            rank,
            stage_idx,
            num_stages - 1,
            stage_ranges[stage_idx],
            model_dir,
        )

        stage_model = build_pipeline_stage_model(
            base_model,
            stage_idx=stage_idx,
            num_stages=num_stages,
            split_points=split_points,
            device=device,
            dtype=dtype,
        )

        load_model_and_optim_state(
            str(model_dir),
            stage_model,
            strict=False,
            flatten_optimizer_state=True,
            process_group=dist.group.WORLD,
        )

        for key, tensor in stage_model.state_dict().items():
            if not is_first and key.startswith("embedding."):
                continue
            if not is_last and (key.startswith("lm_head.") or key.startswith("backbone.out_norm")):
                continue
            new_key = _offset_block_key(key, start_offset)
            local_state[new_key] = tensor.cpu()

    gathered: list[Dict[str, Any]] | None = None
    if rank == 0:
        gathered = [None for _ in range(world_size)]  # type: ignore
    dist.gather_object(local_state, gathered, dst=0)

    if rank == 0:
        merged: Dict[str, Any] = {}
        assert gathered is not None
        for state in gathered:
            assert isinstance(state, dict)
            for k, v in state.items():
                if k in merged:
                    raise RuntimeError(f"Duplicate key during merge: {k}")
                merged[k] = v
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": merged}, output_path)
        log.info("Wrote merged checkpoint to '%s' (%d tensors)", output_path, len(merged))

    dist.barrier()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
