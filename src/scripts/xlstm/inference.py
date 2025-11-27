#!/usr/bin/env python3
"""
Run autoregressive inference for xLSTM-large checkpoints trained with ``scripts/xlstm/train.py``.

This script loads a checkpoint directory (unsharded), restores the model weights, then repeatedly
prefills the model with chunks from a ``data.bin`` file before streaming decoded tokens.

Example::

    # Single-GPU inference
    uv run src/scripts/xlstm/inference.py \
        ~/ckpts/unshard/model.pt \
        --data-bin ~/datasets/SYNTH/test_rank_00_0000.bin \
        --tokenizer ~/OLMo-core/tokenizer \
        --prefill-len 8192 \
        --max-new-tokens 1024 \
        --num-chunks 1 \
        --offset 13999 \
        --stream \
        --device cuda:0 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import traceback
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import torch

from olmo_core.nn.xlstm_large.generate import get_sampling_fn
from olmo_core.generate.sampling import select_next_token
from olmo_core.nn.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="xLSTM inference with streaming output from pre-recorded tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the unsharded model checkpoint file (.pt or .safetensors).",
    )
    parser.add_argument(
        "--data-bin",
        type=str,
        required=True,
        help="Binary file of uint tokens (matching the tokenizer used during training).",
    )
    parser.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        help="Numpy dtype of the binary data file.",
    )
    parser.add_argument(
        "--prefill-len",
        type=int,
        default=2048,
        help="Number of tokens to use from the data file for each prefill window.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of new tokens to sample after each prefill window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between prefill windows. Defaults to --prefill-len.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting token offset within the binary file.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="How many prefill windows to process. Set to 0 to iterate until EOF.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="greedy",
        choices=["greedy", "temperature"],
        help="Sampling strategy to use for decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run on (default picks CUDA if available).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit each generated token immediately instead of buffering a chunk.",
    )
    parser.add_argument(
        "--keep-state",
        action="store_true",
        help="Carry inference state across chunks (treat chunks as a single long sequence).",
    )
    parser.add_argument(
        "--bos-token-id",
        type=int,
        default=None,
        help="Optional BOS token to prepend when a chunk would otherwise start empty.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the model in float16 (may cause NaN issues). Default is float32 for stability.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--orig-ckpt-dir",
        type=str,
        help="Optional path to the original training checkpoint directory (used for config files).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional HuggingFace tokenizer identifier/path. Defaults to the training config tokenizer identifier if present.",
    )
    parser.add_argument(
        "--debug-logits",
        action="store_true",
        help="Print top-k logits for the final prefill position and first decode step.",
    )
    parser.add_argument(
        "--debug-top-k",
        type=int,
        default=5,
        help="Top-k tokens to display when --debug-logits is set.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 0.7).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k tokens to sample from (default: -1).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p tokens to sample from (default: 1.0).",
    )
    return parser.parse_args()


def resolve_config_path(checkpoint_path: str, config_override: Optional[str]) -> Path:
    """
    Resolve the path to config.json.
    """
    ckpt = Path(checkpoint_path).expanduser().resolve()
    
    if config_override:
        config_dir = Path(config_override).expanduser().resolve()
    else:
        # Try to find config in the same directory or parent
        if ckpt.is_file():
            config_dir = ckpt.parent
        else:
            config_dir = ckpt

    if config_dir.is_file():
        raise FileNotFoundError(f"Invalid config directory '{config_dir}'")

    config_path = config_dir / "config.json"
    if not config_path.is_file():
        # One fallback: if we are in unshard dir, maybe check parent
        if config_dir.name == "unshard" and (config_dir.parent / "config.json").is_file():
             return config_dir.parent / "config.json"
        
        raise FileNotFoundError(f"Missing config.json in '{config_dir}' or parent")

    return config_path


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return json.load(f)


def _load_tokenizer(args: argparse.Namespace, config_dict: Dict[str, Any]):
    identifier = args.tokenizer
    if identifier is None:
        identifier = (
            config_dict.get("train_module", {})
            .get("tokenizer", {})
            .get("identifier")
        )
    if identifier is None:
        log.warning("Tokenizer identifier not provided; tokens will be printed as ids.")
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.warning(
            "transformers not installed; cannot load tokenizer '%s'. Install transformers to decode tokens.",
            identifier,
        )
        return None

    try:
        tok = AutoTokenizer.from_pretrained(identifier, trust_remote_code=True)
        log.info("Loaded tokenizer '%s' for decoding output.", identifier)
        return tok
    except Exception as exc:
        log.warning("Failed to load tokenizer '%s': %s. Falling back to token ids.", identifier, exc)
        return None


def _load_local_state_file(
    model: xLSTMLarge,
    state_path: Path,
    *,
    strict: bool,
) -> None:
    log.info("Loading local state dict from '%s'", state_path)
    if state_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except ImportError as exc:
            raise ImportError(
                "Please install safetensors to load checkpoints saved in safetensors format."
            ) from exc
        state_obj: Any = safetensors_load_file(str(state_path))
    else:
        state_obj = torch.load(state_path, map_location="cpu")

    if isinstance(state_obj, dict) and "model" in state_obj:
        state_to_load: Any = state_obj["model"]
    else:
        state_to_load = state_obj

    if not isinstance(state_to_load, Mapping):
        raise TypeError(
            f"Checkpoint at '{state_path}' did not contain a mapping-compatible state dict."
        )

    if any("_orig_mod" in key for key in state_to_load.keys()):
        state_to_load = {
            _sanitize_compiled_key(key): value for key, value in state_to_load.items()
        }

    missing, unexpected = model.load_state_dict(state_to_load, strict=strict)
    if missing:
        log.warning("Missing keys during load: %s", missing)
    if unexpected:
        log.warning("Unexpected keys during load: %s", unexpected)


def _sanitize_compiled_key(key: str) -> str:
    new_key = key
    replacements = [
        ("._orig_mod.", "."),
        ("_orig_mod.", ""),
        ("._orig_mod", ""),
        ("_orig_mod", ""),
    ]
    for old, new in replacements:
        new_key = new_key.replace(old, new)
    while ".." in new_key:
        new_key = new_key.replace("..", ".")
    if new_key.startswith("."):
        new_key = new_key[1:]
    if new_key.endswith("."):
        new_key = new_key[:-1]
    return new_key


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(t) for t in token_ids)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        log.warning("Tokenizer decode failed: %s; falling back to ids", exc)
        return " ".join(str(t) for t in token_ids)


def _print_topk(tag: str, logits: torch.Tensor, k: int, tokenizer: Any = None) -> None:
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    scores, idx = torch.topk(logits, k, dim=-1)
    tokens = idx[0].tolist()
    scores_list = scores[0].tolist()
    decoded = _decode_tokens(tokenizer, tokens)
    log.info(
        "[%s] topk tokens=%s decoded=%s scores=%s",
        tag,
        tokens,
        decoded,
        [float(s) for s in scores_list],
    )


def load_data(path: str, dtype: str) -> np.memmap:
    np_dtype = np.dtype(dtype)
    return np.memmap(path, dtype=np_dtype, mode="r")


def iter_prefill_chunks(
    data: np.memmap,
    chunk_len: int,
    stride: int,
    offset: int,
    limit: Optional[int],
) -> Iterator[Tuple[int, int, np.ndarray]]:
    idx = offset
    chunk_id = 0
    while idx + chunk_len <= len(data):
        yield chunk_id, idx, np.asarray(data[idx : idx + chunk_len])
        chunk_id += 1
        if limit and chunk_id >= limit:
            break
        idx += stride


def stream_generate(
    model: xLSTMLarge,
    prefill_tokens: torch.Tensor,
    max_new_tokens: int,
    state: Optional[Dict] = None,
    callback: Optional[Callable[[int, int], None]] = None,
    debug: bool = False,
    debug_top_k: int = 5,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    tokenizer: Any = None,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    device = prefill_tokens.device

    logits, state = model(prefill_tokens, state)
    # Sync to ensure prefill completes before decode loop
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if debug:
        _print_topk("prefill_last", logits[:, -1:], debug_top_k, tokenizer=tokenizer)

    # Use contiguous tensor to avoid potential issues with strided memory access
    last_token = prefill_tokens[:, -1:].contiguous()
    generated: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        logits, state = model(last_token, state)
        if debug:
            _print_topk("decode_step0", logits, debug_top_k, tokenizer=tokenizer)
        next_token = select_next_token(
            logits=logits.squeeze(1),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).unsqueeze(-1).contiguous()
        generated.append(next_token)
        if callback is not None:
            callback(step, int(next_token[0, 0].item()))
        last_token = next_token

    return torch.cat(generated, dim=1), state


def run_single_device_inference(
    *,
    args: argparse.Namespace,
    model: xLSTMLarge,
    temperature: float,
    top_k: int,
    top_p: float,
    data: np.memmap,
    device: torch.device,
    tokenizer: Any,
):
    stride = args.stride or args.prefill_len
    state: Optional[Dict] = None
    chunk_iter = iter_prefill_chunks(
        data,
        chunk_len=args.prefill_len,
        stride=stride,
        offset=args.offset,
        limit=args.num_chunks if args.num_chunks != 0 else None,
    )

    for chunk_id, start_idx, chunk in chunk_iter:
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64, copy=False))
        if chunk_tensor.numel() == 0 and args.bos_token_id is not None:
            chunk_tensor = torch.tensor([args.bos_token_id], dtype=torch.long)
        elif chunk_tensor.numel() == 0:
            log.warning("Skipping empty chunk at offset %d", start_idx)
            continue

        prefill = chunk_tensor.unsqueeze(0).to(device=device, dtype=torch.long)
        prefill_text = _decode_tokens(tokenizer, chunk_tensor.tolist())
        marker = "<|GEN|>"

        print(f"[chunk {chunk_id}] prefill: {prefill_text} {marker}", end="", flush=True)

        generated, state = stream_generate(
            model,
            prefill_tokens=prefill,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sampling == "temperature",
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            state=state,
            callback=lambda step, token: print(f"{_decode_tokens(tokenizer, [token])}", end="", flush=True),
            debug=args.debug_logits,
            debug_top_k=args.debug_top_k,
            tokenizer=tokenizer,
        )

        if not args.keep_state:
            state = None


def main():
    try:
        args = parse_args()
        prepare_cli_environment()
        log.setLevel(logging.DEBUG if args.verbose else logging.WARNING)

        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            torch.cuda.set_device(device)

        config_path = resolve_config_path(args.checkpoint_path, args.orig_ckpt_dir)
        config_dict = load_config(config_path)

        tokenizer = _load_tokenizer(args, config_dict)
        
        model_cfg = xLSTMLargeConfig.from_dict(config_dict["model"])
        model_cfg.mode = "inference"
        model_cfg.step_kernel = "native"
        model_cfg.sequence_kernel = "native_sequence__native"
        model_cfg.chunkwise_kernel = "chunkwise--native_autograd"
        model_cfg.return_last_states = True

        # Note: bfloat16 causes NaN issues with the mLSTM Triton kernels in inference mode.
        # Use float32 for stable inference, or float16 if explicitly requested.
        if args.fp16:
            dtype = torch.float16
            log.warning("Using float16 - may cause numerical instability with mLSTM kernels")
        else:
            dtype = torch.float32
            log.info("Using float32 for stable inference")
        
        # Build model
        model = model_cfg.build()
        model = model.to(device=device, dtype=dtype)
        model.eval()

        # Load weights
        _load_local_state_file(model, Path(args.checkpoint_path), strict=True)

        data = load_data(args.data_bin, args.data_dtype)
        total_available = len(data) - args.offset
        if total_available < args.prefill_len:
            raise ValueError(
                f"Requested prefill length {args.prefill_len} exceeds available tokens ({total_available})."
            )

        log.info(
            "Starting inference: device=%s chunks=%s stride=%d prefill_len=%d max_new_tokens=%d",
            device,
            args.num_chunks or "until EOF",
            args.stride or args.prefill_len,
            args.prefill_len,
            args.max_new_tokens,
        )

        run_single_device_inference(
            args=args,
            model=model,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            data=data,
            device=device,
            tokenizer=tokenizer,
        )

    except Exception as e:
        log.error(f"Error: {e}")
        log.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
