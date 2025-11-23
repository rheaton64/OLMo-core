"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llm/train.py run_name [OVERRIDES...]
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, cast

import rich
import torch
import torch.nn as nn

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType, PipelineScheduleType, PipelineSplitStyle
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.nn.xlstm_large.model import xLSTMLargeConfig, mLSTMLayer
from olmo_core.nn.xlstm_large.components import RMSNorm
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimConfig, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


def init_xlstm_gate_biases(model: nn.Module) -> None:
    """
    Mirror the bias initialization used in `init_gate_biases_v2` for xLSTM models.
    Sets:
        - output gate bias to +1
        - input gate bias to -10
        - forget gate bias to logit(0.95)
    """

    def logit(p: float) -> float:
        return float(math.log(p / (1.0 - p)))

    forget_bias = logit(0.95)
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, mLSTMLayer):
                continue

            if layer.config.weight_mode == "single":
                if layer.ogate_preact.bias is not None:
                    layer.ogate_preact.bias.fill_(1.0)
                if layer.igate_preact.bias is not None:
                    layer.igate_preact.bias.fill_(-10.0)
                if layer.fgate_preact.bias is not None:
                    layer.fgate_preact.bias.fill_(forget_bias)
            elif layer.config.weight_mode == "fused":
                if layer.qkv_opreact.bias is not None:
                    layer.qkv_opreact.bias[-layer.v_dim :].fill_(1.0)
                if layer.ifgate_preact.bias is not None:
                    num_heads = layer.config.num_heads
                    layer.ifgate_preact.bias[:num_heads].fill_(-10.0)
                    layer.ifgate_preact.bias[num_heads:].fill_(forget_bias)


DATA_ROOT = "/raid/datasets/SYNTH"
DATA_PATHS = []
for file in os.listdir(DATA_ROOT):
    if file.startswith("train"):
        DATA_PATHS.append(f"{DATA_ROOT}/{file}")


# docs: start-define-config
@dataclass
class ExperimentConfig(Config):
    model: xLSTMLargeConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 1264
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""
    # docs: end-define-config

@dataclass
class AdamWInstanceConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.99, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    foreach: Optional[bool] = None
    fused: Optional[bool] = None
    _is_adamw_instance: bool = True

    @classmethod
    def optimizer(cls): # type: ignore
        def create_optimizer(model, **kwargs):
            decay, no_decay = [], [] # type: ignore
            for m in model.modules():
                for name, p in m.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    # no decay for norms & embeddings & biases
                    if isinstance(m, (torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)) or name.endswith(
                        "bias"
                    ):
                        no_decay.append(p)
                    else:
                        decay.append(p)

            return torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": kwargs["weight_decay"]},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                **kwargs
            )
        return create_optimizer


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # docs: start-build-components
    # Build components.
    model = config.model.build(init_device="meta")
    init_xlstm_gate_biases(model)
    train_module = config.train_module.build(model) # type: ignore
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    # docs: end-build-components

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # docs: start-load-path
    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
    # docs: end-load-path

    # Train.
    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig(
        vocab_size=65536,
        eos_token_id=2,
        pad_token_id=3,
        bos_token_id=1,
        identifier="/raid/weights/xlstm",
    )

    # docs: start-model-config
    model_config = xLSTMLargeConfig(
        vocab_size=tokenizer_config.padded_vocab_size(),
        embedding_dim=1152,
        num_blocks=72,
        num_heads=8,
    )
    # docs: end-model-config

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size_tokens = opts.micro_batch_size * opts.sequence_length
    global_batch_size_tokens = micro_batch_size_tokens * world_size * opts.grad_accum_steps

    log.info(f"Using data root: {DATA_ROOT}")
    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_tokens,  # NOTE: this is specified in tokens, not instances
        seed=0,
        num_workers=4,
    )

    pp_config = None
    if opts.pipeline_degree > 1:
        pp_config = TransformerPipelineParallelConfig(
            degree=opts.pipeline_degree,
            schedule=opts.pipeline_schedule,
            style=opts.pipeline_style,
        )

    ac_config = None
    if opts.act_ckpt:
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.9,
        )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=micro_batch_size_tokens,  # NOTE: this is specified in tokens, not instances
        max_sequence_length=opts.sequence_length,
        optim=AdamWInstanceConfig(
            lr=3e-4,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        pp_config=pp_config,
        max_grad_norm=0.5,
        scheduler=CosWithWarmup(warmup_steps=2000),
        ac_config=ac_config,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                project="SYNTH-xLSTM-large",
                cancel_check_interval=10,
                enabled=True,
                name="fiery-silence-12"
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    # Apply overrides.
    # docs: start-config-merge
    config = config.merge(overrides)
    # docs: end-config-merge

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train a transformer language model on c4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--model-factory",
        type=str,
        default="llama2_271M",
        help="""The name of the model factory to use.
        This can be any classmethod on the TransformerConfig class.""",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8192,
        help="""The sequence length to train and eval on.""",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="""The micro batch size in instances.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    parser.add_argument(
        "--pipeline-degree",
        type=int,
        default=3,
        help="""Number of pipeline stages to use. Set to 1 to disable pipeline parallelism.""",
    )
    parser.add_argument(
        "--pipeline-schedule",
        type=PipelineScheduleType,
        default=PipelineScheduleType.interleaved_1F1B,
        choices=list(PipelineScheduleType),
        help="""Pipeline schedule to use when pipeline parallelism is enabled.""",
    )
    parser.add_argument(
        "--pipeline-style",
        type=PipelineSplitStyle,
        choices=list(PipelineSplitStyle),
        help="""Optional override for pipeline split style.""",
    )
    parser.add_argument(
        "--act-ckpt",
        action="store_true",
        help="""Enable activation checkpointing.""",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="""Number of gradient accumulation steps.""",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    config = build_config(opts, overrides)
    torch._dynamo.config.capture_scalar_outputs = True

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


if __name__ == "__main__":
    main()
