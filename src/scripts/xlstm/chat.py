#!/usr/bin/env python3
"""
CLI script for chatbot-style text generation using xLSTM-large models.

Example usage:
    # Basic usage
    uv run src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \
        --device cuda:0 \
        --tokenizer ~/OLMo-core/tokenizer \
        --show-special-tokens \
        --max-new-tokens 4096

    # With custom generation parameters
    python src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \\
        --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 \\
        --max-new-tokens 512 \\
        --temperature 0.7

    # Greedy decoding (deterministic)
    python src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \\
        --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 \\
        --no-do-sample
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from olmo_core.generate.sampling import select_next_token
from olmo_core.nn.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)
console = Console()

BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|end_of_text|>"
IM_START_USER = "<|im_start|>user\n"
IM_START_ASSISTANT = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
THINK_START = "<|think_start|>\n"
THINK_END = "<|think_end|>\n"

# ChatML-style template with <|im_start|>/<|im_end|> formatting and <think> for reasoning
DEFAULT_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}<|think_start|>{% endif %}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<|think_start|>
{% endif %}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chatbot-style text generation using xLSTM-large models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \\
      --tokenizer allenai/gpt-neox-olmo-dolma-v1_5

  # With custom generation parameters
  python src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \\
      --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 \\
      --max-new-tokens 512 --temperature 0.7 --top-p 0.9

  # Greedy decoding (deterministic)
  python src/scripts/xlstm/chat.py ~/ckpts/unshard/model.pt \\
      --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 \\
      --no-do-sample
        """,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the unsharded model checkpoint file (.pt or .safetensors).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer identifier/path. Defaults to the training config tokenizer identifier if present.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0). Lower values are more deterministic.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling. -1 means no top-k filtering (default: -1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 0.7)",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling (default: True). Set --no-do-sample for greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run on (default picks CUDA if available).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the model in float16 (may cause NaN issues). Default is float32 for stability.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to all conversations",
    )
    parser.add_argument(
        "--show-special-tokens",
        action="store_true",
        default=False,
        help="Show special tokens in generated text (default: False)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=DEFAULT_CHAT_TEMPLATE,
        help="Jinja2 chat template string. Default uses ChatML-style <|im_start|>/<|im_end|> with <think> reasoning tag.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream tokens as they are generated (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# UI Rendering (adapted from chat.py)
# -----------------------------------------------------------------------------


def render_assistant_message(message: str) -> Panel:
    """Render an assistant message as a chat bubble."""
    text = Text(message, style="default")
    return Panel(
        text,
        title="[bold magenta]Assistant[/bold magenta]",
        title_align="left",
        border_style="magenta",
        padding=(0, 1),
        width=None,
    )


def render_system_message(message: str) -> Panel:
    """Render a system prompt message with a distinct style."""
    text = Text(message, style="dim")
    return Panel(
        text,
        title="[bold yellow]System[/bold yellow]",
        title_align="left",
        border_style="yellow",
        padding=(0, 1),
        width=None,
    )


def render_tokenizer_info(tokenizer, chat_template: str) -> Panel:
    """Render tokenizer configuration details."""
    lines = []
    lines.append("HuggingFace Tokenizer:")
    lines.append(f"  • Vocab size: {tokenizer.vocab_size:,}")

    model_max_length = getattr(tokenizer, "model_max_length", None)
    lines.append(f"  • Model max length: {model_max_length}")

    # Special tokens
    all_special_tokens = getattr(tokenizer, "all_special_tokens", None)
    if all_special_tokens:
        special_tokens_str = ", ".join(all_special_tokens[:5])
        if len(all_special_tokens) > 5:
            special_tokens_str += f" ... ({len(all_special_tokens)} total)"
        lines.append(f"  • Special tokens: {special_tokens_str}")
    else:
        lines.append("  • Special tokens: N/A")

    # Token IDs
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    lines.append(f"  • EOS token: {eos_token} (ID: {eos_token_id})")
    pad_token = getattr(tokenizer, "pad_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    lines.append(f"  • Pad token: {pad_token} (ID: {pad_token_id})")
    bos_token = getattr(tokenizer, "bos_token", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    lines.append(f"  • BOS token: {bos_token} (ID: {bos_token_id})")

    info_text = Text("\n".join(lines))

    chat_template_panel = Panel(
        chat_template,
        title="[bold cyan]Chat Template[/bold cyan]",
        border_style="dim",
        padding=(0, 1),
    )
    combined_content = Group(info_text, "", chat_template_panel)
    return Panel(
        combined_content, title="[bold green]Tokenizer Info[/bold green]", border_style="green"
    )


def render_generation_config(
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    stream: bool,
) -> Panel:
    """Render generation configuration details."""
    left_items = []
    right_items = []

    left_items.append(f"• Max new tokens: {max_new_tokens}")

    if do_sample:
        left_items.append("• Sampling: enabled")
        left_items.append(f"• Temperature: {temperature}")
        top_k_str = "unlimited" if top_k == -1 else str(top_k)
        left_items.append(f"• Top-k: {top_k_str}")
        left_items.append(f"• Top-p: {top_p}")
    else:
        left_items.append("• Sampling: disabled (greedy)")

    right_items.append(f"• Streaming: {'enabled' if stream else 'disabled'}")

    left_text = Text("\n".join(left_items))
    right_text = Text("\n".join(right_items))
    columns = Columns([left_text, right_text], equal=True, expand=True)
    content = Group("[bold]Generation Parameters:[/bold]", columns)
    return Panel(content, title="[bold blue]Generation Config[/bold blue]", border_style="blue")


def render_model_info(model_cfg: xLSTMLargeConfig, dtype: torch.dtype) -> Panel:
    """Render model configuration details."""
    lines = []
    lines.append("xLSTM-Large Model:")
    lines.append(f"  • embedding_dim: {model_cfg.embedding_dim}")
    lines.append(f"  • num_heads: {model_cfg.num_heads}")
    lines.append(f"  • num_blocks: {model_cfg.num_blocks}")
    lines.append(f"  • vocab_size: {model_cfg.vocab_size}")
    lines.append(f"  • dtype: {dtype}")

    text = Text("\n".join(lines))
    return Panel(text, title="[bold purple]Model Info[/bold purple]", border_style="purple")


# -----------------------------------------------------------------------------
# Model Loading (adapted from inference.py)
# -----------------------------------------------------------------------------


def resolve_config_path(checkpoint_path: str, config_override: Optional[str]) -> Path:
    """Resolve the path to config.json."""
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
        # Fallback: if we are in unshard dir, maybe check parent
        if config_dir.name == "unshard" and (config_dir.parent / "config.json").is_file():
            return config_dir.parent / "config.json"

        raise FileNotFoundError(f"Missing config.json in '{config_dir}' or parent")

    return config_path


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return json.load(f)


def load_tokenizer(args: argparse.Namespace, config_dict: Dict[str, Any]):
    """Load tokenizer from HuggingFace."""
    identifier = args.tokenizer
    if identifier is None:
        identifier = (
            config_dict.get("train_module", {}).get("tokenizer", {}).get("identifier")
        )
    if identifier is None:
        raise ValueError(
            "Tokenizer identifier not provided. Use --tokenizer to specify a HuggingFace tokenizer."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers not installed. Install with: pip install transformers"
        ) from exc

    tok = AutoTokenizer.from_pretrained(identifier, trust_remote_code=True)
    log.info("Loaded tokenizer '%s'", identifier)
    return tok


def _sanitize_compiled_key(key: str) -> str:
    """Remove torch.compile prefixes from state dict keys."""
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


def load_model_state(
    model: xLSTMLarge,
    state_path: Path,
    *,
    strict: bool = True,
) -> None:
    """Load model state dict from a checkpoint file."""
    log.info("Loading state dict from '%s'", state_path)
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

    # Handle torch.compile prefixes
    if any("_orig_mod" in key for key in state_to_load.keys()):
        state_to_load = {
            _sanitize_compiled_key(key): value for key, value in state_to_load.items()
        }

    missing, unexpected = model.load_state_dict(state_to_load, strict=strict)
    if missing:
        log.warning("Missing keys during load: %s", missing)
    if unexpected:
        log.warning("Unexpected keys during load: %s", unexpected)


def load_xlstm_model(
    checkpoint_path: str,
    config_override: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[xLSTMLarge, xLSTMLargeConfig, Dict[str, Any]]:
    """Load xLSTM model from checkpoint."""
    config_path = resolve_config_path(checkpoint_path, config_override)
    config_dict = load_config(config_path)

    model_cfg = xLSTMLargeConfig.from_dict(config_dict["model"])
    model_cfg.mode = "inference"
    model_cfg.step_kernel = "native"
    model_cfg.sequence_kernel = "native_sequence__native"
    model_cfg.chunkwise_kernel = "chunkwise--native_autograd"
    model_cfg.return_last_states = True

    # Build model
    model = model_cfg.build()
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load weights
    load_model_state(model, Path(checkpoint_path), strict=True)

    return model, model_cfg, config_dict


# -----------------------------------------------------------------------------
# Generation (adapted from inference.py)
# -----------------------------------------------------------------------------


def generate_response(
    model: xLSTMLarge,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    stream_callback: Optional[Callable[[int], None]] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens from the xLSTM model.

    Args:
        model: The xLSTM model
        input_ids: Input token IDs (batch_size, seq_len)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to sample or use greedy decoding
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        stream_callback: Optional callback for each generated token
        eos_token_id: Optional EOS token ID to stop generation

    Returns:
        Generated token IDs (batch_size, num_generated)
    """
    device = input_ids.device

    # Prefill: process all input tokens
    logits, state = model(input_ids, None)

    # Sync to ensure prefill completes before decode loop
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Start with the last token from logits
    last_token = input_ids[:, -1:].contiguous()
    generated: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        logits, state = model(last_token, state)

        next_token = select_next_token(
            logits=logits.squeeze(1),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).unsqueeze(-1).contiguous()

        generated.append(next_token)

        if stream_callback is not None:
            stream_callback(int(next_token[0, 0].item()))

        # Check for EOS
        if eos_token_id is not None and next_token[0, 0].item() == eos_token_id:
            break

        last_token = next_token

    if generated:
        return torch.cat(generated, dim=1)
    else:
        return torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)


# -----------------------------------------------------------------------------
# Main Chat Loop
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    prepare_cli_environment()
    log.setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Determine dtype
    if args.fp16:
        dtype = torch.float16
        log.warning("Using float16 - may cause numerical instability with mLSTM kernels")
    else:
        dtype = torch.float32
        log.info("Using float32 for stable inference")

    console.print(f"[dim]Using device: {device}, dtype: {dtype}[/dim]")
    console.print()

    # Load model
    console.print("[dim]Loading model...[/dim]")
    try:
        model, model_cfg, config_dict = load_xlstm_model(
            args.checkpoint_path,
            None,
            device,
            dtype,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to load model:[/bold red] {e}")
        log.error("Failed to load model", exc_info=True)
        return

    console.print("[bold green]✓ Model loaded successfully![/bold green]")
    console.print()

    # Load tokenizer
    try:
        tokenizer = load_tokenizer(args, config_dict)
    except Exception as e:
        console.print(f"[bold red]Failed to load tokenizer:[/bold red] {e}")
        log.error("Failed to load tokenizer", exc_info=True)
        return

    # Override tokenizer special token ids
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 3
    tokenizer.bos_token_id = 1

    # Get EOS token ID
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Display info panels
    console.print(render_model_info(model_cfg, dtype))
    console.print(render_tokenizer_info(tokenizer, args.chat_template))
    console.print(
        render_generation_config(
            args.max_new_tokens,
            args.do_sample,
            args.temperature,
            args.top_k,
            args.top_p,
            args.stream,
        )
    )

    # Welcome message
    welcome_text = Text()
    welcome_text.append("xLSTM Chatbot ready! ", style="bold green")
    welcome_text.append("Type your message and press Enter.\n\n", style="dim")
    welcome_text.append("Commands:\n", style="bold")
    welcome_text.append("  /quit or /exit ", style="cyan")
    welcome_text.append("- Exit the chatbot\n", style="dim")
    welcome_text.append("  /clear ", style="cyan")
    welcome_text.append("- Clear conversation history\n", style="dim")
    welcome_text.append("  /help ", style="cyan")
    welcome_text.append("- Show this help message", style="dim")

    console.print(Panel(welcome_text, title="[bold red]Welcome[/bold red]", border_style="red"))
    console.print()

    # Conversation history
    conversation_history: list[dict[str, str]] = []
    if args.system_prompt:
        conversation_history.append({"role": "system", "content": args.system_prompt})
        console.print(render_system_message(args.system_prompt))
        console.print()

    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                console.print()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            elif user_input.lower() == "/clear":
                conversation_history = []
                if args.system_prompt:
                    conversation_history.append({"role": "system", "content": args.system_prompt})
                    console.print()
                    console.print(render_system_message(args.system_prompt))
                    console.print()
                console.print("[bold green]✓ Conversation history cleared.[/bold green]\n")
                continue
            elif user_input.lower() == "/help":
                help_text = Text()
                help_text.append("Commands:\n", style="bold")
                help_text.append("  /quit or /exit ", style="cyan")
                help_text.append("- Exit the chatbot\n", style="dim")
                help_text.append("  /clear ", style="cyan")
                help_text.append("- Clear conversation history\n", style="dim")
                help_text.append("  /help ", style="cyan")
                help_text.append("- Show this help message", style="dim")
                console.print(
                    Panel(help_text, title="[bold blue]Help[/bold blue]", border_style="blue")
                )
                console.print()
                continue

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Build prompt using chat template
            prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=args.chat_template,
            )
            console.print(f"[bold cyan]Prompt:[/bold cyan] [{prompt}]")

            try:
                # Tokenize input
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Container for streamed tokens
                generated_tokens: list[int] = []

                def stream_callback(token_id: int):
                    generated_tokens.append(token_id)
                    if args.stream:
                        token_text = tokenizer.decode(
                            [token_id], skip_special_tokens=not args.show_special_tokens
                        )
                        console.print(token_text, end="")

                if args.stream:
                    console.print()
                    console.print("[bold magenta]Assistant:[/bold magenta] ", end="")

                with torch.inference_mode():
                    generate_response(
                        model,
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        stream_callback=stream_callback,
                        eos_token_id=eos_token_id,
                    )

                # Decode full response
                response_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=not args.show_special_tokens
                )

                if args.stream:
                    console.print()  # Newline after streaming
                else:
                    console.print()
                    console.print(render_assistant_message(response_text))

                conversation_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                log.error(f"Generation error: {e}", exc_info=True)
                error_panel = Panel(
                    Text(f"Error: {e}", style="red"),
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
                console.print()
                console.print(error_panel)
                # Remove the user message that failed
                conversation_history.pop()

    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        log.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

