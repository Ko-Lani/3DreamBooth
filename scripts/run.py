#!/usr/bin/env python3
"""Run a 3DreamBooth experiment from one YAML configuration file."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    ("train", "3dreambooth"): "train_3dreambooth.py",
    ("train", "3dapter"): "train_3dapter.py",
    ("train", "joint"): "train_joint.py",
    ("validate", "3dreambooth"): "validate_3dreambooth.py",
    ("validate", "3dapter"): "validate_3dapter.py",
    ("validate", "joint"): "validate_joint.py",
}
LORA_MARKUP = re.compile(r"\[([^\[\]]+)\]")


def load_config(path: Path, seen: set[Path] | None = None) -> DictConfig:
    """Load a config and recursively merge its optional ``extends`` parent."""
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)

    config = OmegaConf.load(path)
    parent = config.pop("extends", None)
    if parent is None:
        return config

    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return OmegaConf.merge(load_config(parent_path, seen), config)


def parse_lora_markup(prompt: str) -> tuple[str, list[str]]:
    """Return a clean prompt and LoRA spans marked as ``[text]``."""
    spans = [match.strip() for match in LORA_MARKUP.findall(prompt)]
    if any(not span for span in spans):
        raise ValueError(f"Empty LoRA span in prompt: {prompt!r}")
    clean_prompt = LORA_MARKUP.sub(lambda match: match.group(1).strip(), prompt)
    if "[" in clean_prompt or "]" in clean_prompt:
        raise ValueError(f"Unbalanced LoRA brackets in prompt: {prompt!r}")
    return clean_prompt, list(dict.fromkeys(spans))


def normalize_prompts(value: Any) -> tuple[Any, list[str]]:
    is_list = OmegaConf.is_list(value) or isinstance(value, (list, tuple))
    prompts = list(value) if is_list else [value]
    cleaned: list[str] = []
    spans: list[str] = []
    for prompt in prompts:
        clean_prompt, prompt_spans = parse_lora_markup(str(prompt))
        cleaned.append(clean_prompt)
        spans.extend(prompt_spans)
    unique_spans = list(dict.fromkeys(spans))
    return (cleaned if is_list else cleaned[0]), unique_spans


def apply_prompt_markup(config: DictConfig) -> None:
    """Translate bracket markup into the legacy prompt/span CLI arguments."""
    stage = str(config.stage).lower()
    method = str(config.method).lower()
    if method not in {"3dreambooth", "joint"}:
        return

    args = config.setdefault("args", {})
    args.pop("text_lora_spans", None)
    args.pop("validation_lora_spans", None)

    if stage == "train":
        if "train_prompts" in args:
            args.train_prompts, _ = normalize_prompts(args.train_prompts)
        if "validation_prompts" in args:
            args.validation_prompts, spans = normalize_prompts(args.validation_prompts)
            if not spans:
                raise ValueError(
                    "Validation prompts for 3DreamBooth/Joint must mark the LoRA phrase, "
                    "for example: 'A video of a [rhs plushie] on a beach.'"
                )
            args.validation_lora_spans = spans
    elif stage == "validate" and "prompt" in args:
        args.prompt, spans = normalize_prompts(args.prompt)
        if not spans:
            raise ValueError(
                "Validation prompts for 3DreamBooth/Joint must mark the LoRA phrase, "
                "for example: 'A video of a [rhs plushie] on a beach.'"
            )
        args.text_lora_spans = spans


def append_argument(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    command.append(f"--{key}")
    if isinstance(value, bool):
        command.append(str(value).lower())
    elif OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def build_command(config: DictConfig) -> list[str]:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    stage = str(config.stage).lower()
    method = str(config.method).lower()
    try:
        entrypoint = ENTRYPOINTS[(stage, method)]
    except KeyError as error:
        supported = ", ".join(f"{s}/{m}" for s, m in ENTRYPOINTS)
        raise ValueError(f"Unsupported stage/method '{stage}/{method}'. Choose: {supported}") from error

    apply_prompt_markup(config)
    runtime = config.get("runtime", {})
    python = str(runtime.get("python", sys.executable))
    command = [
        python,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={runtime.get('nproc_per_node', 1)}",
        f"--master_port={runtime.get('master_port', 29500)}",
        str(ROOT / entrypoint),
    ]
    for key, value in config.get("args", {}).items():
        append_argument(command, key, value)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value; repeat for multiple values",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    config = load_config(cli.config)
    if cli.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(cli.overrides))

    command = build_command(config)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    for key, value in config.get("environment", {}).items():
        env[str(key)] = str(value)

    print(f"Experiment: {config.get('name', cli.config.stem)}")
    print(shlex.join(command))
    if cli.dry_run:
        return 0

    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
