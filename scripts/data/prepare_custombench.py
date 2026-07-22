#!/usr/bin/env python3
"""Convert the internal 3D-CustomBench folders into the public dataset layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SUBJECT_NAMES = {
    "black_bag": "black_handbag",
    "black_bear": "bear_keychain",
    "black_printer": "multifunction_printer",
    "chair2": "office_chair",
    "cheese": "cat_figurine",
    "flowerpot": "deer_flowerpot",
    "graduation_bear": "graduation_bear",
    "handcream": "hand_cream",
    "headset": "gaming_headset",
    "lamp": "rattan_light_bulb",
    "lamp3": "white_light_bulb",
    "loacker": "wafer_bag",
    "lotion7": "lotion_bottle",
    "milk2": "milk_carton",
    "moose": "moose_plush",
    "motorcycle2": "covered_motorcycle",
    "mug": "blue_pig_mug",
    "mug2": "lavender_pitcher",
    "mug3": "floral_mug",
    "mug4": "black_gold_mug",
    "pants1": "drawstring_pants",
    "pill_bottle": "blue_label_pill_bottle",
    "pill_bottle2": "small_pill_bottle",
    "pink_duck1": "pink_rubber_duck",
    "rockery": "textured_rock",
    "sculpture2": "ceramic_bust",
    "sculpture5": "headband_bust",
    "seal": "pink_plush",
    "toilet": "toy_toilet",
    "yogurt_drink": "yogurt_drink",
}

PROMPT_ALIASES = {"pants1": "pants"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def copy_sequence(source: Path, destination: Path) -> int:
    files = image_files(source)
    destination.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(files, start=1):
        shutil.copy2(item, destination / f"{index:03d}{item.suffix.lower()}")
    return len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Legacy folder containing full/, cond/, and prompts/")
    parser.add_argument("output", type=Path, help="Empty output directory for the Hugging Face dataset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory (destructive; use only for generated release copies)",
    )
    return parser.parse_args()


def validate_source(source: Path) -> None:
    missing = [name for name in ("full", "cond", "prompts") if not (source / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing source directories: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    validate_source(source)

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Pass --force to replace it.")
        shutil.rmtree(output)

    subjects_dir = output / "subjects"
    manifest: dict[str, object] = {"format_version": 1, "subjects": []}

    source_subjects = {path.name for path in (source / "full").iterdir() if path.is_dir()}
    unknown = sorted(source_subjects - SUBJECT_NAMES.keys())
    if unknown:
        raise ValueError(f"Add public names for unknown subjects: {', '.join(unknown)}")

    for legacy_name, public_name in SUBJECT_NAMES.items():
        image_source = source / "full" / legacy_name
        reference_source = source / "cond" / legacy_name
        prompt_name = PROMPT_ALIASES.get(legacy_name, legacy_name)
        prompt_source = source / "prompts" / f"{prompt_name}.txt"
        if not image_source.is_dir() or not reference_source.is_dir() or not prompt_source.is_file():
            raise FileNotFoundError(f"Incomplete subject '{legacy_name}'")

        destination = subjects_dir / public_name
        image_count = copy_sequence(image_source, destination / "images")
        reference_count = copy_sequence(reference_source, destination / "references")
        prompt = prompt_source.read_text(encoding="utf-8").strip()
        (destination / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata = {
            "id": public_name,
            "legacy_id": legacy_name,
            "prompt": prompt,
            "num_images": image_count,
            "num_references": reference_count,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["subjects"].append(metadata)

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(SUBJECT_NAMES)} subjects at {output}")


if __name__ == "__main__":
    main()
