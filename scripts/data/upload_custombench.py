#!/usr/bin/env python3
"""Validate and optionally upload a prepared 3D-CustomBench release to Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


PLACEHOLDER_LICENSE_TEXT = "must replace this section"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("3D-CustomBench-release"))
    parser.add_argument("--repo-id", default="lanikoworld/3D-CustomBench")
    parser.add_argument("--public", action="store_true", help="Create a public repo; private is the safe default")
    parser.add_argument("--upload", action="store_true", help="Actually create/update the Hub dataset")
    return parser.parse_args()


def validate_release(folder: Path) -> tuple[int, int, int]:
    required = [folder / "README.md", folder / "manifest.json", folder / "subjects"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {', '.join(missing)}")

    readme = (folder / "README.md").read_text(encoding="utf-8")
    if PLACEHOLDER_LICENSE_TEXT in readme.lower():
        raise ValueError(
            f"Choose the dataset license in {folder / 'README.md'} before upload; "
            "the current dataset card still contains the license placeholder."
        )

    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    subjects = manifest.get("subjects", [])
    if not subjects:
        raise ValueError("manifest.json contains no subjects")
    return (
        len(subjects),
        sum(int(subject["num_images"]) for subject in subjects),
        sum(int(subject["num_references"]) for subject in subjects),
    )


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()
    subjects, images, references = validate_release(folder)
    visibility = "public" if args.public else "private"
    print(f"Validated {subjects} subjects, {images} images, and {references} references")
    print(f"Target: https://huggingface.co/datasets/{args.repo_id} ({visibility})")

    if not args.upload:
        print("Dry run only. Add --upload after reviewing the target and visibility.")
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder,
        private=not args.public,
        ignore_patterns=[".DS_Store", "**/.DS_Store", "__pycache__/**", ".cache/**"],
        num_workers=8,
        print_report_every=30,
    )
    if args.public:
        api.update_repo_settings(args.repo_id, repo_type="dataset", private=False)

    print(f"Uploaded: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
