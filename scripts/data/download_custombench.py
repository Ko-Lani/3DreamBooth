#!/usr/bin/env python3
"""Download the release-ready 3D-CustomBench dataset from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="lanikoworld/3D-CustomBench",
        help="Hugging Face dataset repository (default: lanikoworld/3D-CustomBench)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/3d-custombench"),
        help="Local dataset directory",
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
    )
    print(f"3D-CustomBench downloaded to {Path(path).resolve()}")


if __name__ == "__main__":
    main()
