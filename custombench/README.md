---
pretty_name: 3D-CustomBench
language:
  - en
task_categories:
  - text-to-video
tags:
  - multi-view
  - subject-customization
  - video-generation
  - 3d-aware
license: cc-by-4.0
---

# 3D-CustomBench

3D-CustomBench is the multi-view subject benchmark introduced in **3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model**. It contains 30 subjects with ordered multi-view captures, background-normalized reference images, and evaluation prompts for customized video generation.

## Dataset summary

| Item | Count |
|---|---:|
| Subjects | 30 |
| Multi-view images | 897 |
| Reference images | 122 |

Each subject provides full 360-degree visual coverage for evaluating subject fidelity and 3D geometric consistency.

## Dataset structure

```text
3D-CustomBench/
├── README.md
├── manifest.json
└── subjects/
    └── graduation_bear/
        ├── images/
        │   ├── 001.jpeg
        │   └── ...
        ├── references/
        │   ├── 001.png
        │   └── ...
        ├── metadata.json
        └── prompt.txt
```

- `images/`: ordered multi-view captures used for subject customization and evaluation.
- `references/`: background-normalized conditioning images used by 3Dapter and Joint.
- `prompt.txt`: subject-specific evaluation prompt.
- `metadata.json`: stable public ID, legacy ID, prompt, and file counts.
- `manifest.json`: index and metadata for all subjects.

Public subject IDs use descriptive `snake_case`. The `legacy_id` field is retained only to reproduce internal experiments.

## Download

```bash
hf download lanikoworld/3D-CustomBench \
  --repo-type dataset \
  --local-dir ./datasets/3d-custombench
```

From the 3DreamBooth repository:

```bash
python scripts/data/download_custombench.py
```

## 3DreamBooth example

```bash
python scripts/run.py configs/examples/graduation_bear/train_joint.yaml
python scripts/run.py configs/examples/graduation_bear/validate_joint.yaml
```

## License

3D-CustomBench is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (`CC BY 4.0`). You may share and adapt the dataset for any purpose with appropriate attribution. This license covers only rights held by the dataset authors; third-party rights such as trademarks are not granted.

## Citation

If you use 3D-CustomBench, please cite the 3DreamBooth paper:

```bibtex
@misc{ko20263dreambooth,
  title         = {3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model},
  author        = {Hyun-kyu Ko and Jihyeon Park and Younghyun Kim and Dongheok Park and Eunbyung Park},
  year          = {2026},
  eprint        = {2603.18524},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.18524}
}
```

- [Paper](https://arxiv.org/abs/2603.18524)
- [Project page](https://ko-lani.github.io/3DreamBooth/)
- [Code](https://github.com/Ko-Lani/3DreamBooth)
