# Data layout

Use one stable subject layout for local data and downloaded 3D-CustomBench data:

```text
data/my_subject/
├── images/
│   ├── 001.jpeg
│   └── ...
├── references/
│   ├── 001.png
│   └── ...
└── prompt.txt
```

- `images/` is `instance_data_root` for 3DreamBooth and Joint training.
- `references/` is `reference_path` for Joint training and validation.
- One file inside `references/` is used for 3Dapter validation.

3D-CustomBench follows the same layout under `datasets/3d-custombench/subjects/<subject_id>/`. Download instructions and the release conversion workflow are in [custombench.md](custombench.md).

Keep datasets outside Git. The repository ignores `data/` and `datasets/`; only small illustrative assets should be committed intentionally.
