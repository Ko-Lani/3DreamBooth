# 3D-CustomBench release guide

3D-CustomBench is distributed separately from the source repository through the Hugging Face dataset repository [`lanikoworld/3D-CustomBench`](https://huggingface.co/datasets/lanikoworld/3D-CustomBench). The code repository contains download and release tools but does not track the images.

## Download for training and validation

```bash
python scripts/data/download_custombench.py
```

This is equivalent to:

```bash
hf download lanikoworld/3D-CustomBench \
  --repo-type dataset \
  --local-dir ./datasets/3d-custombench
```

The example configs expect:

```text
datasets/3d-custombench/
├── manifest.json
└── subjects/
    └── graduation_bear/
        ├── images/
        ├── references/
        ├── metadata.json
        └── prompt.txt
```

## First-time upload walkthrough

The prepared release is already available at `./3D-CustomBench-release`. The following steps publish that folder.

### 1. Create a Hugging Face account

Create or sign in to an account at [huggingface.co](https://huggingface.co). The account name or organization name becomes the first part of the repository ID. For example, `lanikoworld/3D-CustomBench` is owned by the authenticated `lanikoworld` account.

### 2. Confirm the dataset license

The dataset card is configured as `CC BY 4.0`. Confirm that every image may be redistributed under those terms before publishing. The first upload remains private for review.

### 3. Log in from this machine

The current Hugging Face CLI supports browser login:

```bash
hf auth login
```

Follow the printed URL and code. Alternatively, create a write-capable user token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it when prompted. Do not put the token in YAML, README, shell scripts, or Git.

Confirm the active account:

```bash
hf auth whoami
```

See the official [authentication documentation](https://huggingface.co/docs/huggingface_hub/package_reference/authentication) for token and login details.

### 4. Validate without uploading

The helper defaults to a private repository and performs no network write unless `--upload` is present:

```bash
python scripts/data/upload_custombench.py \
  --repo-id lanikoworld/3D-CustomBench
```

It checks the dataset card, manifest, subject count, image count, and target URL. The validator also confirms that the dataset card contains a finalized license.

### 5. Upload privately

After reviewing the dry run:

```bash
python scripts/data/upload_custombench.py \
  --repo-id lanikoworld/3D-CustomBench \
  --upload
```

This creates the dataset repository if needed and uploads the release folder. Rerunning the same command updates the existing repository.

### 6. Make it public

The safest workflow is to inspect the private dataset page first. When ready, either change visibility in the Hugging Face repository settings or explicitly create/upload it as public:

```bash
python scripts/data/upload_custombench.py \
  --repo-id lanikoworld/3D-CustomBench \
  --public \
  --upload
```

Hugging Face documents folder upload and resumability in its official [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Rebuild the release folder

The legacy internal layout uses separate `full/`, `cond/`, and `prompts/` trees with inconsistent names. Convert it into the public layout in a new directory:

```bash
python scripts/data/prepare_custombench.py \
  /path/to/legacy/custombench \
  /path/to/3D-CustomBench-release
```

The converter:

1. Places each subject under `subjects/<descriptive_snake_case>/`.
2. Renames images and references to ordered `001`, `002`, ... filenames.
3. Places prompts and metadata next to the corresponding subject.
4. Retains the original folder name as `legacy_id` for reproducibility.

Copy the dataset card into a newly rebuilt folder:

```bash
cp custombench/README.md /path/to/3D-CustomBench-release/README.md
```

Do not upload checkpoints, generated videos, caches, `.DS_Store`, W&B logs, or training outputs.
