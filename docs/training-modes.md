# Training and validation modes

The public interface uses the same three method names everywhere: `3dreambooth`, `3dapter`, and `joint`. Configure experiments in YAML and run them through `scripts/run.py`; the existing Python entrypoints remain available for advanced use.

| Method | Training entrypoint | Validation entrypoint | Purpose |
|---|---|---|---|
| `3dreambooth` | `train_3dreambooth.py` | `validate_3dreambooth.py` | Subject-specific spatial LoRA baseline |
| `3dapter` | `train_3dapter.py` | `validate_3dapter.py` | Reference-conditioned adapter pretraining and inference |
| `joint` | `train_joint.py` | `validate_joint.py` | Joint subject LoRA and 3Dapter optimization |

## Config runner

```bash
python scripts/run.py configs/train/3dreambooth.yaml --dry-run
python scripts/run.py configs/train/3dapter.yaml --dry-run
python scripts/run.py configs/train/joint.yaml --dry-run
```

Every config defines `stage`, `method`, `runtime`, `environment`, and the entrypoint `args`. Override settings without editing YAML:

```bash
python scripts/run.py configs/train/joint.yaml \
  --set args.max_steps=800 \
  --set runtime.nproc_per_node=4 \
  --set args.sp_size=4
```

### 3DreamBooth

Requires `instance_data_root` and the canonical prompt `A video of a [v] [class].`, for example `A video of a [rhs plushie].`. The runner removes brackets and trains LoRA over the full prompt. It produces the subject LoRA baseline without reference conditioning.

In validation prompts, square brackets mark the only token span that receives text LoRA. The runner generates the legacy span arguments automatically; see [Prompt and LoRA span format](prompt-format.md).

### 3Dapter

The default config pretrains on Subjects200K. For paired local data, disable `use_subjects200k` and set both `instance_data_root` and `reference_data_root`. Validation requires `tdapter_path`, a reference image, and prompts.

### Joint

Uses the same bracketed prompt convention as 3DreamBooth. It also requires subject images, normalized reference images, and a pretrained 3Dapter. Set `instance_data_root`, `reference_path`, and `tdapter_path`. Its checkpoint contains the subject adapter and updated 3Dapter used by `validate_joint.py`.

## Multi-GPU launch

Keep `runtime.nproc_per_node` and `args.sp_size` aligned for sequence parallelism. Choose a free `runtime.master_port`. FSDP and optimizer settings remain method arguments under `args`.
