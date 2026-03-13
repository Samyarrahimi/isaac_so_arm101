# RSL-RL Training & Play Scripts

Scripts for training and evaluating SO-ARM100 policies with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (PPO).

All commands must be run from the **repository root** (`isaac_so_arm101/`).

---

## Registered Tasks

| Task ID | Purpose |
|---|---|
| `SO-ARM100-Grasp-Object-v0` | Pretraining — single object (green block) |
| `SO-ARM100-Grasp-Object-Play-v0` | Play pretrained policy |
| `SO-ARM100-Grasp-Object-Finetune-v0` | Fine-tuning — multi-object pool (bolt, gear, nut, peg, block) |
| `SO-ARM100-Grasp-Object-Finetune-Play-v0` | Play fine-tuned policy |

Logs are saved to `logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/`.

---

## train.py — Arguments

### Core

| Argument | Type | Default | Description |
|---|---|---|---|
| `--task` | str | required | Task ID from the table above |
| `--num_envs` | int | from config | Number of parallel environments |
| `--seed` | int | from config | Random seed (`-1` = random) |
| `--max_iterations` | int | from config | Override max PPO iterations |
| `--headless` | flag | — | Run without GUI (required on servers) |
| `--distributed` | flag | — | Multi-GPU training |
| `--device` | str | `cuda:0` | Compute device |

### Video Recording

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | flag | — | Record video clips during training. Automatically enables cameras. |
| `--video_length` | int | `200` | Clip length in simulation steps |
| `--video_interval` | int | `2000` | Record a clip every N steps |

### Experiment & Checkpoint (RSL-RL)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--experiment_name` | str | from config | Override experiment name (changes save directory) |
| `--run_name` | str | — | Suffix appended to the timestamped log directory |
| `--resume` | flag | — | **Required** to load weights from a checkpoint |
| `--load_run` | str | latest | Directory name of the run to load from (regex, matched inside `logs/rsl_rl/{experiment_name}/`) |
| `--load_experiment_name` | str | same as save | Experiment to **load from** when different from the save experiment (e.g. loading pretrained weights into a finetune run) |
| `--checkpoint` | str | latest | Specific checkpoint filename inside the run directory |
| `--logger` | str | tensorboard | Logging backend: `tensorboard`, `wandb`, `neptune` |

---

## play.py — Arguments

### Core

| Argument | Type | Default | Description |
|---|---|---|---|
| `--task` | str | required | Use the `-Play-` task variant |
| `--num_envs` | int | from config | Number of parallel environments |
| `--seed` | int | from config | Random seed |
| `--headless` | flag | — | Run without GUI |
| `--real-time` | flag | — | Throttle simulation to real-time speed |
| `--disable_fabric` | flag | — | Disable fabric and use USD I/O (for debugging) |

### Checkpoint

| Argument | Type | Default | Description |
|---|---|---|---|
| `--load_run` | str | latest | Run directory name to load from |
| `--experiment_name` | str | from config | Experiment to load from |
| `--checkpoint` | str | latest | Absolute path to a specific `.pt` file |
| `--use_pretrained_checkpoint` | flag | — | Download and use the published Isaac Nucleus checkpoint |

### Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | flag | — | Record a single video clip. Automatically enables cameras. |
| `--video_length` | int | `200` | Clip length in steps |
| `--save_images` | flag | — | Save context and wrist camera frames as PNG. Automatically enables cameras. |
| `--save_images_interval` | int | `10` | Save a frame every N steps |

Images are saved to `isaac_so_arm101_imgs/{task}/{object}/context/` and `.../wrist/`.
Exported policy (JIT + ONNX) is saved to `{run_dir}/exported/` on every play run.

---

## Examples

### 1 — Pretrain from scratch

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Grasp-Object-v0 \
    --num_envs 4096 \
    --seed 42 \
    --headless
```

### 2 — Pretrain with video recording

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Grasp-Object-v0 \
    --num_envs 4096 \
    --seed 42 \
    --headless \
    --video \
    --video_length 300 \
    --video_interval 1000
```

### 3 — Resume interrupted pretraining

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Grasp-Object-v0 \
    --num_envs 4096 \
    --seed 42 \
    --headless \
    --resume \
    --load_run 2026-02-02_15-12-13
```

`--load_run` is a regex matched against directory names inside `logs/rsl_rl/grasp_object/`.
Omit `--load_run` to automatically pick the latest run.

### 4 — Fine-tune from pretrained weights

Load from the `grasp_object` experiment, save to `grasp_object_finetune` (the default for the finetune task).

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Grasp-Object-Finetune-v0 \
    --num_envs 1024 \
    --seed 42 \
    --headless \
    --resume \
    --load_experiment_name grasp_object \
    --load_run 2026-02-02_15-12-13
```

### 5 — Resume interrupted fine-tuning

Both load and save are in `grasp_object_finetune`, so no `--load_experiment_name` needed.

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Grasp-Object-Finetune-v0 \
    --num_envs 1024 \
    --seed 42 \
    --headless \
    --resume \
    --load_run 2026-03-01_10-00-00
```

### 6 — Play pretrained policy

```bash
python scripts/rsl_rl/play.py \
    --task SO-ARM100-Grasp-Object-Play-v0 \
    --num_envs 50 \
    --load_run 2026-02-02_15-12-13 \
    --enable_cameras
```

`--enable_cameras` is required for play (cameras are disabled during training).

### 7 — Play fine-tuned policy

```bash
python scripts/rsl_rl/play.py \
    --task SO-ARM100-Grasp-Object-Finetune-Play-v0 \
    --num_envs 50 \
    --load_run 2026-03-01_10-00-00 \
    --enable_cameras
```

### 8 — Play with video + image saving

```bash
python scripts/rsl_rl/play.py \
    --task SO-ARM100-Grasp-Object-Play-v0 \
    --num_envs 50 \
    --load_run 2026-02-02_15-12-13 \
    --video \
    --video_length 300 \
    --save_images \
    --save_images_interval 5
```

### 9 — Play specific checkpoint file

```bash
python scripts/rsl_rl/play.py \
    --task SO-ARM100-Grasp-Object-Play-v0 \
    --num_envs 50 \
    --checkpoint /abs/path/to/logs/rsl_rl/grasp_object/2026-02-02_15-12-13/model_2000.pt \
    --enable_cameras
```

---

## Log Structure

```
logs/rsl_rl/
├── grasp_object/
│   └── 2026-02-02_15-12-13/
│       ├── model_1000.pt
│       ├── model_2000.pt
│       ├── model_3000.pt          ← final checkpoint
│       ├── params/
│       │   ├── env.yaml           ← env config snapshot
│       │   └── agent.yaml         ← agent config snapshot
│       ├── videos/
│       │   ├── train/             ← training video clips
│       │   └── play/              ← play video clips
│       └── exported/
│           ├── policy.pt          ← TorchScript export
│           └── policy.onnx        ← ONNX export
└── grasp_object_finetune/
    └── 2026-03-01_10-00-00/
        └── ...
```

---

## Notes

- `--resume` is **required** to load weights. `--load_run` alone does not load weights.
- `--load_experiment_name` is only needed when loading from a **different** experiment than the one being saved to (e.g. fine-tuning from a pretrained run).
- `--load_run` is treated as a regex. `.*` matches the latest run. An exact timestamp matches that specific run.
- Cameras are `None` during training for speed. The `-Play-` task variants enable them automatically. Pass `--enable_cameras` when running play.
- `--video` during training automatically sets `--enable_cameras`.
