#!/usr/bin/env python3
"""
train_swinyolo.py — SwinYOLO Training Launcher
===============================================
A convenience launcher that wraps train.py with SwinYOLO best-practice defaults.
Suitable for any dataset — point it at your data yaml with --data.

Architecture:
  • Backbone : YOLOv5s-CSP + C3SWT (Swin Window Attention @ all PANet levels)
               - MLP dropout=0.1, DropPath=0.1, Dropout2d=0.05 on C3SWT output
  • Neck      : BiFPN (4-level bidirectional feature pyramid, optional DCNv2 @ P3)
  • Attention : CoordAttMulti (Coordinate Attention per FPN level)
  • Heads     : 4-scale Detection (P2/P3/P4/P5) for small-object sensitivity

V4 Best Practices applied automatically:
  - lr=1e-6        : Prevents numerical instability in Swin blocks at epoch ~10
  - AdamW          : Recommended optimizer for Transformer-based models
  - ValLoss ES     : Early stopping on validation loss (more stable than mAP on dense datasets)
  - Close-mosaic   : Disables heavy augmentation for the final N epochs (YOLOv8 trick)
  - Grad clip      : max_norm=1.0 — standard for Swin Transformer training
  - NaN guard      : Skips batches with non-finite loss to protect optimizer state
  - DETR logger    : Per-batch training log with ETA and loss breakdown

Usage:
    # Minimal — only --data is required
    python train_swinyolo.py --data data/your_dataset.yaml

    # Custom epochs and batch size
    python train_swinyolo.py --data data/your_dataset.yaml --epochs 150 --batch-size 8

    # Use cyclic LR scheduler (fine-tuning with oscillating LR)
    python train_swinyolo.py --data data/your_dataset.yaml --scheduler cyclic

    # Custom image size (must be multiple of 32)
    python train_swinyolo.py --data data/your_dataset.yaml --img-size 640

    # Resume from checkpoint
    python train_swinyolo.py --data data/your_dataset.yaml --resume runs/swinyolo/exp/weights/checkpoint.pth

Checkpoints saved to:
    runs/swinyolo/<name>/weights/
        best.pth         ← best validation mAP
        checkpoint.pth   ← most recent epoch
        evals/latest.pth
        evals/NNN.pth    (every 50 epochs)

Training log (DETR-style per-batch):
    runs/swinyolo/<name>/training_log.txt

Linux/tmux tip — run inside tmux to survive SSH disconnects:
    tmux new -s swinyolo
    python train_swinyolo.py --data data/your_dataset.yaml
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# SwinYOLO default configuration — best practices for dense small-object tasks
# Override any value via CLI (e.g. --epochs 200)
# ---------------------------------------------------------------------------
SWINYOLO_DEFAULTS = dict(
    cfg             = "models/yolov5s_swint.yaml",   # SwinYOLO architecture
    hyp             = "data/hyps/hyp.swinyolo-adamw.yaml",  # AdamW hyps (lr=1e-6)
    weights         = "",                             # train from scratch (recommended)
    batch_size      = 4,                              # conservative: 1024px is memory-heavy
    imgsz           = 1024,                           # high-res default for small objects
    epochs          = 100,                            # ValLoss ES will stop earlier if needed
    optimizer       = "AdamW",
    cos_lr          = True,                           # cosine LR annealing
    scheduler       = "cosine",
    workers         = 8,                              # data loader workers (tune per machine)
    project         = "runs/swinyolo",
    name            = "exp",
    label_smoothing = 0.1,                            # recommended for dense datasets
    patience        = 30,                             # ValLossEarlyStopping patience
    no_augs         = 10,                             # close-mosaic: disable augment last N epochs
    seed            = 42,
)


def parse_opt():
    """Parse CLI overrides; any unrecognised key is passed through to train.py."""
    parser = argparse.ArgumentParser(
        description="SwinYOLO training launcher — wraps train.py with V4 best-practice defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--data",       required=True, type=str,
                        help="Path to dataset yaml (e.g. data/coco128.yaml)")
    # Optional overrides
    parser.add_argument("--cfg",        type=str,   default=SWINYOLO_DEFAULTS["cfg"])
    parser.add_argument("--hyp",        type=str,   default=SWINYOLO_DEFAULTS["hyp"])
    parser.add_argument("--weights",    type=str,   default=SWINYOLO_DEFAULTS["weights"])
    parser.add_argument("--batch-size", type=int,   default=SWINYOLO_DEFAULTS["batch_size"])
    parser.add_argument("--img-size",   type=int,   default=SWINYOLO_DEFAULTS["imgsz"])
    parser.add_argument("--epochs",     type=int,   default=SWINYOLO_DEFAULTS["epochs"])
    parser.add_argument("--optimizer",  type=str,   default=SWINYOLO_DEFAULTS["optimizer"])
    parser.add_argument("--scheduler",  type=str,   default=SWINYOLO_DEFAULTS["scheduler"],
                        choices=["cosine", "linear", "cyclic"])
    parser.add_argument("--workers",    type=int,   default=SWINYOLO_DEFAULTS["workers"])
    parser.add_argument("--project",    type=str,   default=SWINYOLO_DEFAULTS["project"])
    parser.add_argument("--name",       type=str,   default=SWINYOLO_DEFAULTS["name"])
    parser.add_argument("--label-smoothing", type=float, default=SWINYOLO_DEFAULTS["label_smoothing"])
    parser.add_argument("--patience",   type=int,   default=SWINYOLO_DEFAULTS["patience"])
    parser.add_argument("--no-augs",    type=int,   default=SWINYOLO_DEFAULTS["no_augs"],
                        help="Disable mosaic+copy_paste for the last N epochs (close-mosaic)")
    parser.add_argument("--seed",       type=int,   default=SWINYOLO_DEFAULTS["seed"])
    parser.add_argument("--device",     type=str,   default="0",
                        help="CUDA device (e.g. 0, 0,1, or cpu)")
    parser.add_argument("--resume",     type=str,   default="",
                        help="Resume from checkpoint path")
    parser.add_argument("--cos-lr",     action="store_true", default=True,
                        help="Use cosine LR scheduler (default: on)")
    parser.add_argument("--no-cos-lr",  dest="cos_lr", action="store_false",
                        help="Disable cosine LR scheduler")
    return parser.parse_args()


def build_command(opt) -> list:
    """Translate parsed options into a train.py CLI argument list."""
    cmd = [sys.executable, "train.py"]
    cmd += ["--data",            opt.data]
    cmd += ["--cfg",             opt.cfg]
    cmd += ["--hyp",             opt.hyp]
    cmd += ["--batch-size",      str(opt.batch_size)]
    cmd += ["--img",             str(opt.img_size)]
    cmd += ["--epochs",          str(opt.epochs)]
    cmd += ["--optimizer",       opt.optimizer]
    cmd += ["--scheduler",       opt.scheduler]
    cmd += ["--workers",         str(opt.workers)]
    cmd += ["--project",         opt.project]
    cmd += ["--name",            opt.name]
    cmd += ["--label-smoothing", str(opt.label_smoothing)]
    cmd += ["--patience",        str(opt.patience)]
    cmd += ["--no-augs",         str(opt.no_augs)]
    cmd += ["--seed",            str(opt.seed)]
    cmd += ["--device",          opt.device]
    if opt.weights:
        cmd += ["--weights", opt.weights]
    else:
        cmd += ["--weights", ""]
    if opt.cos_lr:
        cmd += ["--cos-lr"]
    if opt.resume:
        cmd += ["--resume", opt.resume]
    return cmd


def main():
    opt = parse_opt()
    cmd = build_command(opt)

    print("=" * 70)
    print("SwinYOLO Training Launcher")
    print("=" * 70)
    print(f"  Data        : {opt.data}")
    print(f"  Architecture: {opt.cfg}")
    print(f"  Epochs      : {opt.epochs}  (ValLoss ES patience={opt.patience})")
    print(f"  Batch size  : {opt.batch_size}")
    print(f"  Image size  : {opt.img_size}px")
    print(f"  Optimizer   : {opt.optimizer} @ lr=1e-6")
    print(f"  Scheduler   : {opt.scheduler} LR")
    print(f"  Device      : {opt.device}")
    print(f"  Close-mosaic: last {opt.no_augs} epochs")
    print(f"  Label smooth: {opt.label_smoothing}")
    print(f"  Output dir  : {opt.project}/{opt.name}/")
    print()
    print("V4 best practices active:")
    print("  ✅ lr=1e-6  (prevents Swin block gradient spike)")
    print("  ✅ ValLossEarlyStopping (stable on dense datasets)")
    print("  ✅ Close-mosaic (cleaner final epochs)")
    print("  ✅ Dropout=0.1 + DropPath=0.1 in C3SWT blocks")
    print("  ✅ NaN loss guard (protects optimizer state)")
    print("  ✅ Grad clip max_norm=1.0 (Swin Transformer standard)")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    print("=" * 70)

    # Ensure we're in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Create output dir for the system log
    log_dir = os.path.join(opt.project, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    system_log = os.path.join(log_dir, "system_output.log")
    print(f"System output → {system_log}")
    print("=" * 70)

    # Cross-platform subprocess launch
    # On Linux: os.setsid() puts train.py in a new process group so it
    # survives SSH disconnects / terminal closes.
    # On Windows: CREATE_NEW_PROCESS_GROUP achieves similar isolation.
    kwargs = dict(check=False, stdout=open(system_log, "w"), stderr=subprocess.STDOUT)
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid   # Linux/macOS — survive SIGHUP
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # Windows

    result = subprocess.run(cmd, **kwargs)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
