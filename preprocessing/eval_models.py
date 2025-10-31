#!/usr/bin/env python3
# Unified single-set evaluator for T-PRIME (fixed validation set)
# Usage example:
#   python3 -u eval_models.py \
#     --model_path /home/ehh588/t-prime/TPrime_transformer/model_cp/on_clean_smalldata_ft_best.pt \
#     --val_root   /mnt/dm-3/DroneDetect/dronedetect_eval/CLEAN/ON \
#     --datasets   AIR MP1 MP2 MIN INS PHA DIS \
#     --transformer_version v1 \
#     --transformer sm \
#     --rmsnorm \
#     --back_class \
#     --testing_mode future \
#     --use_gpu \
#     --outdir ./results_eval_ON_ON_val

import os, sys, argparse, json, math, types
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inspect import signature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo-local imports: make parent visible then import ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Dataset
from TPrime_dataset import TPrimeDataset_Transformer

# Models
from TPrime_transformer.model_transformer import (
    TransformerModel,
    TransformerModel_v2,
)

# Optional RMSNorm (if present in your repo)
try:
    from TPrime_transformer.rmsnorm import RMSNorm
except Exception:
    RMSNorm = None

# ---------- helpers ----------
def filter_ds_kwargs(ds_cls, **kwargs):
    """Keep only kwargs supported by the dataset class in THIS repo."""
    params = set(signature(ds_cls.__init__).parameters.keys())
    params.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs.keys() if k not in filtered]
    if dropped:
        print(f"[warn] dataset ignored unsupported args: {dropped}")
    return filtered

def flex_load_checkpoint(path):
    """
    Load a torch checkpoint in a robust way and return a plain state_dict.
    Supports formats like:
      - torch.save(model.state_dict())
      - torch.save({'state_dict': ..., ...})
      - torch.save({'model_state_dict': ..., ...})
      - torch.save(nn.Module)
    """
    obj = torch.load(path, map_location="cpu")
    # nn.Module?
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    # dict variants
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "module", "model", "net"]:
            if k in obj and isinstance(obj[k], (dict, types.MappingProxyType)):
                return dict(obj[k])
        # plain statedict?
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError(f"Unrecognized checkpoint format at {path}")

def infer_ckpt_dims(sd):
    """
    Try to infer d_model (embedding dim) and classifier out dim from state_dict.
    Looks for common parameter names; falls back heuristics.
    """
    out_classes = None
    d_model = None

    # classifier head search (common names)
    head_names = [
        "classifier.weight", "fc.weight", "head.weight", "proj.weight",
        "output_layer.weight"
    ]
    for name in head_names:
        if name in sd and sd[name].ndim == 2:
            out_classes = sd[name].shape[0]
            break
    if out_classes is None:
        # fallback: pick any 2D weight with small-ish out-dim 5..64
        candidates = [(k, v.shape) for k, v in sd.items() if v.ndim == 2 and 5 <= v.shape[0] <= 64]
        if candidates:
            out_classes = candidates[0][1][0]

    # d_model: look for LayerNorm / out_proj shapes or pos_encoder
    for k in [
        "norm.weight",
        "transformer_encoder.layers.0.self_attn.out_proj.weight",
        "transformer_encoder.layers.0.norm1.weight",
        "pos_encoder.pe",
        "pre_classifier.weight",
    ]:
        if k in sd:
            t = sd[k]
            if k.endswith("weight") and t.ndim == 1:
                d_model = t.shape[0]
                break
            if "out_proj.weight" in k and t.ndim == 2:
                d_model = t.shape[0]
                break
            if k == "pos_encoder.pe" and t.ndim == 3:
                d_model = t.shape[2]
                break
            if k == "pre_classifier.weight" and t.ndim == 2:
                d_model = t.shape[0]
                break

    return d_model, out_classes

def build_model(args, d_model, nclasses, device):
    # SM vs LG
    if args.transformer == "sm":
        nhead = 4
        num_layers = 2
        dim_feedforward = 2048
    else:  # lg
        nhead = 8
        num_layers = 4
        dim_feedforward = 4096

    ModelCls = TransformerModel if args.transformer_version == "v1" else TransformerModel_v2
    model = ModelCls(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        num_classes=nclasses,
    )
    model.to(device)
    return model

def apply_state_dict(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

def make_rms_layer(args, d_model, device):
    if not args.rmsnorm:
        return None
    if RMSNorm is None:
        print("[warn] --rmsnorm requested but RMSNorm not found in repo, skipping.")
        return None
    layer = RMSNorm(d_model)
    return layer.to(device)

@torch.no_grad()
def evaluate(model, criterion, dataloader, nclasses, device, rms_layer=None, class_order=None):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    cm = np.zeros((nclasses, nclasses), dtype=np.float64)

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        if rms_layer is not None:
            X = rms_layer(X)

        logits = model(X.float())
        loss_sum += criterion(logits, y).item() * y.numel()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()

        # confusion
        y_cpu = y.detach().cpu().numpy()
        p_cpu = preds.detach().cpu().numpy()
        for yt, pt in zip(y_cpu, p_cpu):
            if 0 <= yt < nclasses and 0 <= pt < nclasses:
                cm[yt, pt] += 1

    acc = 100.0 * correct / max(1, total)
    avg_loss = loss_sum / max(1, total)

    # Pretty print
    print(f"\n=== EVAL SUMMARY ===")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Avg loss: {avg_loss:.6f}")
    print("Confusion matrix (counts):")
    print(cm)

    # Normalize rows to percentages (avoid divide by zero)
    cm_pct = cm.copy()
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_pct = cm_pct / row_sums * 100.0

    return acc, avg_loss, cm, cm_pct

def save_results(cm, cm_pct, labels, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    # CSVs
    csv_counts = os.path.join(outdir, f"{tag}_confusion_counts.csv")
    csv_pct    = os.path.join(outdir, f"{tag}_confusion_percent.csv")

    def write_csv(path, mat):
        with open(path, "w") as f:
            f.write("," + ",".join(labels) + "\n")
            for i, row in enumerate(mat):
                f.write(labels[i] + "," + ",".join(f"{v:.6f}" for v in row) + "\n")

    write_csv(csv_counts, cm)
    write_csv(csv_pct, cm_pct)
    print(f"[ok] wrote {csv_counts}")
    print(f"[ok] wrote {csv_pct}")

    # Plots (PNG + PDF)
    for suffix in ["png", "pdf"]:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm_pct, interpolation="nearest")
        ax.set_title("Confusion Matrix (%)")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # annotate
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(outdir, f"{tag}_confusion.{suffix}")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] wrote {out_path}")

def main():
    p = argparse.ArgumentParser("Fixed validation evaluator (single set)")
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--val_root",   required=True, type=str, help="Root folder of validation set (class subdirs).")
    p.add_argument("--datasets",   nargs="+", required=True, help="Class names present in val set, e.g., AIR MP1 ...")
    p.add_argument("--transformer_version", choices=["v1","v2"], default="v1")
    p.add_argument("--transformer", choices=["sm","lg"], default="sm")
    p.add_argument("--rmsnorm", action="store_true")
    p.add_argument("--back_class", action="store_true", help="If training used a 'noise' back class at the end")
    p.add_argument("--testing_mode", choices=["future","random_sampling"], default="future")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="./results_eval_fixed")
    args = p.parse_args()

    device = torch.device("cuda:0") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    # Build label list (optionally include 'noise' class for alignment with trained heads)
    local_classes = list(args.datasets)
    if args.back_class and "noise" not in local_classes:
        local_classes.append("noise")
    nclasses = len(local_classes)
    print(f"[INFO] classes (order): {local_classes}")

    # Build dataset (fixed val)
    print(f"[INFO] building dataset for root={args.val_root} with classes={local_classes}")
    # SM v1 shapes you standardized on
    seq_len_used   = 64
    slice_len_used = 128

    base_kwargs = dict(
        protocols=local_classes,
        ds_path=args.val_root,
        ds_type="val",
        seq_len=seq_len_used,
        slice_len=slice_len_used,
        slice_overlap_ratio=0,
        test_ratio=0.0,
        testing_mode=args.testing_mode,
        # these will be dropped if not supported in this repo's dataset
        back_class=args.back_class,
        ota_dataset=None,
    )
    ds_val = TPrimeDataset_Transformer(**filter_ds_kwargs(TPrimeDataset_Transformer, **base_kwargs))
    print(f"[INFO] dataset size = {len(ds_val)}")

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Load checkpoint and infer dimensions
    sd = flex_load_checkpoint(args.model_path)
    d_model_ckpt, out_ckpt = infer_ckpt_dims(sd)
    if d_model_ckpt is None:
        # fallback to common value you’ve used
        d_model_ckpt = 256 if args.transformer == "lg" else 128
    print(f"[info] checkpoint inferred d_model = {d_model_ckpt}")
    if out_ckpt is not None:
        print(f"[info] checkpoint output classes = {out_ckpt} (expected {nclasses})")

    # Build model to match checkpoint dims and desired classes
    model = build_model(args, d_model=d_model_ckpt, nclasses=nclasses, device=device)
    apply_state_dict(model, sd)

    # Optional RMSNorm preprocessing
    rms_layer = make_rms_layer(args, d_model_ckpt, device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Eval
    acc, avg_loss, cm, cm_pct = evaluate(
        model, criterion, dl_val, nclasses, device, rms_layer=rms_layer, class_order=local_classes
    )

    # Save outputs
    tag = os.path.splitext(os.path.basename(args.model_path))[0] + "_on_" + os.path.basename(args.val_root.rstrip("/")).replace("/", "_")
    save_results(cm, cm_pct, local_classes, args.outdir, tag)

if __name__ == "__main__":
    main()
