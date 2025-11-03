# preprocessing/eval_models.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# repo imports (your files)
from TPrime_dataset import TPrimeDataset_Transformer
from TPrime_transformer.model_transformer import TransformerModel  # small v1
from preprocessing.model_rmsnorm import RMSNorm

# ---------------------------- helpers ----------------------------

def load_state_dict_from_ckpt(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "module", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # maybe already a plain state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    raise RuntimeError(f"[FATAL] Unrecognized checkpoint format: {type(obj)}")

def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def build_dataset(val_root: Path, classes_wo_noise, seq_len=24, slice_len=64):


    ds = TPrimeDataset_Transformer(
        protocols=classes_wo_noise,
        ds_path=str(val_root),
        ds_type='test',
        seq_len=seq_len,
        slice_len=slice_len,
        slice_overlap_ratio=0,
        test_ratio=1.0,           # fixed eval set: use all files
        testing_mode='future',
        raw_data_ratio=1.0,       # no subsampling on eval
        override_gen_map=False,
        ota=True,
        apply_wchannel=None,
        apply_noise=False,
        file_postfix='.dat',
        transform=chan2sequence
    )
    return ds

def evaluate(model, device, loader, nclasses_global, rms_layer=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    cm = np.zeros((nclasses_global, nclasses_global), dtype=np.int64)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).float()
            if rms_layer is not None:
                X = rms_layer(X)
            y = y.to(device).long()
            logits = model(X)
            loss_sum += criterion(logits, y).item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            # fill confusion matrix (labels assumed 0..K-1 for present classes)
            for t, p in zip(y.view(-1), preds.view(-1)):
                if 0 <= t.item() < nclasses_global and 0 <= p.item() < nclasses_global:
                    cm[t.item(), p.item()] += 1

    acc = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss, cm

# ----------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple evaluator for small v1 transformer on fixed eval set.")
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--val_root", required=True, type=str)
    ap.add_argument("--datasets", nargs="+", required=True, help="Class folders to use (e.g., AIR MP1 ...)")
    ap.add_argument("--back_class", action="store_true", help="Append 'noise' as an extra head output.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="./results_eval_fixed")
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--slice_len", type=int, default=64)
    # model basics (small v1)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--rmsnorm", action="store_true", help="Apply training-time RMS normalization to inputs.")
    args = ap.parse_args()

    val_root = Path(args.val_root)
    classes_wo_noise = list(args.datasets)
    classes_global = classes_wo_noise + (["noise"] if args.back_class else [])
    nclasses_global = len(classes_global)

    print(f"[INFO] val_root={val_root}")
    print(f"[INFO] classes={classes_global}  (head outputs={nclasses_global} incl. noise={args.back_class})")

    # dataset
    ds = build_dataset(val_root, classes_wo_noise, seq_len=args.seq_len, slice_len=args.slice_len)

    # quick listing for sanity
    for c in classes_wo_noise:
        # your dataset already prints counts when generating map, but this helps too
        # silently ignore if directory missing, dataset ctor will have warned
        p = val_root / c
        if p.exists():
            n = len(list(p.glob("*_n37440.dat")))
            print(f"[DEBUG] Found {n:4d} files in {p}")

    print(f"[INFO] dataset size = {len(ds)} (seq_len={args.seq_len}, slice_len={args.slice_len})")
    if len(ds) == 0:
        raise SystemExit("[FATAL] dataset is empty. Check class names and *_n37440.dat under val_root.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.use_gpu)

    # device
    device = torch.device("cuda:0" if (args.use_gpu and torch.cuda.is_available()) else "cpu")

    # build model (small v1) with correct kwargs to avoid dropout=64 bug
    state = load_state_dict_from_ckpt(args.model_path)
    model = TransformerModel(classes=nclasses_global, d_model=64*2, seq_len=24, nlayers=2, use_pos=False).to(device)
    rms_layer = RMSNorm(model="Transformer").to(device) if args.rmsnorm else None


    # load weights (non-strict in case you trained with different head size)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", len(missing))
    if unexpected:
        print("[warn] unexpected keys:", len(unexpected))

    # eval
    acc, loss, cm = evaluate(model, device, loader, nclasses_global, rms_layer)
    print("\n=== EVAL SUMMARY ===")
    print(f"Model   : {args.model_path}")
    print(f"Root    : {val_root}")
    print(f"Classes : {classes_global}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Avg loss: {loss:.6f}")
    print("Confusion matrix (global axis):")
    np.set_printoptions(suppress=True)
    print(cm)

    # save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"CM_{val_root.name}.csv"
    header = ",".join(classes_global)
    np.savetxt(csv_path, cm, fmt="%d", delimiter=",", header=header, comments="")
    print(f"\nSaved: {csv_path}")

if __name__ == "__main__":
    main()
