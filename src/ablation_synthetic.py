"""Controlled synthetic study: fusion strategies and component ablations.

The synthetic generator lets us set how informative each modality is
(``mri_strength`` / ``vep_strength`` in [0, 1]); this study asks *when does
fusing help* with everything else fixed, which real cohorts this small cannot
answer. Every (strength, seed) draws its own train/val/test sets; every model
variant sees identical data. Checkpoints are chosen on validation balanced
accuracy; all numbers are on a separate test set. Runs are resumable.

    python -m src.ablation_synthetic --out results/synth --settings 1.0:0.35 --seeds 0 1 2 3 4
    python -m src.ablation_synthetic --pilot --out results/synth_pilot     # pick a difficulty
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.config import Config, MRIConfig, ModelConfig, TrainConfig
from src.data.dataset import SyntheticNeuroDataset
from src.models.cafn import CAFN
from src.train import _verify_device_supports_model, train_model
from src.utils import auto_device, count_parameters, seed_everything

RAW_SHAPE = (48, 56, 40)       # synthetic raw volume (kept small: this is a controlled benchmark)
N_TRAIN, N_VAL, N_TEST = 144, 48, 120
SEED_BASE = 1000

# name -> (ModelConfig overrides, aux_loss_weight)
VARIANTS: dict[str, tuple[dict, float]] = {
    "CAFN (full)": ({}, 0.3),
    "w/o bidirectional": ({"fusion": "cross"}, 0.3),
    "w/o positional enc.": ({"pos_encoding": False}, 0.3),
    "w/o attention pooling": ({"pooling": "mean"}, 0.3),
    "w/o SE blocks": ({"se_reduction": None}, 0.3),
    "w/o auxiliary loss": ({}, 0.0),
    "2 fusion blocks": ({"fusion_blocks": 2}, 0.3),
    "v1-style (all off)": ({"fusion": "cross", "pos_encoding": False, "pooling": "mean", "se_reduction": None}, 0.3),
    "Concatenation": ({"fusion": "concat"}, 0.3),
    "Element-wise sum": ({"fusion": "sum"}, 0.3),
    "MRI only": ({"fusion": "mri_only"}, 0.3),
    "VEP only": ({"fusion": "vep_only"}, 0.3),
}


def make_config(overrides: dict, aux: float, args) -> Config:
    return Config(
        # percentile mask + min-max + centre crop: on this generator the Otsu/z-score pipeline erases the lesion contrast
        # (MRI-only drops to chance), so the synthetic study uses the original v1 MRI preprocessing.
        mri=MRIConfig(target_shape=(32, 32, 32), mask_method="percentile", normalization="minmax", fit="crop"),
        model=ModelConfig(num_classes=3, **overrides),
        train=TrainConfig(
            batch_size=args.batch_size, epochs=args.epochs, lr=1e-3, weight_decay=1e-4, aux_loss_weight=aux,
            patience=args.patience, seed=SEED_BASE, augment=True, select_metric="val_balanced_accuracy",
            amp=True, deterministic=True, checkpoint_dir=args.out, num_workers=0,
        ),
    )


@torch.no_grad()
def test_metrics(model, subset, device) -> dict:
    model.eval()
    ys, ps = [], {"fused": [], "spatial": [], "temporal": []}
    for b in DataLoader(subset, batch_size=16):
        o = model(b["mri"].to(device), b["vep"].to(device))
        for h in ps:
            ps[h].append(o[f"logits_{h}"].argmax(1).cpu().numpy())
        ys.append(b["label"].numpy())
    y = np.concatenate(ys)
    out = {}
    for h, v in ps.items():
        p = np.concatenate(v)
        out[h] = dict(accuracy=float((p == y).mean()), balanced_accuracy=float(balanced_accuracy_score(y, p)),
                      macro_f1=float(f1_score(y, p, average="macro", zero_division=0)))
    return out


def run(args, log=print):
    base_cfg = make_config({}, 0.3, args)
    device = _verify_device_supports_model(auto_device(args.device), base_cfg)
    log(f"device={device}")
    os.makedirs(args.out, exist_ok=True)
    names = args.variants or list(VARIANTS)
    for setting in args.settings:
        mri_s, vep_s = (float(x) for x in setting.split(":"))
        strength = f"{mri_s}-{vep_s}"
        for seed in args.seeds:
            todo = [n for n in names if not os.path.exists(_path(args.out, strength, seed, n))]
            if not todo:
                continue
            t0 = time.time()
            mk = lambda n, sd: SyntheticNeuroDataset(
                n, cfg=base_cfg, seed=sd, raw_mri_shape=RAW_SHAPE, mri_strength=mri_s, vep_strength=vep_s)
            trainval = mk(N_TRAIN + N_VAL, SEED_BASE + seed)
            test = mk(N_TEST, SEED_BASE + 500 + seed)
            for i in range(len(trainval)):
                trainval[i]
            for i in range(len(test)):
                test[i]
            log(f"[strength={strength} seed={seed}] data ready in {time.time() - t0:.0f}s")
            tr, va = list(range(N_TRAIN)), list(range(N_TRAIN, N_TRAIN + N_VAL))
            for name in todo:
                over, aux = VARIANTS[name]
                cfg = make_config(over, aux, args)
                cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=SEED_BASE + seed))
                seed_everything(SEED_BASE + seed, deterministic=True)
                model = CAFN(cfg.model)
                t1 = time.time()
                hist = train_model(model, Subset(trainval, tr), Subset(trainval, va), cfg=cfg, device=device, augment=True)
                res = dict(variant=name, mri_strength=mri_s, vep_strength=vep_s, strength=strength, seed=seed, params=count_parameters(model),
                           epochs_run=hist["epochs_run"], minutes=round((time.time() - t1) / 60, 2),
                           test=test_metrics(model, test, device),
                           best_val_bacc=float(max(hist["val_balanced_accuracy"])))
                with open(_path(args.out, strength, seed, name), "w") as f:
                    json.dump(res, f)
                log(f"  {name:<24} s={strength} seed={seed}  test acc={res['test']['fused']['accuracy']:.3f}  "
                    f"({res['epochs_run']} ep, {res['minutes']} min)")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def _path(out, strength, seed, name) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    return os.path.join(out, f"s{strength}_seed{seed}_{safe}.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/synth")
    ap.add_argument("--settings", nargs="+", default=["1.0:0.35"],
                    help="modality informativeness as mri:vep pairs in [0,1] (1.0 = default generator)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pilot", action="store_true", help="unimodal models over a strength grid, 1 seed")
    args = ap.parse_args(argv)
    if args.pilot:
        args.settings, args.seeds, args.variants = ["1.0:0.15", "1.0:0.25", "1.0:0.35", "1.0:0.5", "1.0:1.0"], [0], ["MRI only", "VEP only"]
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
