"""Single-subject inference: raw MRI/VEP files -> clinical prediction report.

Two entry points:
- `predict_single(model, mri_path, vep_path, ...)`: runs preprocessing and
  the fused forward pass, returns probabilities + predicted class; optional
  explainability artifacts.
- `run(checkpoint_path, mri_path, vep_path, outdir)`: loads a trained
  checkpoint and writes prediction.json, cross_attention.png and
  gradcam.png -- a clinician-ready evidence bundle.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.config import CLASS_NAMES, DEFAULT_CONFIG


def _load_and_preprocess(mri_path: str, vep_path: str, fs: float | None, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    from src.preprocessing.mri_transforms import preprocess_mri_file
    from src.preprocessing.signal_cleaner import clean_vep_signal

    mri = preprocess_mri_file(mri_path, cfg.mri)
    raw_signal = np.loadtxt(vep_path, delimiter=",", skiprows=1)
    fs = fs or cfg.vep.sample_rate
    vep = clean_vep_signal(raw_signal, fs, cfg.vep)
    return (
        torch.from_numpy(mri).unsqueeze(0),            # (1, 1, D, H, W)
        torch.from_numpy(vep).unsqueeze(0).unsqueeze(0),  # (1, 1, L)
    )


def predict_single(
    model: torch.nn.Module,
    mri_path: str,
    vep_path: str,
    fs: float | None = None,
    cfg=DEFAULT_CONFIG,
    device: torch.device | None = None,
) -> dict:
    """Classify one subject from raw files. Returns a JSON-ready dict."""
    device = device or next(model.parameters()).device
    model.eval()
    mri, vep = _load_and_preprocess(mri_path, vep_path, fs, cfg)
    with torch.no_grad():
        outputs = model(mri.to(device), vep.to(device))
        probs = torch.softmax(outputs["logits_fused"].float(), dim=1)[0].cpu()

    predicted_class = int(probs.argmax().item())
    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class),
        "probabilities": {
            name: float(p) for name, p in zip(CLASS_NAMES, probs.tolist())
        },
        "mri_path": mri_path,
        "vep_path": vep_path,
    }


def _explain(model, mri: torch.Tensor, vep: torch.Tensor, outdir: str, cfg) -> dict:
    from src.explainability.grad_cam_3d import compute_gradcam, plot_gradcam_slices
    from src.explainability.plot_attention import plot_cross_attention

    device = next(model.parameters()).device
    cam, _ = compute_gradcam(model, mri.to(device), vep.to(device))
    plot_gradcam_slices(
        mri[0, 0].numpy(), cam[0].cpu().numpy(),
        os.path.join(outdir, "gradcam.png"),
    )
    with torch.no_grad():
        outputs = model(mri.to(device), vep.to(device))
    plot_cross_attention(outputs["attn_weights"], os.path.join(outdir, "cross_attention.png"))
    return {}


def run(
    checkpoint_path: str,
    mri_path: str,
    vep_path: str,
    outdir: str,
    fs: float | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint and write the full prediction + explainability bundle.

    Defaults to CPU: single-subject inference is fast there, and CPU has no
    missing-operator gaps (notably `adaptive_avg_pool3d` is unimplemented on
    some MPS builds). Pass `device` explicitly to override.
    """
    from src.train import load_full_checkpoint

    checkpoint = load_full_checkpoint(checkpoint_path)
    cfg = checkpoint.get("config") or DEFAULT_CONFIG

    from src.models.cafn import CAFN

    model = CAFN(checkpoint.get("model_config", cfg.model))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = device or torch.device("cpu")
    model.to(device)

    os.makedirs(outdir, exist_ok=True)
    result = predict_single(model, mri_path, vep_path, fs=fs, cfg=cfg, device=device)

    mri, vep = _load_and_preprocess(mri_path, vep_path, fs, cfg)
    _explain(model, mri, vep, outdir, cfg)

    result["checkpoint"] = checkpoint_path
    with open(os.path.join(outdir, "prediction.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="CAFN inference on one MRI+VEP pair: prediction + evidence bundle"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="trained checkpoint (best.pt / last.pt / any save_checkpoint file)")
    parser.add_argument("--mri", type=str, required=True, help="NIfTI volume (.nii / .nii.gz)")
    parser.add_argument("--vep", type=str, required=True,
                        help="VEP waveform CSV (single numeric column) or .npy")
    parser.add_argument("--fs", type=float, default=None,
                        help="VEP sampling rate in Hz (default: checkpoint config)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="writes prediction.json, cross_attention.png, gradcam.png here")
    parser.add_argument("--device", type=str, default=None,
                        help="force a device (default: CPU, no accelerator gaps)")
    args = parser.parse_args(argv)

    result = run(
        args.checkpoint, args.mri, args.vep,
        outdir=args.outdir, fs=args.fs, device=args.device,
    )
    print(f"Predicted: {result['predicted_label']}  "
          f"(P={result['probabilities'][result['predicted_label']]:.3f})")
    print(f"Artifacts written to {args.outdir}")
    return result


if __name__ == "__main__":
    main()
