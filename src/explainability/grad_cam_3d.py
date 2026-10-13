"""Grad-CAM and Grad-CAM++ for the 3D spatial stream of CAFN.

Backprop starts from the *fused* classification logit by default, so the
resulting activation map reflects which anatomical zones drove the
multimodal decision (not just the unimodal spatial head) -- gradients flow
from `logits_fused` back through the cross-attention fusion module into the
spatial encoder's last residual block.

v2 adds **Grad-CAM++** (Selvaraju et al., ICCV 2017): instead of weighting
each channel by the plain average of its gradients, second-order terms
re-weight pixels that contribute partially but repeatedly -- measurably
better localization when a class activates *multiple* distinct regions,
which is exactly the lesion-plus-atrophy picture of the pathologies here.

Both variants return a (B, D, H, W) map in [0, 1]; switch with
`method="gradcam" | "gradcam++"`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        method: str = "gradcam",
    ):
        if method not in ("gradcam", "gradcam++"):
            raise ValueError(f"method must be 'gradcam' or 'gradcam++', got {method!r}")
        self.model = model
        self.target_layer = target_layer or model.spatial_encoder.last_conv
        self.method = method
        self._activations = None
        self._gradients = None
        self._activations_requires_grad = False
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # Keep the raw tensor (not detached): Grad-CAM++ needs to
        # differentiate *through* it, and it already carries requires_grad
        # from the conv weights upstream.
        self._activations = out

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _backward_to_activations(self, selected: torch.Tensor):
        """Propagate `selected` down to the target activations.

        Returns (activations, first_gradients). For Grad-CAM++ the first
        gradients are created with create_graph=True so higher-order
        derivative terms can be formed; vanilla mode frees the graph
        immediately. `torch.autograd.grad` targets the captured (non-leaf)
        activation tensor directly -- it already requires grad through the
        upstream conv weights, so no manual flag flipping is needed.
        """
        create_graph = self.method == "gradcam++"
        first = torch.autograd.grad(
            selected,
            self._activations,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return self._activations.detach(), first.detach()

    def _gradcam_weights(self, activations: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        """Per-channel spatial weights (B, C, 1, 1, 1) for vanilla Grad-CAM."""
        return first.mean(dim=(2, 3, 4), keepdim=True)

    def _gradcam_pp_weights(self, activations: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        """Grad-CAM++ weights via second/third-order derivative estimates.

        Uses the standard second-derivative approximation of the original
        paper implemented by pytorch-grad-cam: alpha_k combines the first
        gradient, its square, and its cube with the activation magnitudes,
        emphasizing channels with many small contributions spread spatially.
        """
        second = first**2
        third = first**3
        # Sum over spatial dims per (B, C); broadcast back over space.
        act_sum = activations.sum(dim=(2, 3, 4), keepdim=True)
        numerator = 2.0 * second
        denominator = 2.0 * second + third * act_sum
        alpha = numerator / (denominator + 1e-8)
        weights = torch.relu(first) * alpha
        return weights.sum(dim=(2, 3, 4), keepdim=True)

    def generate(
        self,
        mri_volume: torch.Tensor,
        vep_signal: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        logits_key: str = "logits_fused",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cam, target_class): cam is (B, D, H, W) in [0, 1]."""
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(mri_volume, vep_signal)
        logits = outputs[logits_key]

        if target_class is None:
            target_class = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.full(
                (logits.shape[0],), target_class, dtype=torch.long, device=logits.device
            )

        selected = logits.gather(1, target_class.view(-1, 1)).sum()

        activations, first = self._backward_to_activations(selected)
        if self.method == "gradcam++":
            weights = self._gradcam_pp_weights(activations, first)
        else:
            weights = self._gradcam_weights(activations, first)

        self._activations = None
        self._gradients = None
        if was_training:
            self.model.train()

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (B, 1, d, h, w)

        target_size = mri_volume.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
        cam = cam.squeeze(1)  # (B, D, H, W)

        cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
        cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
        denom = (cam_max - cam_min).clamp(min=1e-8)
        cam = (cam - cam_min) / denom

        return cam.detach(), target_class.detach()

    def generate_with_fallback(
        self,
        mri_volume: torch.Tensor,
        vep_signal: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        logits_key: str = "logits_fused",
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Run the configured method, falling back to vanilla Grad-CAM if the
        higher-order backward pass is unsupported by the current backend
        (e.g. some MPS builds). Returns (cam, target_class, method_used)."""
        methods = ("gradcam",) if self.method == "gradcam" else (self.method, "gradcam")
        last_exc: Exception | None = None
        for method in methods:
            self.method = method
            try:
                cam, tc = self.generate(mri_volume, vep_signal, target_class, logits_key)
                return cam, tc, method
            except Exception as exc:  # backend may not support double backward
                last_exc = exc
        assert last_exc is not None
        raise last_exc


def compute_gradcam(
    model: nn.Module,
    mri_volume: torch.Tensor,
    vep_signal: torch.Tensor,
    target_class: int | torch.Tensor | None = None,
    method: str = "gradcam",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot convenience wrapper around GradCAM3D that cleans up its hooks."""
    cam_engine = GradCAM3D(model, method=method)
    try:
        return cam_engine.generate(mri_volume, vep_signal, target_class)
    finally:
        cam_engine.remove_hooks()


def plot_gradcam_slices(
    volume: np.ndarray,
    cam: np.ndarray,
    save_path: str,
    title: str = "Grad-CAM: Visual Cortex Activation",
) -> None:
    """Overlay a Grad-CAM heatmap on the mid axial/coronal/sagittal slices."""
    if volume.shape != cam.shape:
        raise ValueError(f"volume shape {volume.shape} != cam shape {cam.shape}")

    d, h, w = volume.shape
    slices = [
        (volume[d // 2, :, :], cam[d // 2, :, :], "Axial"),
        (volume[:, h // 2, :], cam[:, h // 2, :], "Coronal"),
        (volume[:, :, w // 2], cam[:, :, w // 2], "Sagittal"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (vol_slice, cam_slice, name) in zip(axes, slices):
        ax.imshow(vol_slice, cmap="gray")
        ax.imshow(cam_slice, cmap="jet", alpha=0.5)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
