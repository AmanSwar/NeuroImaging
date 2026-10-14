"""Central configuration for shapes, hyperparameters, and class labels.

Single source of truth so the preprocessing stream, both encoders, the
fusion module, and the training loop cannot silently disagree on shapes or
feature names. `merge_model_config` upgrades legacy/older checkpoints to the
current field set.

Every field has a default, so `dataclasses.replace`, explicit kwargs in
tests, and the CLI all keep working as fields evolve.
"""

from dataclasses import dataclass, field, fields, is_dataclass


CLASS_NAMES = ("Healthy Control", "Glaucoma", "Optic Neuritis / CVI")


@dataclass(frozen=True)
class MRIConfig:
    target_shape: tuple[int, int, int] = (64, 64, 64)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "RAS"
    foreground_percentile: float = 60.0  # intensity-threshold skull-strip approx
    # "percentile" (v1) or "otsu" (histogram-optimal threshold; recommended
    # for real MRI where tissue classes are separated).
    mask_method: str = "percentile"
    mask_morphology: bool = True         # connected-component cleanup for Otsu masks
    # "minmax" -> [0, 1] scaling (v1); "zscore" -> foreground ~N(0,1), bg zeroed.
    normalization: str = "minmax"
    # How the standardised volume is fitted to ``target_shape``:
    # "crop"        -> centre pad/crop (v1; fine for small synthetic volumes);
    # "bbox_resize" -> crop to the foreground bounding box, then resample to
    #                  ``target_shape``. Use for real whole-head MRI (~170 mm
    #                  at 1 mm), where a centre crop to 64^3 sees only a sliver.
    fit: str = "crop"


@dataclass(frozen=True)
class VEPConfig:
    length: int = 500
    sample_rate: float = 250.0  # Hz -> 500 samples spans 2.0s per ISCEV sweep
    bandpass_low: float = 1.0
    bandpass_high: float = 100.0
    notch_freq: float = 50.0
    notch_quality: float = 30.0
    # Baseline correction over the first `baseline_window_ms` of each sweep
    # (standard electrophysiology: subtract pre-stimulus mean). None disables
    # it; synthetic sweeps carry no true pre-stimulus interval.
    baseline_window_ms: float | None = None
    # "waveform": time-domain sweep -> baseline/bandpass/notch/resample/z-score.
    # "spectrum": the 1-D input is already a spectrum (e.g. a log-PSD from
    # resting MEG/EEG, see src/data/cohorts.py); filtering would be wrong, so
    # only resample-to-length + z-score are applied.
    signal_mode: str = "waveform"


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    spatial_tokens: int = 64   # 4x4x4 adaptive pool
    temporal_tokens: int = 16
    spatial_pool: tuple[int, int, int] = (4, 4, 4)
    ffn_hidden: int = 256
    dropout: float = 0.1
    num_classes: int = len(CLASS_NAMES)
    # --- v2 ---
    fusion: str = "bidirectional"         # label for the fusion strategy (CLI/docs)
    pos_encoding: bool = True             # learned spatial / sinusoidal temporal embeddings
    pooling: str = "attn"                 # "attn" (learned pool) or "mean" (v1)
    modality_dropout: float = 0.0         # train-time p of dropping a sample's VEP stream
    fusion_blocks: int = 1                # stacked pre-norm co-attention blocks
    se_reduction: int | None = 4          # Squeeze-Excitation reduction in spatial blocks; None off

    @property
    def bidirectional(self) -> bool:
        """Back-compat alias for the v1 boolean flag name."""
        return self.fusion == "bidirectional"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    aux_loss_weight: float = 0.3
    val_split: float = 0.2
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    # --- v2 ---
    optimizer: str = "adamw"              # reserved for optimizer selection
    scheduler: str = "cosine"             # linear-warmup -> cosine decay | "none"
    warmup_fraction: float = 0.1          # fraction of total steps: linear LR warmup
    # explicit warmup length in epochs; when set it overrides warmup_fraction
    warmup_epochs: float | None = None
    max_grad_norm: float = 1.0            # 0 disables gradient clipping
    patience: int | None = None           # early-stopping patience; None disables
    num_workers: int = 0
    amp: bool = False                     # mixed precision (CUDA only)
    balance_classes: bool = False         # inverse-frequency CE weights for imbalanced data
    # explicit class-weight policy ("auto" = inverse frequency); alternative
    # to the `balance_classes` boolean
    class_weights: str | None = None
    label_smoothing: float = 0.0          # CE label smoothing on all heads
    deterministic: bool = False           # seed_everything(deterministic=True)
    # Which validation quantity picks the restored "best" checkpoint:
    # "val_loss" (default, v2 behaviour) | "val_balanced_accuracy" | "val_macro_f1".
    # The accuracy-style options break ties on val loss. Prefer them on small
    # cohorts: an early, untrained epoch can have the lowest val loss.
    select_metric: str = "val_loss"
    augment: bool = True                  # train-time augmentation (val always clean)
    device: str | None = None             # explicit device override; None = auto

    @property
    def grad_clip_norm(self) -> float:
        """Alias for `max_grad_norm` (older call sites used this name)."""
        return self.max_grad_norm


@dataclass(frozen=True)
class AugmentConfig:
    """Train-time augmentation knobs (see src/data/augment.py).

    Augmentation never destroys class-defining markers: MRI gets flips,
    intensity jitter and noise; VEP gets amplitude scale and noise only --
    the waveform is never time-shifted because P100 latency carries the
    class signal. `enabled` gates every transform; dataset/train opt in
    explicitly, so default-on here changes nothing for existing callers.
    """

    enabled: bool = True
    mri_flip_prob: float = 0.5
    mri_intensity_scale: tuple[float, float] = (0.95, 1.05)
    mri_intensity_shift: float = 0.02
    mri_noise_std: float = 0.01
    vep_amplitude_scale: tuple[float, float] = (0.9, 1.1)
    vep_noise_std: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.mri_flip_prob <= 1.0:
            raise ValueError("mri_flip_prob must be in [0, 1]")
        if not (0.0 <= self.mri_intensity_scale[0] <= self.mri_intensity_scale[1]):
            raise ValueError("mri_intensity_scale must satisfy 0 <= lo <= hi")
        for std in (self.mri_noise_std, self.vep_noise_std):
            if std < 0:
                raise ValueError("noise std must be >= 0")
        if not (0.0 <= self.vep_amplitude_scale[0] <= self.vep_amplitude_scale[1]):
            raise ValueError("vep_amplitude_scale must satisfy 0 <= lo <= hi")


@dataclass(frozen=True)
class Config:
    mri: MRIConfig = field(default_factory=MRIConfig)
    vep: VEPConfig = field(default_factory=VEPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


DEFAULT_CONFIG = Config()


def merge_model_config(legacy: object | None, **overrides) -> ModelConfig:
    """Build a current `ModelConfig` from a stored/partial config.

    Accepts a ModelConfig (any vintage), a plain dict, or None, plus explicit
    overrides. Legacy boolean/renamed fields are mapped forward:
    `bidirectional` -> `fusion`, `use_positional_encoding` -> `pos_encoding`.
    Unknown fields are ignored so old checkpoints always load.
    """
    if legacy is None:
        legacy = {}
    if isinstance(legacy, dict):
        raw = dict(legacy)
    elif is_dataclass(legacy):
        raw = {f.name: getattr(legacy, f.name) for f in fields(legacy)}
    else:
        # Plain object (e.g. a legacy stub from an old checkpoint)
        raw = dict(vars(legacy))

    if "fusion" not in raw and "bidirectional" in raw:
        raw["fusion"] = "bidirectional" if raw.pop("bidirectional") else "cross"
    else:
        raw.pop("bidirectional", None)
    if "fusion_blocks" not in raw and "num_fusion_layers" in raw:
        raw["fusion_blocks"] = raw.pop("num_fusion_layers")

    current = {f.name: f.default for f in fields(ModelConfig)}

    merged: dict = {}
    for name, value in raw.items():
        if name == "use_positional_encoding":
            merged["pos_encoding"] = bool(value)
        elif name in current:
            merged[name] = value
    merged.update(overrides)

    invalid = set(merged) - set(current)
    if invalid:
        raise ValueError(f"Unknown ModelConfig field(s): {sorted(invalid)}")
    return ModelConfig(**merged)
