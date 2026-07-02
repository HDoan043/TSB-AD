"""
DualStreamAD
============
Hybrid CNN-BiLSTM anomaly detector for multivariate time-series, combining
context-aware reconstruction, forecasting, latent-density estimation,
frequency-domain consistency, adversarial cross-cycle reconstruction,
idempotent reconstruction, and residual-flow likelihood scoring.

The model first applies global z-score normalization and per-window RevIN,
then encodes each sliding window with a multi-scale CNN encoder
(kernels 3, 5, 7) followed by a bidirectional LSTM. The resulting temporal
hidden states are decoded for reconstruction and forecasting, while the
mean-pooled latent vector z is used for density estimation, frequency
reconstruction, contextual score fusion, and auxiliary self-supervised losses.

Main scoring branches
---------------------
  1. Reconstruction
     Masked timestep reconstruction error from the primary Conv1d decoder.
     Random masking during training prevents the model from learning a trivial
     identity mapping and forces contextual inference.

  2. Prediction
     Multi-horizon next-step forecasting error with γ^h horizon discount,
     so nearer prediction steps contribute more strongly than farther ones.

  3. Latent RBF density
     RESTAD-inspired non-parametric latent density using learnable RBF centers.
     Low maximum RBF response means the window is far from normal latent modes.

  4. Adversarial cross-cycle
     A second decoder reconstructs the input through an adversarial/cross-cycle
     path. Large disagreement or cross-cycle reconstruction error indicates
     that the window is unstable under the learned normal manifold.

  5. Frequency consistency
     Combines banded FFT reconstruction error with masked frequency-spectrum
     reconstruction. This improves sensitivity to oscillatory anomalies,
     spectral drift, and patterns that may be subtle in the time domain.

  6. Idempotent reconstruction
     IGAD-inspired re-reconstruction gap: recon1 is passed through the same
     encoder-decoder again. Normal windows should remain stable, while
     anomalous windows tend to drift away from the learned manifold.

  7. Residual Flow NLL
     A small post-hoc RealNVP model is fitted on multi-branch residual vectors
     from normal training windows. At inference, high negative log-likelihood
     means the joint error pattern is unlikely under normal behavior.

Training objectives
-------------------
The network is trained with a weighted combination of:
  • masked reconstruction loss
  • multi-horizon prediction loss
  • banded FFT alignment loss
  • masked frequency reconstruction loss
  • RBF commitment loss
  • idempotent reconstruction loss
  • temporal/frequency contrastive loss
  • gate-guidance KL loss
  • optional adversarial dual-decoder loss

Post-training calibration
-------------------------
After the main model is trained:
  • RBF centers are initialized with KMeans++ on training latents.
  • A RealNVP residual flow is fitted on training residual vectors.
  • Channel-wise reconstruction statistics are stored for z-normalized scoring.
  • Training score distributions are stored for inference-time normalization.

Inference
---------
For each timestep, overlapping window-level scores are accumulated back to the
time axis, gap-filled, z-normalized against training statistics, smoothed, and
robustly normalized.

Final anomaly score:
  • reconstruction score is blended with the idempotent gap
  • reconstruction, prediction, latent/RBF, and adversarial scores are fused
    by a learned contextual 4-branch gate
  • frequency score is added as a spectral anomaly bonus
  • residual-flow NLL is blended into the final score as a joint residual
    likelihood penalty

The output is a per-timestep anomaly score where higher values indicate more
abnormal behavior.
"""

import os
import sys
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset

from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset


def _silent_get_gpu(**kwargs):
    """Call get_gpu() without printing device info to stdout."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        device = get_gpu(**kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return device


# ── Building blocks ────────────────────────────────────────────────────────────

class _MultiScaleCNN(nn.Module):
    """Parallel Conv1d with kernel sizes 3, 5, 7 – outputs are concatenated
    then projected back to *hidden_dim* channels."""

    def __init__(self, in_ch: int, hidden_dim: int):
        super().__init__()
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, hidden_dim, 3, padding=1), nn.GELU())
        self.b5 = nn.Sequential(nn.Conv1d(in_ch, hidden_dim, 5, padding=2), nn.GELU())
        self.b7 = nn.Sequential(nn.Conv1d(in_ch, hidden_dim, 7, padding=3), nn.GELU())
        self.proj = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)

    def forward(self, x):          # x: (B, in_ch, W)
        cat = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.proj(cat)      # (B, hidden_dim, W)


class _ResBlock(nn.Module):
    """Residual Conv1d block with skip connection."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1),
        )

    def forward(self, x):          # x: (B, ch, W)
        return x + self.block(x)   # skip connection


class _RevIN(nn.Module):
    """Reversible Instance Normalization (per-sample, per-feature).

    Normalizes each input window by its own mean/std, then denormalizes the
    output – corrects distribution shift between train and test windows.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:    # (B, W, F)
        self._mean = x.mean(dim=1, keepdim=True)              # (B, 1, F)
        self._std  = x.std(dim=1, keepdim=True) + self.eps    # (B, 1, F)
        return (x - self._mean) / self._std * self.weight + self.bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:   # (B, *, F)
        return (x - self.bias) / (self.weight + self.eps) * self._std + self._mean


# ── Frequency-domain masked reconstruction ─────────────────────────────────────

class _FreqMaskedRecon(nn.Module):
    """Lightweight frequency-domain reconstruction: BiLSTM latent → FFT magnitude.

    During training, random frequency bands are masked in the target spectrum;
    the head must reconstruct them from the latent context alone.  This is
    complementary to time-domain masked reconstruction — anomalies that are
    subtle in time (e.g. oscillation drift) produce large frequency errors.
    """

    def __init__(self, latent_dim: int, n_freq: int, hidden_dim: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_freq),
        )

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """h_pooled: (B, latent_dim) → predicted magnitude: (B, n_freq)."""
        return F.softplus(self.net(h_pooled))  # magnitude must be ≥ 0


# ── RBF Density Head (RESTAD-inspired) ────────────────────────────────────────

class _RBFDensityHead(nn.Module):
    """Non-parametric density estimation via learnable RBF neurons.

    Replaces single-Gaussian Mahalanobis (too rigid for multi-modal telecom
    data).  K learnable RBF centers with per-center bandwidths model arbitrary
    normal-latent distributions.
    """

    def __init__(self, latent_dim: int, n_centers: int = 32):
        super().__init__()
        self.n_centers = n_centers
        self.centers = nn.Parameter(torch.randn(n_centers, latent_dim) * 0.1)
        self.log_sigma = nn.Parameter(torch.zeros(n_centers))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) → density: (B,) — high = normal, low = anomalous."""
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)   # (B, K, D)
        dist_sq = (diff * diff).sum(dim=-1)                  # (B, K)
        sigma_sq = (F.softplus(self.log_sigma) + 1e-4).unsqueeze(0) ** 2
        rbf = torch.exp(-dist_sq / (2. * sigma_sq))          # (B, K)
        return rbf.max(dim=1).values                          # (B,)

    def init_centers(self, latents: np.ndarray):
        """K-means++ init from training latents (N, D)."""
        from sklearn.cluster import KMeans
        n = min(len(latents), 5000)
        idx = np.random.choice(len(latents), n, replace=False)
        km = KMeans(n_clusters=self.n_centers, n_init=3,
                    random_state=42).fit(latents[idx])
        with torch.no_grad():
            self.centers.copy_(torch.tensor(
                km.cluster_centers_, dtype=torch.float32))
            labels = km.labels_
            log_sigma_vals = self.log_sigma.cpu().numpy().copy()
            for k in range(self.n_centers):
                mask = labels == k
                if mask.sum() > 1:
                    d = np.linalg.norm(
                        latents[idx][mask] - km.cluster_centers_[k], axis=1).mean()
                    log_sigma_vals[k] = np.log(max(d, 0.01))
            self.log_sigma.copy_(torch.tensor(
                log_sigma_vals, dtype=torch.float32))


class _ResidualFlow:
    """Small RealNVP over multi-branch residual vector (DBR-AF-inspired).

    Learns the joint distribution of per-feature recon errors + pred error
    + freq band error + latent RBF score + adv score + idem score.
    Anomalies produce residual vectors far from the learned normal manifold
    → high negative log-likelihood (NLL) — no linear score fusion needed.

    The flow is a separate tiny model trained post-hoc on training residuals;
    zero overhead during main DualStreamAD training.
    """

    class _Coupling(nn.Module):
        """Affine coupling: z_a unchanged, z_b' = z_b * exp(s(z_a)) + t(z_a)."""

        def __init__(self, dim: int, hidden_dim: int, reverse: bool = False):
            super().__init__()
            self.reverse = reverse
            self.dim_a = dim // 2
            self.dim_b = dim - self.dim_a
            self.net = nn.Sequential(
                nn.Linear(self.dim_a, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.dim_b * 2),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.reverse:
                x_a, x_b = x[:, self.dim_b:], x[:, :self.dim_b]
            else:
                x_a, x_b = x[:, :self.dim_a], x[:, self.dim_a:]
            st = self.net(x_a)                # (B, dim_b*2)
            s, t = st[:, :self.dim_b], st[:, self.dim_b:]
            s = torch.tanh(s)                 # bound scale for stability
            x_b = x_b * torch.exp(s) + t
            if self.reverse:
                x_out = torch.cat([x_b, x_a], dim=1)
            else:
                x_out = torch.cat([x_a, x_b], dim=1)
            return x_out, s.sum(dim=1)        # log-det = sum of s

    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 32):
        self.dim = dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(n_layers):
            layers.append(self._Coupling(dim, hidden_dim, reverse=(i % 2 == 1)))
        self.model = nn.Sequential(*layers)
        self._log2pi_k = 0.5 * dim * np.log(2 * np.pi)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood. (B, D) → (B,)."""
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.model:
            z, ld = layer(z)
            log_det = log_det + ld
        return 0.5 * (z * z).sum(dim=1) + self._log2pi_k - log_det

    def fit(self, data: np.ndarray, device: torch.device,
            lr: float = 1e-3, epochs: int = 200, batch_size: int = 2048):
        """Fit RealNVP via MLE on training residual vectors."""
        self.to(device)
        self.train()
        t = torch.tensor(data, dtype=torch.float32, device=device)
        n = len(t)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=device)
            total = 0.0
            count = 0
            for b in range(0, n, batch_size):
                idx = perm[b:b + batch_size]
                loss = self.nll(t[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
                count += len(idx)
            scheduler.step()
            avg = total / count
            if avg < best_loss:
                best_loss = avg
            if epoch % 50 == 0:
                print(f'  [ResidualFlow] epoch {epoch:3d}  NLL={avg:.4f}')
        self.eval()


class DualStreamNet(nn.Module):
    """Encoder-decoder network with reconstruction + prediction heads."""

    def __init__(self, feats: int, window_size: int,
                 hidden_dim: int = 64, lstm_hidden: int = 32,
                 num_layers: int = 2, pred_len: int = 1,
                 rbf_n_centers: int = 32):
        super().__init__()
        self.feats = feats
        self.window_size = window_size
        self.pred_len = pred_len

        latent_dim = lstm_hidden * 2   # BiLSTM is bidirectional

        # ── RevIN – reversible instance normalization ─────────────────────────
        self.revin = _RevIN(feats)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.cnn_enc = nn.Sequential(
            _MultiScaleCNN(feats, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=0.1)

        # ── Reconstruction decoder (ResBlocks → sharper normal reconstruction) ─
        self.recon_dec = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 1),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            nn.Conv1d(hidden_dim, feats, 1),
        )

        # ── Adversarial decoder ────────────────────────────────────────────────
        self.adv_dec = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 1),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            nn.Conv1d(hidden_dim, feats, 1),
        )
        self.revin_cross = _RevIN(feats)  # separate RevIN for cross-cycle path

        # ── Prediction head (uses last BiLSTM time-step) ─────────────────────
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, feats * pred_len),
        )

        # ── Fusion gate: latent context → per-window branch weights ──────────
        self.fusion_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 4),  # recon, pred, latent, adversarial
        )

        # ── Masked-reconstruction token (learnable) ──────────────────────────
        # Replaces masked timesteps after RevIN; trained to signal "missing".
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feats))

        # ── Frequency-domain masked reconstruction head ──────────────────────
        # Lightweight MLP: BiLSTM latent → predicted FFT magnitude spectrum.
        # During training, random frequency bands are masked and the head
        # must reconstruct them — complementary to time-domain masking.
        n_freq = window_size // 2 + 1
        self.freq_recon_head = _FreqMaskedRecon(latent_dim, n_freq)

        # ── RBF density head (RESTAD-inspired) ───────────────────────────────
        self.rbf_head = _RBFDensityHead(latent_dim, n_centers=rbf_n_centers)

    def forward(self, x, mask=None,
                need_adv: bool = False, need_pred: bool = True,
                return_latent: bool = False):
        """x: (B, W, F), mask: (B, W, 1) or None.

        need_adv      – compute adversarial decoder + cross-cycle encoder pass.
                        Skip when adv_weight=0 (default) to save ~35% compute.
        need_pred     – compute prediction head.  Skip for idempotent-only calls.
        return_latent – return pooled latent z alongside outputs.
                        Saves re-encoding in _accumulate_scores().
        """
        x_n = self.revin.normalize(x)                # (B, W, F) – saves x stats

        # Apply random timestep mask (training only — inference passes mask=None)
        if mask is not None:
            x_n = x_n * (1. - mask) + self.mask_token * mask

        h = x_n.transpose(1, 2)     # (B, F, W)
        h = self.cnn_enc(h)       # (B, hidden_dim, W)
        h = h.transpose(1, 2)     # (B, W, hidden_dim)
        h, _ = self.bilstm(h)     # (B, W, latent_dim)
        z = h.mean(dim=1)         # (B, latent_dim) – pooled latent for RBF/freq/gate

        # Decoder 1 — reconstruction (always computed)
        recon1 = self.recon_dec(h.transpose(1, 2))    # (B, F, W)
        recon1 = self.revin.denormalize(recon1.transpose(1, 2))  # (B, W, F)

        recon2 = None
        recon21 = None
        if need_adv:
            # Decoder 2 — adversarial
            recon2 = self.adv_dec(h.transpose(1, 2))      # (B, F, W)
            recon2 = self.revin.denormalize(recon2.transpose(1, 2))  # (B, W, F)

            # Cross-cycle consistency: adv_dec(encoder(recon1))
            recon1_n = self.revin_cross.normalize(recon1.detach())
            h1 = recon1_n.transpose(1, 2)
            h1 = self.cnn_enc(h1)
            h1 = h1.transpose(1, 2)
            h1, _ = self.bilstm(h1)
            recon21 = self.adv_dec(h1.transpose(1, 2))    # (B, F, W)
            recon21 = self.revin_cross.denormalize(recon21.transpose(1, 2))  # (B, W, F)

        pred = None
        if need_pred:
            pred = self.pred_head(h[:, -1, :])            # (B, F*pred_len)
            pred = self.revin.denormalize(
                pred.view(-1, self.pred_len, self.feats))  # (B, pred_len, F)

        if return_latent:
            return recon1, recon2, recon21, pred, z
        return recon1, recon2, recon21, pred

    def get_latent(self, x):          # x: (B, W, F) → (B, latent_dim)
        x_n = self.revin.normalize(x)
        h = x_n.transpose(1, 2)
        h = self.cnn_enc(h)
        h = h.transpose(1, 2)
        h, _ = self.bilstm(h)
        return h.mean(dim=1)

    def get_gate_weights(self, x):    # x: (B, W, F) → (B, 4)
        """Per-window branch weights derived from latent context."""
        return F.softmax(self.fusion_gate(self.get_latent(x)), dim=1)

    def get_freq_recon(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """Predict FFT magnitude spectrum from pooled latent. (B, D) → (B, n_freq)."""
        return self.freq_recon_head(h_pooled)

    def get_freq_recon_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window freq recon error for inference scoring. (B, W, F) → (B,)."""
        h = self.get_latent(x)                              # (B, D)
        pred_freq = self.freq_recon_head(h)                 # (B, n_freq)
        actual_freq = torch.fft.rfft(
            self.revin.normalize(x), dim=1).abs().mean(dim=2)  # (B, n_freq)
        return (pred_freq - actual_freq).pow(2).mean(dim=1)    # (B,)

    def get_rbf_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window RBF anomaly score: 1 − density. (B, W, F) → (B,)."""
        h = self.get_latent(x)                              # (B, D)
        density = self.rbf_head(h)                           # (B,) ∈ [0,1]
        return 1.0 - density                                 # (B,)


# ── Anomaly Detector ───────────────────────────────────────────────────────────

class DualStreamAD:
    """
    Hybrid CNN-BiLSTM anomaly detector trained with reconstruction,
    prediction and frequency-domain objectives.

    Parameters
    ----------
    window_size     : sliding window length
    pred_len        : prediction horizon (keep 1 for single-step)
    feats           : number of input channels
    hidden_dim      : CNN hidden channels
    lstm_hidden     : per-direction LSTM hidden size (latent = 2×)
    num_layers      : number of BiLSTM layers
    lr              : Adam learning rate
    alpha           : weight of prediction loss during training
    freq_weight     : weight of frequency-domain alignment loss
    temp_neighbor_weight : weight of temporal neighbor contrastive loss
                           (0.0 = disabled; uses windows δ apart as positives)
    temp_neighbor_delta  : temporal offset for positive pairs (default 2)
    temp_exclude_radius  : exclusion zone for contrastive negatives.
                           Windows within ±radius of the anchor are NOT
                           treated as negatives — prevents nearby (similar)
                           windows from polluting the contrastive signal.
                           Default = 2×delta (only the positive pair itself
                           is kept as the sole nearby window).
    adv_weight      : weight of adversarial dual-decoder loss
                      (0.0 = disabled) — adds a second decoder with
                      cross-cycle consistency scoring
    mask_ratio      : fraction of timesteps randomly masked during
                      training (0.3 = 30%).  Forces BiLSTM to infer
                      rather than copy — breaks the identity shortcut
                      that makes reconstruction trivial.
    mask_weight     : weight on masked-position MSE vs full-window MSE
                      (0.7 = 70% masked, 30% full).  Higher → model
                      relies more on context than identity.
    freq_mask_weight: weight of frequency-domain masked reconstruction
                      loss (0.03).  TSPulse-style — mask random freq
                      bands and reconstruct from latent.
    freq_mask_bands : number of log-spaced frequency bands for masking
                      (8).  More bands = finer-grained masking.
    rbf_n_centers   : number of RBF centers for latent density (32).
                      RESTAD-inspired — replaces single-Gaussian
                      Mahalanobis with non-parametric multi-modal fit.
    rbf_weight      : weight of RBF commitment loss (0.01).
    idem_weight     : weight of idempotent reconstruction loss (0.1).
                      IGAD-inspired — re-reconstruct recon1 and penalise
                      deviation. Normal windows stay on the learned manifold
                      (small gap); anomalies drift (large gap).
    flow_weight     : blend weight for flow-based residual NLL in final
                      score (0.25).  DBR-AF-inspired — a small RealNVP
                      models the joint distribution of all branch errors;
                      anomalies produce low-likelihood residual vectors.
                      Higher = more trust in flow vs. linear fusion.
    flow_epochs     : training epochs for the post-hoc RealNVP (200).
    batch_size
    epochs
    validation_size : fraction of training data used for validation
    """

    def __init__(self, window_size: int = 100, pred_len: int = 1, feats: int = 1,
                 hidden_dim: int = 64, lstm_hidden: int = 32, num_layers: int = 2,
                 lr: float = 1e-3, alpha: float = 0.5, freq_weight: float = 0.05,
                 temp_neighbor_weight: float = 0.0, temp_neighbor_delta: int = 2,
                 temp_exclude_radius: int | None = None,
                 adv_weight: float = 0.0,
                 mask_ratio: float = 0.3, mask_weight: float = 0.7,
                 freq_mask_weight: float = 0.03, freq_mask_bands: int = 8,
                 rbf_n_centers: int = 32, rbf_weight: float = 0.01,
                 idem_weight: float = 0.1,
                 flow_weight: float = 0.25, flow_epochs: int = 200,
                 batch_size: int = 128, epochs: int = 60,
                 validation_size: float = 0.2,
                 max_train_windows: int = 50_000):
        self.window_size = window_size
        self.pred_len = pred_len
        self.feats = feats
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.lr = lr
        self.alpha = alpha
        self.freq_weight = freq_weight
        self.temp_neighbor_weight = temp_neighbor_weight
        self.temp_neighbor_delta = temp_neighbor_delta
        self.temp_exclude_radius = (temp_exclude_radius
                                    if temp_exclude_radius is not None
                                    else temp_neighbor_delta * 2)
        self.adv_weight = adv_weight
        self.mask_ratio = mask_ratio
        self.mask_weight = mask_weight
        self.freq_mask_weight = freq_mask_weight
        self.freq_mask_bands = freq_mask_bands
        self.rbf_n_centers = rbf_n_centers
        self.rbf_weight = rbf_weight
        self.idem_weight = idem_weight
        self.flow_weight = flow_weight
        self.flow_epochs = flow_epochs
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.max_train_windows = max_train_windows

        self.device = _silent_get_gpu(cuda=True)

        self.model = DualStreamNet(feats, window_size, hidden_dim, lstm_hidden,
                                    num_layers, pred_len,
                                    self.rbf_n_centers).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.early_stopping = EarlyStoppingTorch(save_path=None, patience=7)

    @staticmethod
    def _freq_loss(recon: torch.Tensor, target: torch.Tensor,
                   n_bands: int = 4) -> torch.Tensor:
        """Banded frequency loss: split FFT into log-spaced bands, MSE per band.

        Instead of global FFT MSE (which washes out per-band and per-channel
        signals), this splits the frequency axis into n_bands log-spaced
        regions and averages the per-band reconstruction error.  Anomalies
        in specific frequency ranges (pump oscillations, control instability)
        are penalised more sharply.
        """
        f_recon  = torch.fft.rfft(recon,  dim=1).abs()   # (B, W//2+1, F)
        f_target = torch.fft.rfft(target, dim=1).abs()
        n_freq = f_recon.shape[1]                         # W//2+1
        if n_freq <= n_bands:
            return F.mse_loss(f_recon, f_target)

        # Log-spaced band boundaries
        edges = torch.unique(torch.logspace(
            0, np.log10(n_freq), n_bands + 1, dtype=torch.int32))
        losses = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            lo, hi = int(lo), int(hi)
            if hi <= lo:
                continue
            losses.append(F.mse_loss(f_recon[:, lo:hi], f_target[:, lo:hi]))
        return torch.stack(losses).mean() if losses else F.mse_loss(f_recon, f_target)

    @staticmethod
    def _freq_masked_recon_loss(pred_freq: torch.Tensor, x: torch.Tensor,
                                n_bands: int = 8, mask_frac: float = 0.35
                                ) -> torch.Tensor:
        """Masked frequency reconstruction loss (TSPulse-style).

        Randomly masks frequency bands in the ground-truth spectrum and
        computes MSE only on those bands — forces the latent to encode
        spectral structure that survives band dropout.

        Parameters
        ----------
        pred_freq : (B, n_freq) predicted magnitude from BiLSTM latent.
        x         : (B, W, F)  original input window (before RevIN).
        n_bands   : number of log-spaced frequency bands.
        mask_frac : fraction of bands to mask (0.35 = ~1/3).

        Returns scalar loss.
        """
        B = x.size(0)
        actual_freq = torch.fft.rfft(x, dim=1).abs().mean(dim=2)  # (B, n_freq)
        n_freq = actual_freq.shape[1]

        if n_freq <= n_bands:
            return F.mse_loss(pred_freq, actual_freq)

        # Log-spaced band edges
        edges = torch.unique(torch.logspace(
            0, np.log10(n_freq), n_bands + 1, dtype=torch.int32,
            device=x.device))
        n_actual = len(edges) - 1
        # Randomly select bands to mask
        n_mask = max(1, int(n_actual * mask_frac))
        perm = torch.randperm(n_actual, device=x.device)[:n_mask]

        loss = 0.0
        n_contrib = 0
        for b_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            lo, hi = int(lo), int(hi)
            if hi <= lo:
                continue
            if b_idx in perm:
                # Masked band — compute loss
                loss = loss + F.mse_loss(
                    pred_freq[:, lo:hi], actual_freq[:, lo:hi])
                n_contrib += 1
        return loss / max(n_contrib, 1)

    @staticmethod
    def _banded_freq_error(recon: torch.Tensor, x: torch.Tensor,
                           n_bands: int = 4) -> torch.Tensor:
        """Per-window banded frequency anomaly score (for inference).

        For each band, compute max per-channel MSE, then take max across
        bands.  Returns (B,) tensor — high when any frequency band deviates.
        """
        f_recon = torch.fft.rfft(recon, dim=1).abs()
        f_x     = torch.fft.rfft(x,    dim=1).abs()
        n_freq  = f_recon.shape[1]
        B = f_recon.shape[0]
        if n_freq <= n_bands:
            err = (f_recon - f_x).pow(2).mean(dim=1)         # (B, F)
            return err.max(dim=1).values                      # (B,)

        edges = torch.unique(torch.logspace(
            0, np.log10(n_freq), n_bands + 1, dtype=torch.int32))
        band_max = torch.zeros(B, device=recon.device)
        for lo, hi in zip(edges[:-1], edges[1:]):
            lo, hi = int(lo), int(hi)
            if hi <= lo:
                continue
            err_band = (f_recon[:, lo:hi] - f_x[:, lo:hi]).pow(2)  # (B, bw, F)
            # Max over frequency bins, then max over channels
            band_score = err_band.mean(dim=1).max(dim=1).values      # (B,)
            band_max = torch.maximum(band_max, band_score)
        return band_max

    @staticmethod
    def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                      temperature: float = 0.07,
                      anchor_idx: torch.Tensor | None = None,
                      exclude_radius: int = 0) -> torch.Tensor:
        """NT-Xent contrastive loss with optional temporal exclusion.

        Positive pairs: (z1[i], z2[i]) – two views of the same concept.
        Negative pairs: all other samples in the batch, EXCEPT those
        within ±exclude_radius of the anchor's temporal position.

        Without temporal exclusion, nearby windows (which share latent
        structure in normal regimes) become false negatives → noisy
        gradient.  With exclusion they are simply ignored.
        """
        B  = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)                      # (2B, latent_dim)
        sim = torch.mm(z, z.T) / temperature                  # (2B, 2B)
        # Self-match mask: don't let a sample be its own neighbour
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # ── Temporal exclusion mask ──────────────────────────────────────────
        if anchor_idx is not None and exclude_radius > 0:
            idx = anchor_idx.to(z.device)                        # (B,)
            # Distance matrix in temporal-index space: |idx_i - idx_j|
            idx_all = torch.cat([idx, idx])                       # (2B,)
            temporal_dist = (idx_all[:, None] - idx_all[None, :]).abs()  # (2B, 2B)
            # Exclude pairs closer than exclude_radius (but NOT self-matches
            # — those are already masked above)
            too_close = (temporal_dist > 0) & (temporal_dist < exclude_radius)
            sim.masked_fill_(too_close, -1e9)

        labels = torch.cat([torch.arange(B, 2 * B, device=z.device),
                             torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)

    @staticmethod
    def _freq_view(x: torch.Tensor, drop_p: float = 0.3) -> torch.Tensor:
        """Create frequency-augmented view via random spectral dropout (TFCID).

        FFT → random band dropout (keep DC + low freqs) → IFFT.
        Forces latent to be invariant to frequency perturbations.
        """
        X = torch.fft.rfft(x, dim=1)
        mask = (torch.rand_like(X.real) > drop_p).float()
        lo_cut = max(1, X.shape[1] // 10)
        mask[:, :lo_cut] = 1.0
        return torch.fft.irfft(X * mask, n=x.size(1), dim=1)

    @staticmethod
    def _freq_contrast_loss(z_time: torch.Tensor, z_freq: torch.Tensor,
                            temperature: float = 0.07) -> torch.Tensor:
        """NT-Xent between time and frequency latent views.

        z_time[i] ↔ z_freq[i] are positives; all other pairs negatives.
        """
        B = z_time.size(0)
        z_time = F.normalize(z_time, dim=1)
        z_freq = F.normalize(z_freq, dim=1)
        z = torch.cat([z_time, z_freq], dim=0)
        sim = torch.mm(z, z.T) / temperature
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(B, 2 * B, device=z.device),
                             torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)

    @staticmethod
    def _generate_mask(B: int, W: int, mask_ratio: float,
                       device: torch.device) -> torch.Tensor:
        """Random timestep mask for masked reconstruction.

        Randomly selects floor(mask_ratio * W) timesteps per sample and masks
        ALL features at those positions.  This forces the encoder+decoder to
        infer missing content from surrounding context, breaking the BiLSTM
        identity shortcut.

        Returns (B, W, 1) binary mask: 1.0 = masked, 0.0 = visible.
        """
        n_mask = max(1, int(W * mask_ratio))
        # Random permutation per sample → take first n_mask as masked indices
        rand = torch.rand(B, W, device=device)
        _, idx = rand.topk(n_mask, dim=1, largest=True)   # (B, n_mask)
        mask = torch.zeros(B, W, 1, device=device)
        mask.scatter_(1, idx.unsqueeze(-1), 1.0)
        return mask

    class _TemporalPairDataset(torch.utils.data.Dataset):
        """Returns (anchor_x, target, neighbor_x, anchor_idx) for temporal
        contrastive.  anchor_idx tracks the global window position, enabling
        exclusion-aware contrastive loss.

        Positive pairs are windows δ steps apart in time — nearby windows
        in normal regimes share similar latent structure, making temporal
        adjacency a natural self-supervision signal for anomaly detection.
        """
        def __init__(self, data: np.ndarray, window_size: int,
                     pred_len: int, delta: int):
            self.delta = delta
            self.ds = ForecastDataset(data, window_size, pred_len)
            self.n_pairs = max(0, len(self.ds) - delta)

        def __len__(self):
            return self.n_pairs

        def __getitem__(self, idx):
            x_a, t_a = self.ds[idx]
            x_n, _   = self.ds[idx + self.delta]
            return x_a, t_a, x_n, idx  # idx = global temporal position

    def _make_loader(self, data: np.ndarray, shuffle: bool,
                     stride: int = 1) -> DataLoader:
        ds = ForecastDataset(data, window_size=self.window_size,
                             pred_len=self.pred_len)
        if stride > 1 and len(ds) > stride:
            ds = Subset(ds, list(range(0, len(ds), stride)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                         pin_memory=torch.cuda.is_available())

    @staticmethod
    def _robust_norm(s: np.ndarray) -> np.ndarray:
        """Percentile-based scaling – robust to extreme outliers."""
        lo, hi = np.percentile(s, [1, 99])
        if hi - lo < 1e-8:
            return np.zeros_like(s)
        return np.clip((s - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _fill_gaps(s: np.ndarray, cnt: np.ndarray) -> np.ndarray:
        """Linear interpolation for positions not covered by any window."""
        valid = cnt > 0
        if not valid.any():
            return s
        s[~valid] = np.interp(
            np.where(~valid)[0], np.where(valid)[0], s[valid])
        return s

    @staticmethod
    def _smooth(s: np.ndarray, window: int) -> np.ndarray:
        """Moving-average smoothing to reduce point-wise noise."""
        if window <= 1:
            return s
        kernel = np.ones(window) / window
        return np.convolve(s, kernel, mode='same')

    def _accumulate_scores(self, data: np.ndarray):
        """Shared scoring loop – returns (recon, pred, latent, adv, freq, gate)."""
        ds = ForecastDataset(data, window_size=self.window_size,
                             pred_len=self.pred_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        T = len(data)
        recon_acc  = np.zeros(T, dtype=np.float64)
        recon_cnt  = np.zeros(T, dtype=np.float64)
        pred_acc   = np.zeros(T, dtype=np.float64)
        pred_cnt   = np.zeros(T, dtype=np.float64)
        latent_acc = np.zeros(T, dtype=np.float64)
        latent_cnt = np.zeros(T, dtype=np.float64)
        adv_acc    = np.zeros(T, dtype=np.float64)
        adv_cnt    = np.zeros(T, dtype=np.float64)
        freq_acc   = np.zeros(T, dtype=np.float64)
        freq_cnt   = np.zeros(T, dtype=np.float64)
        freq_recon_acc = np.zeros(T, dtype=np.float64)
        freq_recon_cnt = np.zeros(T, dtype=np.float64)
        idem_acc   = np.zeros(T, dtype=np.float64)
        idem_cnt   = np.zeros(T, dtype=np.float64)
        flow_acc   = np.zeros(T, dtype=np.float64)
        flow_cnt   = np.zeros(T, dtype=np.float64)
        gate_acc   = np.zeros((T, 4), dtype=np.float64)
        gate_cnt   = np.zeros(T,      dtype=np.float64)

        # Convert per-channel stats to device tensors once
        ch_mu    = torch.tensor(self.recon_channel_mu,    dtype=torch.float32,
                                device=self.device)          # (F,)
        ch_sigma = torch.tensor(self.recon_channel_sigma, dtype=torch.float32,
                                device=self.device)          # (F,)

        self.model.eval()
        win_idx = 0
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                recon1, recon2, recon21, pred, z = self.model(
                    x, need_adv=True, need_pred=True, return_latent=True)
                # Per-channel z-score → max across channels (better for multivariate)
                re_raw = (recon1 - x).pow(2)                                  # (B, W, F)
                re_norm = ((re_raw - ch_mu[None, None, :])
                           / ch_sigma[None, None, :])                         # (B, W, F)
                re = re_norm.max(dim=2).values.cpu().numpy()                  # (B, W)
                # Weighted multi-horizon prediction error – γ^h discount
                _pw = self._pred_weights[None, :, None]           # (1, P, 1)
                pe   = ((pred - target).pow(2) * _pw).mean(dim=(1, 2)).cpu().numpy()
                pe_t = ((pred - target).pow(2) * _pw).mean(dim=(1, 2))        # (B,) tensor
                # RBF density scoring (RESTAD-inspired): 1 − max-RBF response.
                # High = far from all learned normal centres → anomalous.
                # Use pooled latent 'z' from forward() — no re-encoding.
                density = self.model.rbf_head(z)
                rbf_score_t = 1.0 - density  # (B,) tensor
                rbf_score = rbf_score_t.cpu().numpy()  # (B,)
                # Mahalanobis replaced by RBF — handles multi-modal normal.
                # Adversarial score: 0.1·AE1 + 0.9·AE2AE1
                adv_raw = (0.1 * (recon1 - x).pow(2).mean(dim=(1, 2))
                           + 0.9 * (recon21 - x).pow(2).mean(dim=(1, 2)))
                adv = adv_raw.cpu().numpy()
                adv_t = adv_raw                                        # (B,) tensor
                freq_err = self._banded_freq_error(recon1, x).cpu().numpy()
                freq_err_t = self._banded_freq_error(recon1, x)        # (B,) tensor
                pred_freq = self.model.freq_recon_head(z)
                actual_freq = torch.fft.rfft(
                    self.model.revin.normalize(x), dim=1).abs().mean(dim=2)
                freq_recon_err = (pred_freq - actual_freq).pow(2).mean(dim=1).cpu().numpy()  # (B,)
                # Idempotent gap: recon → re-reconstruct — large gap = anomaly
                recon_idem, _, _, _ = self.model(
                    recon1, need_adv=False, need_pred=False)
                idem = ((recon_idem - recon1) ** 2).mean(dim=(1, 2)).cpu().numpy()  # (B,)
                idem_t = ((recon_idem - recon1) ** 2).mean(dim=(1, 2))  # (B,) tensor
                # Residual vector for flow NLL:
                # [per-channel recon MSE (F) | pred | freq_band | RBF | adv | idem]
                recon_per_ch = (recon1 - x).pow(2).mean(dim=1)               # (B, F)
                rv = torch.cat([recon_per_ch,
                                pe_t.unsqueeze(1),
                                freq_err_t.unsqueeze(1),
                                rbf_score_t.unsqueeze(1),
                                adv_t.unsqueeze(1),
                                idem_t.unsqueeze(1)], dim=1)   # (B, F+5)
                flow_nll = self.residual_flow.nll(rv).cpu().numpy()  # (B,)
                w_gate = F.softmax(self.model.fusion_gate(z), dim=1).cpu().numpy()   # (B, 4)
                for i in range(re.shape[0]):
                    start = win_idx
                    end   = min(start + self.window_size, T)
                    L     = end - start
                    recon_acc[start:end]  += re[i, :L]
                    recon_cnt[start:end]  += 1.0
                    latent_acc[start:end] += rbf_score[i]
                    latent_cnt[start:end] += 1.0
                    adv_acc[start:end]    += adv[i]
                    adv_cnt[start:end]    += 1.0
                    freq_acc[start:end]   += freq_err[i]
                    freq_cnt[start:end]   += 1.0
                    freq_recon_acc[start:end] += freq_recon_err[i]
                    freq_recon_cnt[start:end] += 1.0
                    idem_acc[start:end]  += idem[i]
                    idem_cnt[start:end]  += 1.0
                    flow_acc[start:end]  += flow_nll[i]
                    flow_cnt[start:end]  += 1.0
                    gate_acc[start:end]   += w_gate[i]   # (4,) broadcast
                    gate_cnt[start:end]   += 1.0
                    pred_pos = start + self.window_size
                    if pred_pos < T:
                        pred_acc[pred_pos] += pe[i]
                        pred_cnt[pred_pos] += 1.0
                    win_idx += 1
        recon_score  = self._fill_gaps(recon_acc  / np.maximum(recon_cnt,  1.), recon_cnt)
        pred_score   = self._fill_gaps(pred_acc   / np.maximum(pred_cnt,   1.), pred_cnt)
        latent_score = self._fill_gaps(latent_acc / np.maximum(latent_cnt, 1.), latent_cnt)
        adv_score    = self._fill_gaps(adv_acc    / np.maximum(adv_cnt,    1.), adv_cnt)
        freq_score   = self._fill_gaps(freq_acc   / np.maximum(freq_cnt,   1.), freq_cnt)
        freq_recon_score = self._fill_gaps(freq_recon_acc / np.maximum(freq_recon_cnt, 1.),
                                           freq_recon_cnt)
        # Blend banded-freq error + masked-freq-recon error (50/50)
        freq_score = 0.5 * freq_score + 0.5 * freq_recon_score
        idem_score  = self._fill_gaps(idem_acc  / np.maximum(idem_cnt,  1.), idem_cnt)
        flow_score  = self._fill_gaps(flow_acc  / np.maximum(flow_cnt,  1.), flow_cnt)
        # Normalize averaged gate weights to sum to 1 per timestep
        gate_weights = gate_acc / np.maximum(gate_cnt[:, None], 1.)     # (T, 4)
        gate_weights /= gate_weights.sum(axis=1, keepdims=True) + 1e-8
        return recon_score, pred_score, latent_score, adv_score, freq_score, gate_weights, idem_score, flow_score

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray):
        n_val = int(self.validation_size * len(data))
        n_trn = len(data) - n_val
        data_train, data_val = data[:n_trn], data[n_trn:]

        # Global z-score normalization – stabilizes cross-dataset scale
        # (mirrors MMPAD's zscore(seq_work, axis=0, ddof=0)).
        # RevIN still handles per-window distribution shift; this handles
        # shift between datasets so the model sees similar input ranges.
        self.data_mean_ = data_train.mean(axis=0, keepdims=True)   # (1, F)
        self.data_std_  = data_train.std(axis=0, ddof=0) + 1e-8   # (F,)
        data_train = (data_train - self.data_mean_) / self.data_std_
        data_val   = (data_val   - self.data_mean_) / self.data_std_

        # Adaptive stride: cap training windows for speed on large data
        # Set max_train_windows=0 to use ALL windows (no cap)
        _max_w = self.max_train_windows
        _n_windows = max(1, len(data_train) - self.window_size - self.pred_len)
        _stride = 1 if _max_w <= 0 else max(1, _n_windows // _max_w)
        if _stride > 1:
            print(f'  [DualStreamAD] large dataset ({len(data_train):,} rows): '
                  f'using train_stride={_stride} '
                  f'({_n_windows // _stride:,}/{_n_windows:,} windows)')

        train_loader = self._make_loader(data_train, shuffle=True,  stride=_stride)
        valid_loader = self._make_loader(data_val,   shuffle=False, stride=_stride)

        # ── Temporal Neighbor Contrastive setup ──────────────────────────────
        # When enabled, replace train_loader with temporal-pair batches
        # (x_anchor, target, x_neighbor) where x_neighbor is δ steps ahead.
        _use_temporal = self.temp_neighbor_weight > 0
        if _use_temporal:
            temp_ds = self._TemporalPairDataset(
                data_train, self.window_size, self.pred_len,
                self.temp_neighbor_delta)
            if _stride > 1 and len(temp_ds) > _stride:
                temp_ds = Subset(temp_ds, list(range(0, len(temp_ds), _stride)))
            train_loader = DataLoader(temp_ds, batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=torch.cuda.is_available())
            print(f'  [DualStreamAD] temporal neighbor contrastive enabled '
                  f'(δ={self.temp_neighbor_delta}, '
                  f'exclude={self.temp_exclude_radius}, '
                  f'λ={self.temp_neighbor_weight:.4f}, '
                  f'{len(temp_ds):,} pairs)')

        # Multi-horizon discount weights: γ^(h-1) / Σγ^(h-1), γ=0.7
        gamma = 0.7
        _w = torch.tensor([gamma ** h for h in range(self.pred_len)],
                          dtype=torch.float32, device=self.device)
        self._pred_weights = _w / _w.sum()   # (pred_len,)

        for epoch in range(1, self.epochs + 1):

            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            avg_loss = 0.0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                             leave=False)
            for idx, batch in loop:
                # ── Unpack batch ─────────────────────────────────────────────
                if _use_temporal:
                    x, target, x_neighbor, anchor_idx = batch
                    x, target, x_neighbor, anchor_idx = (
                        x.to(self.device), target.to(self.device),
                        x_neighbor.to(self.device),
                        anchor_idx.clone())     # (B,) — temporal positions
                else:
                    x, target = batch
                    x, target = x.to(self.device), target.to(self.device)
                    anchor_idx = None
                B = x.size(0)

                # ── Denoising reconstruction view + masked timesteps ──────────
                x_noisy = x + torch.randn_like(x) * 0.005
                # Random timestep mask — forces BiLSTM to infer, not copy
                mask = self._generate_mask(B, self.window_size,
                                            self.mask_ratio, self.device)
                _use_adv = self.adv_weight > 0
                recon1, recon2, recon21, pred = self.model(
                    x_noisy, mask=mask,
                    need_adv=_use_adv, need_pred=True)
                z1 = self.model.get_latent(x_noisy)   # (B, latent_dim) – always available

                # ── Contrastive loss (temporal + frequency + diffusion) ──────
                if _use_temporal and B > 1:
                    z2 = self.model.get_latent(x_neighbor)
                    c_loss = self._nt_xent_loss(
                        z1, z2, anchor_idx=anchor_idx,
                        exclude_radius=self.temp_exclude_radius)
                elif B > 1:
                    x_aug = x + torch.randn_like(x) * 0.01
                    z2 = self.model.get_latent(x_aug)
                    c_loss = self._nt_xent_loss(z1, z2)
                else:
                    c_loss = 0.0

                # Frequency-view contrastive (TFCID-inspired)
                if B > 1:
                    x_freq = self._freq_view(x, drop_p=0.3)
                    z_freq = self.model.get_latent(x_freq)
                    c_loss = c_loss + 0.3 * self._freq_contrast_loss(z1, z_freq)

                # ── Adversarial dual-decoder loss ─────────────────────────────
                if _use_adv:
                    n_adv = epoch + 1.0            # progressive schedule
                    alpha_adv = 1.0 / n_adv        # 1→0.5→0.33→…
                    # Decoder 1: direct recon + cross-cycle consistency.
                    # Decoder 2: independent reconstruction (different init
                    # → different learned function → discrepancy with dec1).
                    # Cross-cycle in l1 does NOT cancel — dec2 has its own loss.
                    l1 = (alpha_adv       * F.mse_loss(recon1, x)
                          + (1. - alpha_adv) * F.mse_loss(recon21, x))
                    l2 = F.mse_loss(recon2, x)
                    adv_loss = l1 + l2
                else:
                    adv_loss = 0.0

                # Gate-weighted loss guidance – gate learns which branch minimises
                # error for each window context without affecting encoder/decoder.
                w_gate = F.softmax(self.model.fusion_gate(z1), dim=1)   # (B, 4)
                with torch.no_grad():
                    err_r = F.mse_loss(recon1, x, reduction='none').detach()
                    # Weighted multi-horizon pred error: discount far steps
                    _pw   = self._pred_weights[None, :, None]        # (1, P, 1)
                    err_p = ((pred - target).pow(2) * _pw).detach()
                    err_r = (err_r.mean(dim=(1, 2)) - err_r.mean()) / (err_r.std() + 1e-8)
                    err_p = (err_p.mean(dim=(1, 2)) - err_p.mean()) / (err_p.std() + 1e-8)
                    # Latent error: Euclidean distance from batch latent centre.
                    # Windows far from the batch mean are anomalous in latent
                    # space — gate learns to down-weight latent when it diverges.
                    err_l = ((z1.detach() - z1.mean(0)) ** 2).sum(1)
                    err_l = (err_l - err_l.mean()) / (err_l.std() + 1e-8)
                    if _use_adv:
                        # Use ONLY cross-cycle error (recon21) to keep adv branch
                        # independent from recon branch — avoids correlation bias.
                        err_a_raw = (recon21 - x).pow(2).mean(dim=(1, 2)).detach()
                        err_a = (err_a_raw - err_a_raw.mean()) / (err_a_raw.std() + 1e-8)
                    else:
                        err_a = torch.zeros_like(err_r)
                    err_all     = torch.stack([err_r, err_p, err_l, err_a], dim=1)   # (B, 4)
                    # Softer temperature (0.5 vs 0.1): prevents one-hot collapse,
                    # lets gate learn truly contextual mixing rather than winner-take-all.
                    gate_target = F.softmax(-err_all / 0.5, dim=1)
                gate_loss = F.kl_div(w_gate.log(), gate_target, reduction='batchmean')

                # Multi-horizon prediction loss with discount γ^h
                pred_loss = ((pred - target).pow(2) * _pw).mean()

                # Masked reconstruction loss — emphasis on masked timesteps.
                # Without masking, BiLSTM can trivially copy the input (identity
                # shortcut), collapsing the reconstruction gap on anomalies.
                recon_se = (recon1 - x).pow(2)                    # (B, W, F)
                masked_mse = (recon_se * mask).sum() / (mask.sum() + 1e-8)
                full_mse   = recon_se.mean()
                recon_loss = (self.mask_weight * masked_mse
                              + (1. - self.mask_weight) * full_mse)

                # ── Frequency-domain masked reconstruction (TSPulse-style) ───
                # Predict FFT magnitude from latent; loss only on masked bands.
                # Complementary to time masking — catches spectral anomalies.
                pred_freq = self.model.get_freq_recon(z1)         # (B, n_freq)
                fmr_loss = self._freq_masked_recon_loss(
                    pred_freq, x, self.freq_mask_bands, mask_frac=0.35)

                # ── RBF commitment loss ──────────────────────────────────────
                # Pull RBF centres towards training latents: penalise low
                # max-RBF response so every normal window is close to at
                # least one centre.  End-to-end trainable.
                rbf_density = self.model.rbf_head(z1)          # (B,)
                rbf_commit_loss = (1.0 - rbf_density).mean()

                # ── Idempotent reconstruction (IGAD-inspired) ────────────────
                # Pass recon1 back through the SAME encoder+decoder.  Normal
                # windows should stay on the learned manifold (small gap);
                # anomalies drift because the model never learned their structure.
                recon_idem, _, _, _ = self.model(
                    recon1.detach(), need_adv=False, need_pred=False)
                idem_loss = F.mse_loss(recon_idem, recon1.detach())

                loss = (recon_loss
                        + self.alpha          * pred_loss
                        + self.freq_weight    * self._freq_loss(recon1, x)
                        + self.freq_mask_weight * fmr_loss
                        + self.rbf_weight     * rbf_commit_loss
                        + self.idem_weight    * idem_loss
                        + (self.temp_neighbor_weight if _use_temporal else 0.05) * c_loss
                        + 0.05                * gate_loss
                        + self.adv_weight     * adv_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                avg_loss += loss.item()
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=f'{loss.item():.4f}',
                                 avg=f'{avg_loss / (idx + 1):.4f}')

            # ── Validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, target in valid_loader:
                    x, target = x.to(self.device), target.to(self.device)
                    recon1, _, _, pred = self.model(
                        x, need_adv=False, need_pred=True)
                    val_loss += (F.mse_loss(recon1, x)
                                 + self.alpha * ((pred - target).pow(2)
                                                  * self._pred_weights[None, :, None]).mean()
                                 + self.freq_weight * self._freq_loss(recon1, x)).item()

            val_loss /= max(len(valid_loader), 1)
            self.scheduler.step()

            _old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                self.early_stopping(val_loss, self.model)
            finally:
                sys.stdout.close()
                sys.stdout = _old_stdout
            if self.early_stopping.early_stop:
                break

        # ── Initialise RBF density centres (k-means++ on training latents) ───
        self.model.eval()
        latents = []
        stat_loader = self._make_loader(data_train, shuffle=False)
        with torch.no_grad():
            for x, _ in stat_loader:
                latents.append(self.model.get_latent(x.to(self.device)).cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        self.model.rbf_head.init_centers(latents)

        # ── Residual flow (RealNVP) over multi-branch error vector ───────────
        # Collect residual vectors from training windows, then fit a small
        # normalizing flow to model the joint distribution.  Anomalies land
        # in low-density regions → high NLL.
        rv_dim = self.feats + 5  # per-ch recon (F) + pred + freq + RBF + adv + idem
        self.residual_flow = _ResidualFlow(rv_dim, n_layers=4, hidden_dim=32)
        rv_list = []
        with torch.no_grad():
            for x, _ in stat_loader:
                x = x.to(self.device)
                recon1, recon2, recon21, pred = self.model(
                    x, need_adv=True, need_pred=True)
                recon_per_ch = (recon1 - x).pow(2).mean(dim=1)        # (B, F)
                pe_t = ((pred - x[:, -1:, :]).pow(2)
                        * self._pred_weights[None, :, None]).mean(dim=(1, 2))
                freq_err_t = self._banded_freq_error(recon1, x)
                rbf_score_t = self.model.get_rbf_score(x)
                adv_t = (0.1 * (recon1 - x).pow(2).mean(dim=(1, 2))
                         + 0.9 * (recon21 - x).pow(2).mean(dim=(1, 2)))
                recon_idem, _, _, _ = self.model(
                    recon1, need_adv=False, need_pred=False)
                idem_t = ((recon_idem - recon1) ** 2).mean(dim=(1, 2))
                rv = torch.cat([recon_per_ch, pe_t.unsqueeze(1),
                                freq_err_t.unsqueeze(1),
                                rbf_score_t.unsqueeze(1),
                                adv_t.unsqueeze(1),
                                idem_t.unsqueeze(1)], dim=1)
                rv_list.append(rv.cpu().numpy())
        rv_all = np.concatenate(rv_list, axis=0)
        # Cap at 30K samples for flow training speed
        if len(rv_all) > 30000:
            idx = np.random.choice(len(rv_all), 30000, replace=False)
            rv_all = rv_all[idx]
        self.residual_flow.fit(rv_all, self.device,
                               epochs=self.flow_epochs, batch_size=2048)

        # ── Channel-wise reconstruction error statistics ─────────────────────
        tr_r_windows = []
        with torch.no_grad():
            for x, _ in stat_loader:
                x = x.to(self.device)
                recon1, _, _, _ = self.model(
                    x, need_adv=False, need_pred=False)
                err = (recon1 - x).pow(2)            # (B, W, F)
                tr_r_windows.append(err.cpu().numpy())
        tr_r_all  = np.concatenate(tr_r_windows, axis=0)   # (N, W, F)
        tr_r_flat = tr_r_all.reshape(-1, self.feats)        # (N*W, F)
        self.recon_channel_mu    = tr_r_flat.mean(axis=0)   # (F,)
        self.recon_channel_sigma = tr_r_flat.std(axis=0) + 1e-8  # (F,)

        # Store training-set score distribution for z-normalization at inference.
        # This anchors test scores to normal behavior, preventing score inversion.
        tr_r, tr_p, tr_l, tr_a, tr_f, _, tr_idem, tr_flow = self._accumulate_scores(data_train)
        self._tr_r = (float(tr_r.mean()), max(float(tr_r.std()), 1e-8))
        self._tr_p = (float(tr_p.mean()), max(float(tr_p.std()), 1e-8))
        self._tr_l = (float(tr_l.mean()), max(float(tr_l.std()), 1e-8))
        self._tr_a = (float(tr_a.mean()), max(float(tr_a.std()), 1e-8))
        self._tr_f = (float(tr_f.mean()), max(float(tr_f.std()), 1e-8))
        self._tr_idem = (float(tr_idem.mean()), max(float(tr_idem.std()), 1e-8))
        self._tr_flow = (float(tr_flow.mean()), max(float(tr_flow.std()), 1e-8))

        # ── Inference ──────────────────────────────────────────────────────────────

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly scores with contextual branch fusion."""
        # Global z-score with training stats (mirrors MMPAD)
        data = (data - self.data_mean_) / self.data_std_
        recon_score, pred_score, latent_score, adv_score, \
            freq_score, gate_w, idem_score, flow_score = self._accumulate_scores(data)

        # Z-normalize relative to training normal distribution.
        # Anomalies deviate positively from training stats; clipped at -3/+10.
        def _z(s, mu, sigma):
            return np.clip((s - mu) / sigma, -3.0, 10.0)

        recon_score  = _z(recon_score,  *self._tr_r)
        pred_score   = _z(pred_score,   *self._tr_p)
        latent_score = _z(latent_score, *self._tr_l)
        adv_score    = _z(adv_score,    *self._tr_a)
        freq_score   = _z(freq_score,   *self._tr_f)
        idem_score   = _z(idem_score,   *self._tr_idem)
        flow_score   = _z(flow_score,   *self._tr_flow)

        # Light smoothing to reduce point-wise noise
        w = max(3, self.window_size // 20)
        recon_score  = self._smooth(recon_score,  w)
        pred_score   = self._smooth(pred_score,   w)
        latent_score = self._smooth(latent_score, w)
        adv_score    = self._smooth(adv_score,    w)
        freq_score   = self._smooth(freq_score,   w)
        idem_score   = self._smooth(idem_score,   w)
        flow_score   = self._smooth(flow_score,   w)

        # ── Contextual fusion + banded frequency bonus ───────────────────────
        p_r = self._robust_norm(recon_score)
        p_p = self._robust_norm(pred_score)
        p_l = self._robust_norm(latent_score)
        p_a = self._robust_norm(adv_score)
        p_f = self._robust_norm(freq_score)
        # Blend idempotent gap into recon component (85% recon + 15% idem).
        # Idempotence guards against over-generalization — normal stays
        # on the manifold after re-reconstruction; anomaly drifts.
        p_r = 0.85 * p_r + 0.15 * self._robust_norm(idem_score)
        # Main fusion: 4-branch gate
        final = (gate_w[:, 0] * p_r + gate_w[:, 1] * p_p
                 + gate_w[:, 2] * p_l + gate_w[:, 3] * p_a)
        # Blend in banded frequency score (10% weight — catches oscillation anomalies)
        final = 0.9 * final + 0.1 * p_f
        # Blend flow-based residual NLL (25% weight — DBR-AF-inspired).
        # The RealNVP learns the joint distribution of all branch errors;
        # anomalies fall outside the learned normal manifold → high NLL.
        final = (1.0 - self.flow_weight) * final + self.flow_weight * self._robust_norm(flow_score)

        avg_w = gate_w.mean(axis=0)
        print(f'  [DualStreamAD] gate weights (mean): '
              f'recon={avg_w[0]:.3f}, pred={avg_w[1]:.3f}, '
              f'latent={avg_w[2]:.3f}, adv={avg_w[3]:.3f}')
        return final.astype(np.float32)
