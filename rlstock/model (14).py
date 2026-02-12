import os
import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.nn import Parameter
from replay_buffer import PrioritizedReplayBufferNew
#from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from collections import deque
import torch.jit as jit

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear3(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = std_init

        self.weight_mu = nn.Parameter(T.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(T.full((out_features, in_features), self.sigma_0 / math.sqrt(in_features)))
        self.register_buffer('weight_epsilon', T.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(T.empty(out_features))
        self.bias_sigma = nn.Parameter(T.full((out_features,), self.sigma_0 / math.sqrt(in_features)))
        self.register_buffer('bias_epsilon', T.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @T.no_grad()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.uniform_(-bound, bound)
        self.bias_mu.uniform_(-bound, bound)

    @T.no_grad()
    def _get_noise(self, size):
        noise = T.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    @T.no_grad()
    def reset_noise(self) -> None:
        device = self.weight_mu.device
        # Use the scripted helper function
        eps_in = self._get_noise(self.in_features)
        eps_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(T.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)
        
    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

#########################################################################################
class WeakTieDropout(nn.Module):
    """
    Dropout replacement for linear layers that replaces dropped features
    with a cheap weak-tie signal (sparse linear mixing of other features).
    
    Args:
        p (float): probability of replacement (like dropout)
        k (int): number of acquaintances (nonzeros) per feature
        scale (bool): inverted scaling so expectation matches input
        seed (int): seed for reproducibility
    """
    def __init__(self, p=0.2, k=2, scale=True, seed=0):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self.k = k
        self.scale = scale
        self.seed = seed

        self.register_buffer("_m_idx", None, persistent=False)
        self.register_buffer("_m_w", None, persistent=False)
        self._feat = None

    def _build_mixer(self, feat, device, dtype):
        if self._feat == feat and self._m_idx is not None:
            return
        g = T.Generator(device=device)
        g.manual_seed(self.seed)
        src_idx = T.randint(0, feat, size=(feat, self.k), generator=g, device=device)
        w = T.randn(feat, self.k, device=device, dtype=dtype) / math.sqrt(self.k)
        self._m_idx, self._m_w, self._feat = src_idx, w, feat

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        B, F = x.shape
        self._build_mixer(F, x.device, x.dtype)

        # Weak-tie substitute
        x_gather = x[:, self._m_idx]                 # [B, F, k]
        wt = (x_gather * self._m_w).sum(dim=-1)      # [B, F]

        keep = (T.rand(B, F, device=x.device) > self.p)

        if self.scale:
            out = T.where(keep, x / (1.0 - self.p), wt / (self.p + 1e-12))
        else:
            out = T.where(keep, x, wt)
        return out
        
class WeakTieDropout2d(nn.Module):
    """
    Dropout for ConvNets that replaces dropped activations with a cheap weak-tie signal
    instead of zeros.

    Modes
    -----
    pattern='channel': per-sample channel-wise mask (like nn.Dropout2d)
        keep mask shape: [B, C, 1, 1]
    pattern='spatial': per-sample spatial mask shared over channels
        keep mask shape: [B, 1, H, W]

    Weak-tie signal
    ---------------
    Channel mixing: for each channel c, use k randomly-selected source channels
    and fixed weights ~ N(0, 1/k) to form     wt[:, c] = sum_j w[c, j] * x[:, src[c, j]]

    Args:
        p (float): probability of replacement (0..1)
        k (int): number of cross-channel “acquaintances” per output channel
        pattern (str): 'channel' or 'spatial'
        scale (bool): inverted scaling so E[out] ~= x during train
        seed (int): mixer seed (fixed once per module)
    """
    def __init__(self, p=0.2, k=2, pattern='channel', scale=True, seed=0):
        super().__init__()
        assert 0.0 <= p < 1.0
        assert pattern in ('channel', 'spatial')
        self.p = float(p)
        self.k = int(k)
        self.pattern = pattern
        self.scale = bool(scale)
        self.seed = int(seed)

        self.register_buffer('_m_idx', None, persistent=False)  # [C, k]
        self.register_buffer('_m_w',   None, persistent=False)  # [C, k]
        self._C = None

    def _build_mixer(self, C, device, dtype):
        if self._C == C and self._m_idx is not None:
            return
        g = T.Generator(device=device)
        g.manual_seed(self.seed)
        src = T.randint(0, C, (C, self.k), generator=g, device=device)
        w = T.randn(C, self.k, device=device, dtype=dtype) / math.sqrt(self.k)
        self._m_idx = src
        self._m_w = w
        self._C = C

    def forward(self, x: T.Tensor) -> T.Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        B, C, H, W = x.shape
        self._build_mixer(C, x.device, x.dtype)

        # Compute weak-tie substitute: wt shape [B, C, H, W]
        # x_gather -> [B, C, k, H, W], weights -> [1, C, k, 1, 1]
        x_gather = x[:, self._m_idx, :, :]                   # advanced indexing -> [B, C, k, H, W]
        wt = (x_gather * self._m_w.view(1, C, self.k, 1, 1)).sum(dim=2)

        # Keep mask
        if self.pattern == 'channel':
            keep = (T.rand(B, C, 1, 1, device=x.device) > self.p)
        else:  # 'spatial'
            keep = (T.rand(B, 1, H, W, device=x.device) > self.p)

        if self.scale:
            out = T.where(keep, x / (1.0 - self.p), wt / (self.p + 1e-12))
        else:
            out = T.where(keep, x, wt)
        return out

class WeakTieStochasticDepth2d(nn.Module):
    """
    Wrap a heavy Conv block f(x). During training, with prob p, skip f(x)
    and use a cheap weak-tie 1x1 conv g(x) (optionally sparse across channels).
    At eval, always run f(x).

    Args:
        block (nn.Module): conv block f(x)
        channels (int): number of channels for 1x1 tie
        p (float): skip probability
        sparse_k (int): if >0, sparsify rows of the 1x1 weight to k nonzeros
        seed (int): sparsity seed
    """
    def __init__(self, block: nn.Module, channels: int, p: float = 0.2, sparse_k: int = 0, seed: int = 0):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.block = block
        self.p = float(p)
        self.survival = 1.0 - self.p

        self.tie = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        with T.no_grad():
            nn.init.dirac_(self.tie.weight)  # start as near-identity

        if sparse_k > 0:
            W = self.tie.weight.data.squeeze(-1).squeeze(-1)  # [C, C]
            g = T.Generator(device=W.device).manual_seed(seed)
            mask = T.zeros_like(W)
            rows = T.arange(W.shape[0], device=W.device)
            cols = T.randint(0, W.shape[1], (W.shape[0], sparse_k), generator=g, device=W.device)
            mask[rows.unsqueeze(1), cols] = 1.0
            W.mul_(mask)
            # Renormalize row norms
            row_norm = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
            W.copy_(W / row_norm)
            self.tie.weight.copy_(W[:, :, None, None])

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return self.block(x)
        if T.rand(()) < self.p:
            return self.tie(x)  # skip heavy block, use cheap weak-tie projection
        y = self.block(x)
        return y / self.survival

#########################################################################################
class CustomNorm1d(nn.Module):
    """
    Batch-stat norm for [B, C] with no running stats.
    Adds min-std clamp, optional max-gain/value clamp, and B==1 identity.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        min_std: float = 1e-2,          # <- new: avoid huge scales when var~0 (C=1 safe)
        max_gain: float | None = 10.0,  # <- new: cap 1/std
        clamp: float | None = None,     # <- new: clip normalized values (e.g., 6.0)
        identity_if_b1: bool = False     # <- new: skip norm when B==1 (common at act time)
#z809        identity_if_b1: bool = True     # <- new: skip norm when B==1 (common at act time)
    ):
        super().__init__()
        self.C = num_features
        self.eps = eps
        self.min_std = float(min_std)
        self.max_gain = None if max_gain is None else float(max_gain)
        self.clamp_val = None if clamp is None else float(clamp)
        self.identity_if_b1 = bool(identity_if_b1)

        if affine:
            self.weight = nn.Parameter(T.ones(1, num_features))
            self.bias   = nn.Parameter(T.zeros(1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    def forward(self, x: T.Tensor) -> T.Tensor:  # x: [B, C]
        # (optional) shape check:
        # assert x.dim()==2 and x.size(1)==self.C, f"Expected [B,{self.C}], got {tuple(x.shape)}"
        B = x.size(0)

        # If acting with a single env/sample, skip normalization to avoid jitter
        if B == 1 and self.identity_if_b1:
            y = x
        else:
            mu  = x.mean(dim=0, keepdim=True)                     # [1, C]
            var = (x - mu).pow(2).mean(dim=0, keepdim=True)       # unbiased=False
            std = (var + self.eps).sqrt().clamp(min=self.min_std) # <- clamp

            gain = 1.0 / std
            if self.max_gain is not None:
                gain = gain.clamp(max=self.max_gain)              # <- cap gain

            y = (x - mu) * gain
            if self.clamp_val is not None:
                y = y.clamp(-self.clamp_val, self.clamp_val)      # <- bound values

        if self.weight is not None:
            y = y * self.weight + self.bias
        return y

##############################################################################################
class CustomNorm2d(nn.Module):
    def __init__(self, num_features: int, num_groups: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(T.ones(num_features))
            self.bias   = nn.Parameter(T.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    def forward(self, x: T.Tensor) -> T.Tensor:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
################################################################################################
class GlobalGroupNorm(nn.Module):
    """
    Stateless per-sample, per-group normalization for Conv2d-style windows.
    - Input:  [B, C, T, 1]  (or [B, C, T]; it will auto-unsqueeze)
    - Output: same rank as input (returns [B,C,T,1] if 4D, else [B,C,T])
    - Equal-sized, CONTIGUOUS groups only (C % G == 0)
    - No running stats; identical behavior in train/eval
    """
    def __init__(self, num_channels: int, num_groups: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        self.C = num_channels
        self.G = num_groups
        self.eps = eps
        if affine:
            # per-channel scale/shift (broadcast over T and W)
            self.weight = nn.Parameter(T.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(T.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)

    def forward(self, x: T.Tensor) -> T.Tensor:
        # Accept [B, C, T] or [B, C, T, 1]
        B, C, Tx, W = x.shape
        Cg = C // self.G

        # reshape into contiguous groups: [B, G, Cg, Tx, 1]
        xg = x.view(B, self.G, Cg, Tx, W)

        # per-sample, per-group mean/var over (Cg, Tx, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var  = xg.var (dim=(2, 3, 4), keepdim=True, unbiased=False)

        # normalize & restore shape
        xg = (xg - mean) / T.sqrt(var + self.eps)
        xn = xg.view(B, C, Tx, W)

        # learned affine (per channel)
        if self.weight is not None:
            xn = xn * self.weight + self.bias
            
        return xn

################################################################################################
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = CustomNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = CustomNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
#########################################################################################
class SE2DIdentity(nn.Module):
    """SE for [B,C,H,W] with identity-centered scaling."""
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        h = max(4, channels // r)
        self.fc1 = nn.Linear(channels, h)
        self.fc2 = nn.Linear(h, channels)
        # zero-init last layer → start at identity (scale ≈ 1)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # squeeze
        s = x.mean(dim=(2, 3))                  # [B,C]
        # gamma in (-1,1), scale = 1 + gamma in (0,2)
        gamma = T.tanh(self.fc2(F.relu(self.fc1(s))))
        scale = 1.0 + gamma                     # [B,C]
        return x * scale.unsqueeze(-1).unsqueeze(-1)

class SE2D(nn.Module):
    """
    Squeeze-and-Excitation for [B, C, H, W].
    Squeezes over (H, W) → per-channel weights in (0,1) → reweights channels.
    """
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(4, channels // r)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        # x: [B, C, H, W]
        s = x.mean(dim=(2, 3))            # [B, C]
        w = self.fc(s).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w

class ZGateCentered(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, h_dim)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, h, z):
        g = T.tanh(self.fc(z))   # (-1,1)
        return h * (1 + g)           # (0,2)

class ZGate(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, h_dim), nn.Sigmoid())
    def forward(self, h, z):              # h: [B,H], z: [B,1]
        g = self.fc(z)
        return h * g + h                  # residual gate

class TemporalGate(nn.Module):
    # depthwise 1D conv over time -> pointwise -> sigmoid gate
    def __init__(self, c, k=5):
        super().__init__()
        self.dw = nn.Conv1d(c, c, kernel_size=k, padding=k//2, groups=c)
        self.pw = nn.Conv1d(c, c, kernel_size=1)
        self.act = nn.Sigmoid()
    def forward(self, x):              # x: [B, C, T]
        g = self.act(self.pw(self.dw(x)))
        return x * (1 + g)             # residual gate (≥ identity)

#########################################################################################
# --- NEW: FiLM-style z gate (bounded & centered) ---
class ZGateFiLM(nn.Module):
    """
    FiLM(z): h' = LN(h) * (1 + scale * tanh(Gamma(zc))) + scale * tanh(Beta(zc))
      - zc = z - 1.0 centers z (given your z ~ [0,2])
      - scale (default 0.25) bounds influence so z can't dominate
      - LN stabilizes h magnitude before modulation
    """
    def __init__(self, z_dim: int, h_dim: int, scale: float = 0.25):
        super().__init__()
        self.gamma = nn.Linear(z_dim, h_dim)
        self.beta  = nn.Linear(z_dim, h_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)
        self.h_ln  = nn.LayerNorm(h_dim)
        self.scale = float(scale)

    def forward(self, h: T.Tensor, z: T.Tensor) -> T.Tensor:
        # z comes in as [B, 1]; center to make 0 = neutral inventory
        zc = z - 1.0
        h  = self.h_ln(h)
        g  = T.tanh(self.gamma(zc)) * self.scale   # multiplicative (bounded)
        b  = T.tanh(self.beta(zc))  * self.scale   # additive (bounded)
        return h * (1.0 + g) + b
#####################################################################################################################################################
class DilatedResidualConv2d(nn.Module):
    """
    Simple residual dilated Conv2d block.
    Keeps shape: (B, C, T, 1) -> (B, C, T, 1)
    Uses symmetric padding (like your existing conv_mainA).
    """
    def __init__(self, channels, dilation=1, kernel_size=3, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(dilation, 0),       # keep length T
            dilation=(dilation, 1)
        )
        
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(dilation, 0),
            dilation=(dilation, 1)
        )
        
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act = activation()

    def forward(self, x):
        # x: [B, C, T, 1]
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
#1004        out = self.norm1(out)
        out = self.act(out)
        return x + out     # residual
        

class DilatedTemporalEncoder(nn.Module):
    """
    Stack of dilated residual blocks to give higher-order temporal state.
    Channels must match conv_mainA output (128 in your case).
    """
    def __init__(self, channels):
        super().__init__()
        self.block1 = DilatedResidualConv2d(channels, dilation=1)
        self.block2 = DilatedResidualConv2d(channels, dilation=2)
        self.block3 = DilatedResidualConv2d(channels, dilation=4)
        self.block4 = DilatedResidualConv2d(channels, dilation=8)

    def forward(self, x):
        # x: [B, C, T, 1]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
################################################################################################################################################
class StaticMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class BoundedFiLM(nn.Module):
    """
    h <- LN(h) * (1 + s*tanh(gamma(z))) + s*tanh(beta(z))
    Bounded influence prevents conditioning stream from dominating.
    """
    def __init__(self, z_dim: int, h_dim: int, scale: float = 0.15):
        super().__init__()
        self.ln = nn.LayerNorm(h_dim)
        self.gamma = nn.Linear(z_dim, h_dim)
        self.beta  = nn.Linear(z_dim, h_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)
        self.scale = float(scale)

    def forward(self, h: T.Tensor, z: T.Tensor) -> T.Tensor:
        h = self.ln(h)
        g = T.tanh(self.gamma(z)) * self.scale
        b = T.tanh(self.beta(z))  * self.scale
        return h * (1.0 + g) + b


class GateFusion(nn.Module):
    """
    Learnable gate that balances temporal vs static.
    Produces a fused vector of size out_dim.
    """
    def __init__(self, t_dim: int, s_dim: int, out_dim: int = 512):
        super().__init__()
        self.t_ln = nn.LayerNorm(t_dim)
        self.s_ln = nn.LayerNorm(s_dim)

        self.t_proj = nn.Sequential(nn.Linear(t_dim, out_dim), nn.LayerNorm(out_dim))
        self.s_proj = nn.Sequential(nn.Linear(s_dim, out_dim), nn.LayerNorm(out_dim))

        # vector gate in (0,1) with out_dim channels
        self.gate = nn.Sequential(
            nn.Linear(t_dim + s_dim, out_dim),
            nn.Sigmoid()
        )

        # start near 50/50 (gate = 0.5 everywhere)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, ht: T.Tensor, hs: T.Tensor) -> T.Tensor:
        ht = self.t_ln(ht)
        hs = self.s_ln(hs)

        g = self.gate(T.cat([ht, hs], dim=1))  # [B, out_dim]
        t = self.t_proj(ht)                   # [B, out_dim]
        s = self.s_proj(hs)                   # [B, out_dim]
        return g * t + (1.0 - g) * s          # balanced mix


class EfficientC51DuelingDeepQNetworkCV2J23(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(EfficientC51DuelingDeepQNetworkCV2J23, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.input_size = input_dims
        self.action_size = int(n_actions)

        std_init = 0.5

        # mask to zero conv z-channel over time: [1, 19, 1, 1] with 0 at ch0, 1 elsewhere
        # you currently use 19 conv-stream channels after dropping t=0 from x and slicing to 1:...
        self.register_buffer("_zmask", T.cat([T.zeros(1, 1, 1, 1), T.ones(1, 18, 1, 1)], dim=1))

        # ----- Normalization groups (your originals) -----
        self.groupz_pos = CustomNorm1d(1)
        self.group1 = GlobalGroupNorm(num_groups=1, num_channels=6)
        self.group2 = CustomNorm2d(1)
        self.group3 = GlobalGroupNorm(num_groups=1, num_channels=2)
        self.group4 = CustomNorm2d(1)

        # state channels group: 9 ch (10..18) in your current slicing st = xA[:, 10:19, :, :]
        self.group_state = nn.Conv2d(9, 9, kernel_size=1, groups=9, bias=True)
        nn.init.ones_(self.group_state.weight)
        nn.init.constant_(self.group_state.bias, -0.5)

        # ----- Conv trunk -----
        self.conv_mainA = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=8*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(16*2),
            nn.ReLU(),
            ResNetBlock(16*2),
            SE2DIdentity(32, r=6),

            nn.Conv2d(in_channels=16*2, out_channels=16*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=16*2),

            nn.Conv2d(in_channels=8*2*2, out_channels=16*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(32*2),
            nn.ReLU(),
            ResNetBlock(32*2),
            SE2DIdentity(64, r=16),

            nn.Conv2d(in_channels=32*2, out_channels=32*2, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),
                      dilation=(2, 1), groups=32*2),

            nn.Conv2d(in_channels=16*2*2, out_channels=32*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(64*2),
            nn.ReLU(),
            ResNetBlock(64*2),
            SE2DIdentity(128, r=32),

            nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0),
                      dilation=(4, 1), groups=64*2),
        )

        # higher-order temporal encoder on top of conv_mainA
        self.temporal_encoder = DilatedTemporalEncoder(channels=128)
        self.temp_gate = TemporalGate(c=128, k=5)

        # ---------------------------
        # NEW: explicit static stream
        # ---------------------------
        # static snapshot size = 9 (st channels)
        self.static_mlp  = StaticMLP(in_dim=9, out_dim=128, p=0.0)

        # bounded conditioning of temporal embedding by static
        self.static_film = BoundedFiLM(z_dim=128, h_dim=256, scale=0.15)

        # balanced fusion to 512 dims (so heads remain 512->...)
        self.fuse_gate   = GateFusion(t_dim=256, s_dim=128, out_dim=512)

        # ----- Heads (keep your NoisyLinear3 style) -----
        self.advantage_stream = nn.Sequential(
            NoisyLinear3(512, 256, std_init=std_init),
            nn.ReLU(),
            NoisyLinear3(256, self.action_size, std_init=std_init)
        )

        self.value_stream = nn.Sequential(
            NoisyLinear3(512, 256, std_init=std_init),
            nn.ReLU(),
            NoisyLinear3(256, 1, std_init=std_init)
        )

        # cache noisy modules
        self._reset_noise_modules = [m for m in self.modules() if m is not self and hasattr(m, 'reset_noise')]

        # safety: head/out matches action_size
        out_feats = self.advantage_stream[-1].out_features
        assert out_feats == self.action_size, f"adv head {out_feats} != action_size {self.action_size}"

    def forward(self, x: T.Tensor, log: bool = False, adv_only: bool = False, training: bool = True) -> T.Tensor:
        """
        Expected input layout (your current behavior):
          - x: [B, 20?, T+1, 1] (you treat ch0,t0 as scalar z)
          - you do: z = x[:,0,0,:] and xA = x[:,:,1:,:] for conv over time

        This forward:
          - removes z from the temporal path via _zmask
          - feeds TA channels through conv trunk
          - extracts ONE static snapshot from the 9 state channels (t=0 after dropping t=0 of x)
          - static conditions temporal (bounded)
          - gated fusion -> heads
        """
        # ----- extract z (kept only if you still want it later) -----
        z = x[:, 0, 0, :]  # [B, 1] (unused in this version)

        # drop t=0 for conv stream
        xA = x[:, :, 1:, :].contiguous()     # [B, ?, T, 1]
        xA = xA * self._zmask                # ensure conv never sees ch0 over time

        # ----- split TA vs state (same as your code) -----
        ta = xA[:, :10, :, :]                # 0..9
        st = xA[:, 10:19, :, :]              # 10..18 (9 channels)

        # norms
        group_1 = self.group1(ta[:, :6,  :, :])   # 0..5
        group_2 = self.group2(ta[:, 6:7, :, :])   # 6
        group_3 = self.group3(ta[:, 7:9, :, :])   # 7..8
        group_4 = self.group4(ta[:, 9:10,:, :])   # 9
        group_state = self.group_state(st)        # 9 state channels

        # concat for temporal conv trunk: 6+1+2+1+9 = 19 channels
        xA = T.cat([group_1, group_2, group_3, group_4, group_state], dim=1)

        # ----- conv trunk -----
        xA = self.conv_mainA(xA)                  # [B, 128, T, 1]
        xA = self.temporal_encoder(xA)            # [B, 128, T, 1]
        xA = xA.squeeze(-1)                       # [B, 128, T]
        xA = self.temp_gate(xA)

        # temporal embedding ht: [B,256] via GAP+GMP
        gap = xA.mean(dim=-1)                     # [B,128]
        gmp, _ = xA.max(dim=-1)                   # [B,128]
        ht = T.cat([gap, gmp], dim=1)             # [B,256]

        # ----- static snapshot: take t=0 (or use -1 for most-recent) -----
        # st is pre-group_state and may already be (near-)constant across time if you tile it in env.
        # Taking one slice prevents “T votes” into the conv trunk.
        static_vec = st[:, :, 0, 0]               # [B, 9]
        hs = self.static_mlp(static_vec)          # [B,128]

        # bounded modulation of temporal by static
        ht = self.static_film(ht, hs)             # [B,256]

        # balanced gated fusion
        comb = self.fuse_gate(ht, hs)             # [B,512]

        advantage = self.advantage_stream(comb)   # [B, action_size]
        if adv_only:
            return advantage

        value = self.value_stream(comb)           # [B, 1]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for module in self._reset_noise_modules:
            module.reset_noise()

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))

################################################################################################################################################
class EfficientC51DuelingDeepQNetworkCV2J23_011526(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(EfficientC51DuelingDeepQNetworkCV2J23, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.input_size = input_dims

        # keep consistent with env/agent
        self.action_size = int(n_actions)

        std_init = 0.5

        # mask to zero conv z-channel over time: [1, 16, 1, 1] with 0 at ch0, 1 elsewhere
#        self.register_buffer("_zmask", T.cat([T.zeros(1,1,1,1), T.ones(1,19,1,1)], dim=1))
        self.register_buffer("_zmask", T.cat([T.zeros(1,1,1,1), T.ones(1,18,1,1)], dim=1))
#1000        self.register_buffer("_zmask", T.cat([T.zeros(1,1,1,1), T.ones(1,15,1,1)], dim=1))

        # ----- Normalization groups (your originals) -----
        self.groupz_pos = CustomNorm1d(1)
        self.group1 = GlobalGroupNorm(num_groups=1, num_channels=6)
        self.group2 = CustomNorm2d(1)
        self.group3 = GlobalGroupNorm(num_groups=1, num_channels=2)
        self.group4 = CustomNorm2d(1)

        # state channels group (in_pos, pos_frac, hold, sell, buy, ramp) -> 6 ch
#        self.group_state = nn.Conv2d(10, 10, kernel_size=1, groups=10, bias=True)
        self.group_state = nn.Conv2d(9, 9, kernel_size=1, groups=9, bias=True)
#1000        self.group_state = nn.Conv2d(6, 6, kernel_size=1, groups=6, bias=True)
        nn.init.ones_(self.group_state.weight)
        nn.init.constant_(self.group_state.bias, -0.5)

        # ----- Conv trunk -----
        self.conv_mainA = nn.Sequential(
#            nn.Conv2d(in_channels=20, out_channels=8*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=19, out_channels=8*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
#1000            nn.Conv2d(in_channels=16, out_channels=8*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(16*2),
            nn.ReLU(),
            ResNetBlock(16*2),
            SE2DIdentity(32, r=6),

            nn.Conv2d(in_channels=16*2, out_channels=16*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=16*2),

            nn.Conv2d(in_channels=8*2*2, out_channels=16*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(32*2),
            nn.ReLU(),
            ResNetBlock(32*2),
            SE2DIdentity(64, r=16),

            nn.Conv2d(in_channels=32*2, out_channels=32*2, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), dilation=(2, 1), groups=32*2),

            nn.Conv2d(in_channels=16*2*2, out_channels=32*2*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            CustomNorm2d(64*2),
            nn.ReLU(),
            ResNetBlock(64*2),
            SE2DIdentity(128, r=32),

            nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=(3, 1), stride=(1, 1), padding=(4, 0), dilation=(4, 1), groups=64*2),
        )

        # NEW: higher-order temporal encoder on top of conv_mainA
        self.temporal_encoder = DilatedTemporalEncoder(channels=128)
        
        self.temp_gate = TemporalGate(c=128, k=5)

        # ----- z path (tiny MLP to 32-d) -----
        self.linext_pos = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
#998            nn.Dropout(p=0.2),
            nn.Linear(16, 32),
        )

        # ----- z gate -----
        self.z_gate = ZGateFiLM(z_dim=32, h_dim=256, scale=0.25)  # keep your class; consider bounded variant if needed
#z702        self.z_gate = ZGateCentered(z_dim=32, h_dim=256)  # keep your class; consider bounded variant if needed

        # ----- fusion + heads -----
        self.fusion = nn.Sequential(
            NoisyLinear3(256, 512, std_init=std_init),
#1004            NoisyLinear3(288, 512, std_init=std_init),
            nn.ReLU(),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear3(512, 256, std_init=std_init),
            nn.ReLU(),
            NoisyLinear3(256, self.action_size, std_init=std_init)   # use action_size
        )

        self.value_stream = nn.Sequential(
            NoisyLinear3(512, 256, std_init=std_init),
            nn.ReLU(),
            NoisyLinear3(256, 1, std_init=std_init)
        )

        # cache noisy modules
        self._reset_noise_modules = [m for m in self.modules() if m is not self and hasattr(m, 'reset_noise')]

        # one-time debug flag
        self._dbg_printed_zch = False

        # safety: head/out matches action_size
        out_feats = self.advantage_stream[-1].out_features
        assert out_feats == self.action_size, f"adv head {out_feats} != action_size {self.action_size}"

    def forward(self, x: T.Tensor, log: bool = False, adv_only: bool = False, training: bool = True) -> T.Tensor:
        """
        x: [B, 15, T+1, 1]
        channel 0, time 0 holds scalar z; all (t>=1) for ch-0 must be zero/ignored.
        """
        # ----- extract z (scalar at t=0) & drop t=0 for conv stream -----
        z  = x[:, 0, 0, :]          # [B, 1]
        xA = x[:, :, 1:, :].contiguous()  # [B, 15, T, 1]

        # HARD GUARD: ensure conv never sees z over time
        xA = xA * self._zmask              # zero-out channel 0 for all t>=1 (no in-place)

        # ----- split TA vs state -----
        ta = xA[:, :10, :, :]          # 0..9
#        st = xA[:, 10:20, :, :]        # 10..15 (in_pos, pos_frac, hold, sell, buy, ramp)
        st = xA[:, 10:19, :, :]        # 10..15 (in_pos, pos_frac, hold, sell, buy, ramp)
#1000        st = xA[:, 10:16, :, :]        # 10..15 (in_pos, pos_frac, hold, sell, buy, ramp)

        # group norms
        group_1 = self.group1(ta[:, :6,  :, :])   # 0..5
        group_2 = self.group2(ta[:, 6:7, :, :])   # 6
        group_3 = self.group3(ta[:, 7:9, :, :])   # 7..8
        group_4 = self.group4(ta[:, 9:10,:, :])   # 9
        group_state = self.group_state(st)        # 10..15

        # concat back to 16 channels
        xA = T.cat([group_1, group_2, group_3, group_4, group_state], dim=1)

        # ----- conv → temp gate → LSTM -----
        xA = self.conv_mainA(xA)      # [B, 128, T, 1]
        ################################################################################################################
        xA = self.temporal_encoder(xA)        # [B, 128, T, 1] (higher-order temporal state)
        ################################################################################################################
        xA = xA.squeeze(-1)           # [B, 128, T]
        xA = self.temp_gate(xA)
        # ----- replace LSTM with global pooling → 256-d -----
        gap = xA.mean(dim=-1)         # [B, 128]
        gmp, _ = xA.max(dim=-1)       # [B, 128]
        lstm_out = T.cat([gap, gmp], dim=1)   # [B, 256]  (drop-in for former biLSTM output)

        # ----- z path (center z before embed) -----
#1004        z_centered = z - 1.0          # z ~ [0,2] → [-1,1]
#1004        z_emb = self.linext_pos(z_centered)  # [B, 32]

        # ----- fuse & heads -----
#1004        lstm_out = self.z_gate(lstm_out, z_emb)        # [B, 256]
        comb = lstm_out
#1004        comb = T.cat([lstm_out, z_emb], dim=1)         # [B, 288]
        comb = self.fusion(comb)

        advantage = self.advantage_stream(comb)        # [B, action_size]
        if adv_only:
            return advantage
        value = self.value_stream(comb)                # [B, 1]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
    
    def reset_noise(self):
        """Reset all noisy layers using a cached list of modules."""
        for module in self._reset_noise_modules:
            module.reset_noise()
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))

##############################################################################################################################

class AffineRewardNorm(nn.Module):
    """
    Reward normalizer with affine transform:
        r' = a * r + b,
        a = target_std / std,  b = target_mean - a * mean
    - Updates running mean/var via EMA on raw rewards.
    - Works for 1-step (default) or n-step aggregated rewards.
    - Callable: forward(r, update=True, n_step=None, gamma=None)
        * update=False to freeze stats during eval.
    """

    def __init__(self, target_mean=0.0, target_std=1.0, momentum=1e-3, eps=1e-8, clip=3.0, freeze=False):
        super().__init__()
        # Running stats as buffers so .to(device) moves them
        self.register_buffer("mu",  T.zeros(1))
        self.register_buffer("var", T.ones(1))
        self.m       = float(momentum)
        self.eps     = float(eps)
        self.clip    = None if clip is None else float(clip)
        self.freeze  = bool(freeze)

        # Targets as buffers (so device/dtype follow the module)
        self.register_buffer("tgt_mean", T.tensor(float(target_mean)))
        self.register_buffer("tgt_std",  T.tensor(float(target_std)))

    @T.no_grad()
    def update(self, r: T.Tensor):
        """EMA update from RAW rewards r (any float shape)."""
        r = r.detach()
        if not r.is_floating_point():
            r = r.float()
        # Compute batch mean/var over all elements; unbiased=False for EMA stability
        bmean = r.mean()
        bvar  = r.var(unbiased=False)
        self.mu .mul_(1.0 - self.m).add_(self.m * bmean)
        self.var.mul_(1.0 - self.m).add_(self.m * bvar)

    def coeffs(self):
        """Return (a, b) for current running stats and targets."""
        std = self.var.sqrt().clamp_min(self.eps)
        a = self.tgt_std / std
        b = self.tgt_mean - a * self.mu
        return a, b

    def transform(self, r: T.Tensor):
        """Apply affine transform to rewards (no stats update)."""
        if not r.is_floating_point():
            r = r.float()
        a, b = self.coeffs()
        out = a * r + b
        if self.clip is not None:
            out = out.clamp_(-self.clip, self.clip)
        return out
        
    def transform_no_clip(self, r: T.Tensor):
        if not r.is_floating_point():
            r = r.float()
        a, b = self.coeffs()
        return a * r + b

        
    def inverse(self, r_norm: T.Tensor):
        """Inverse transform r = (r' - b) / a."""
        if not r_norm.is_floating_point():
            r_norm = r_norm.float()
        a, b = self.coeffs()
        return (r_norm - b) / a

    def transform_nstep(self, r_n: T.Tensor, n: int, gamma: float):
        """
        If r_n = sum_{i=0}^{n-1} gamma^i * r_{t+i}, then
            r_n' = a * r_n + b * sum_{i=0}^{n-1} gamma^i
                 = a * r_n + b * ((1 - gamma**n) / (1 - gamma)), gamma!=1
        """
        if not r_n.is_floating_point():
            r_n = r_n.float()
        a, b = self.coeffs()
        if gamma == 1.0:
            geom = float(n)
        else:
            geom = (1.0 - (gamma ** n)) / (1.0 - gamma)
        out = a * r_n + b * r_n.new_tensor(geom)
        if self.clip is not None:
            out = out.clamp_(-self.clip, self.clip)
        return out

    def set_targets(self, mean: float = None, std: float = None):
        """Optionally change target mean/std on the fly."""
        if mean is not None:
            self.tgt_mean.fill_(float(mean))
        if std is not None:
            self.tgt_std.fill_(float(std))

    def reset_stats(self, mean: float = 0.0, var: float = 1.0):
        """Reset running stats."""
        self.mu.fill_(float(mean))
        self.var.fill_(float(var))

    def forward(self, r: T.Tensor, update: bool = True, n_step: int = None, gamma: float = None):
        """
        Callable interface:
          - 1-step: groupr(r)  -> normalized rewards
          - n-step: groupr(r, n_step=N, gamma=gamma)
          - Set update=False to avoid EMA updates (e.g., eval)
        """
        if update and not self.freeze:
            self.update(r)
        if n_step is None:
            return self.transform(r)
        if gamma is None:
            raise ValueError("gamma must be provided when using n_step.")
        return self.transform_nstep(r, n_step, gamma)

class C51DuelDQNAgentPER():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 b_step = 16,
                 n_step = 3,
                 ndays = 16,
                 replace=1000, nenvs=64, chkpt_dir='tmp/dueling_ddqn'):
        np.random.seed(2021)
        self.gamma = gamma
        self.gamma_n_step = gamma ** n_step
        self.n_step = n_step
        self.epsilon = epsilon
        self.lr = lr
        adam_eps = 0.005 / batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.loss_c = deque(maxlen=1000)
        
        self.USE_NOISY_NET = True
        beta_start = 0.45# 0.4
        beta_frames = 15_000_000 # past 25M steps you use full importance-sampling correction in PER
        self.beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        
        obs_dim = 53
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.memory = PrioritizedReplayBufferNew(obs_dim, size=mem_size, n_step=n_step, batch_size=self.batch_size, alpha=0.6, nenvs=nenvs, ndays=ndays, dev=self.device)
        print('...allocating a PrioritizedReplayBuffer')
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        self.q_eval, self.q_next = None, None

        self.q_eval = EfficientC51DuelingDeepQNetworkCV2J23(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name='lunar_lander_dueling_ddqn_q_eval',
                                         chkpt_dir=self.chkpt_dir).to(self.device)

        self.q_next = EfficientC51DuelingDeepQNetworkCV2J23(self.lr, self.n_actions,
                                         input_dims=self.input_dims,
                                         name='lunar_lander_dueling_ddqn_q_next',
                                         chkpt_dir=self.chkpt_dir).to(self.device)

        
        # ✅ Speed optimization: move to GPU and set memory layout
        self.q_eval = self.q_eval.to(self.device, memory_format=T.channels_last)
        self.q_next = self.q_next.to(self.device, memory_format=T.channels_last)

        self.q_next.load_state_dict(self.q_eval.state_dict()) #4/30
        self.q_next.eval() #4/30

#        self.optimizer = optim.AdamW(self.q_eval.parameters(), lr=lr, weight_decay=0.01)  # AdamW helps generalization
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr, eps=adam_eps)#, weight_decay=0.01)

        # Create the dynamic group norm layer with batching of computations.
        self.groupr = AffineRewardNorm(target_mean=0.0, target_std=1.0, momentum=1e-3, clip=10.0).to(self.device)
#        self.groupr = AffineRewardNorm(target_mean=0.0, target_std=1.0, momentum=1e-2, clip=3.0).to(self.device)

        # in __init__
        self.per_update_every = 32
        self._per_pending = []  # list of (indices, priorities_cpu)
        self._learn_counter = 1
        self.reward_scale = 1. # 3.0 #5.0

    def _compute_dqn_loss(self, samples, gamma_n):
        states      = samples["obs"]
        next_states = samples["next_obs"]
        actions     = samples["acts"]                 # [B,1]
        rewards_raw = samples["rews"].view(-1, 1)     # [B,1]
        dones       = samples["done"].bool().view(-1, 1)
    
        # Online Q(s,a) with grad
        q_all  = self.q_eval(states)
        curr_q = q_all.gather(1, actions)             # [B,1]
    
        with T.no_grad():
            self.q_next.reset_noise()
            next_q_all = self.q_next(next_states)
    
            next_adv_online = self.q_eval(next_states, adv_only=True)
            best_actions = next_adv_online.argmax(dim=1, keepdim=True)
    
            next_q = next_q_all.gather(1, best_actions)
            not_done = (~dones).float()
    
            y_raw = rewards_raw + gamma_n * next_q * not_done   # [B,1]
    
#            if not self.groupr.freeze:
            self.groupr.update(y_raw)
    
            # normalized target (no grad)
            y = self.groupr.transform_no_clip(y_raw)
    
        # normalized prediction (keep grad!)
        q = self.groupr.transform_no_clip(curr_q)
    
        elem_loss = F.smooth_l1_loss(q, y, reduction="none")    # [B,1]
        return elem_loss, curr_q, y_raw

    
    def _compute_dqn_loss_123035B(self, samples, gamma_n):
        states      = samples["obs"]
        next_states = samples["next_obs"]
        actions     = samples["acts"]                 # [B,1]
        rewards_raw = samples["rews"].view(-1, 1)     # [B,1]
        dones       = samples["done"].bool().view(-1, 1)
    
        # --- Online Q(s,a) (THIS must keep grad) ---
        current_q_values = self.q_eval(states)              # [B, n_actions]
        curr_q = current_q_values.gather(1, actions)        # [B,1]
    
        # --- Target side (no grad) ---
        with T.no_grad():
            self.q_next.reset_noise()
            next_q_values = self.q_next(next_states)        # [B, n_actions]
    
            next_adv_online = self.q_eval(next_states, adv_only=True)   # [B, n_actions]
            best_actions = next_adv_online.argmax(dim=1, keepdim=True)  # [B,1]
    
            next_q_target = next_q_values.gather(1, best_actions)       # [B,1]
            not_done = (~dones).float()
    
            y_raw = rewards_raw + gamma_n * next_q_target * not_done    # [B,1]
    
            # update stats on y_raw (optionally respect freeze)
            # if not self.groupr.freeze:
            self.groupr.update(y_raw)
    
            # normalized target (no grad is fine)
            y = self.groupr.transform(y_raw)                             # [B,1]
    
        # --- Normalize prediction symmetrically (KEEP grad!) ---
        q = self.groupr.transform(curr_q)                                # [B,1]
    
        loss = F.smooth_l1_loss(q, y, reduction="none")
        return loss, curr_q, y_raw


    def _compute_dqn_loss_123025(self, samples, gamma_n):
        """
        DDQN loss with reward affine-normalization.
        Assumes samples["rews"] is an n-step discounted return for n=self.n_step.
        gamma_n should be (gamma_1step ** n_step).
        """
        states      = samples["obs"]
        next_states = samples["next_obs"]
        actions     = samples["acts"]       # [B,1]
        rewards_raw = samples["rews"]       # [B] or [B,1]
        dones       = samples["done"]       # [B] or [B,1] bool
    
        rewards_raw = rewards_raw.view(-1, 1)  # [B,1]
    
        # --- Reward normalization (respect freeze) ---
#        with T.no_grad():
#            if not self.groupr.freeze:
#                self.groupr.update(rewards_raw)
    
        with T.no_grad():
            self.groupr.update(rewards_raw)               # EMA stats from raw
        rewards = self.groupr.transform(rewards_raw) # affine: a*r, then (optional) clip
    
        # --- Online Q(s,a) ---
        current_q_values = self.q_eval(states)         # [B, n_actions]
        curr_q = current_q_values.gather(1, actions)   # [B,1]
    
        with T.no_grad():
            self.q_next.reset_noise()
            next_q_values   = self.q_next(next_states)                 # [B, n_actions]
            next_adv_online = self.q_eval(next_states, adv_only=True)  # [B, n_actions]
            best_actions    = next_adv_online.argmax(dim=1, keepdim=True)  # [B,1]
            next_q_target   = next_q_values.gather(1, best_actions)        # [B,1]
    
            not_done = (~dones.bool()).float().view(-1, 1)
            target_q = rewards + gamma_n * next_q_target * not_done
    
        loss = F.smooth_l1_loss(curr_q, target_q, reduction="none")
        return loss, curr_q, target_q

    
    def _compute_dqn_loss_old_122725(self, samples, gamma):
        """DDQN loss with reward affine-normalization aligned to input pre-norm."""
        # Unpack (already on GPU if your sampler does that)
        states      = samples["obs"]
        next_states = samples["next_obs"]
        actions     = samples["acts"]            # shape [B,1] (long)
        rewards_raw = samples["rews"]            # shape [B,1] or [B]
        dones       = samples["done"]            # bool [B,1] or [B]
    
        # ---- Normalize rewards (update on RAW, then transform) ----
        rewards_raw = rewards_raw.view(-1, 1)             # [B,1]

        with T.no_grad():
            self.groupr.update(rewards_raw)               # EMA stats from raw
        rewards = self.groupr.transform(rewards_raw) # affine: a*r, then (optional) clip
        
        # ---- Online Q(s,a) ----
        current_q_values = self.q_eval(states)            # [B, n_actions]
        curr_q = current_q_values.gather(1, actions)      # [B,1]
    
        with T.no_grad():
            self.q_next.reset_noise()                     # DDQN: target net eval
            next_q_values  = self.q_next(next_states)     # [B, n_actions]
            next_adv_online = self.q_eval(next_states, adv_only=True)  # online for argmax
            best_actions   = next_adv_online.argmax(dim=1, keepdim=True)  # [B,1]
            next_q_target  = next_q_values.gather(1, best_actions)        # [B,1]
    
            not_done = (~dones.bool()).float().view(-1, 1)
            target_q = rewards + gamma * next_q_target * not_done         # [B,1]
    
        loss = F.smooth_l1_loss(curr_q, target_q, reduction="none")
        return loss, curr_q, target_q
            
    
    def _compute_dqn_loss_old(self, samples, gamma):
        """DDQN loss with reward affine-normalization aligned to input pre-norm."""
        # Unpack (already on GPU if your sampler does that)
        states      = samples["obs"]
        next_states = samples["next_obs"]
        actions     = samples["acts"]            # shape [B,1] (long)
        rewards_raw = samples["rews"]            # shape [B,1] or [B]
        dones       = samples["done"]            # bool [B,1] or [B]
    
        # ---- Normalize rewards (update on RAW, then transform) ----
        rewards_raw = rewards_raw.view(-1, 1)             # [B,1]
        with T.no_grad():
            self.groupr.update(rewards_raw)               # EMA stats from raw
        rewards = self.groupr.transform(rewards_raw)      # affine: a*r + b, then (optional) clip
    
        # ---- Online Q(s,a) ----
        current_q_values = self.q_eval(states)            # [B, n_actions]
        curr_q = current_q_values.gather(1, actions)      # [B,1]
    
        with T.no_grad():
            self.q_next.reset_noise()                     # DDQN: target net eval
            next_q_values  = self.q_next(next_states)     # [B, n_actions]
            next_adv_online = self.q_eval(next_states, adv_only=True)  # online for argmax
            best_actions   = next_adv_online.argmax(dim=1, keepdim=True)  # [B,1]
            next_q_target  = next_q_values.gather(1, best_actions)        # [B,1]
    
            not_done = (~dones.bool()).float().view(-1, 1)
            target_q = rewards + gamma * next_q_target * not_done         # [B,1]
    
        loss = F.smooth_l1_loss(curr_q, target_q, reduction="none")
        return loss, curr_q, target_q
            

    def choose_actionc(self, observation, show_action=True):
        action = np.random.choice(self.action_space)
        if show_action:
            print('action=', action)
        self.decrement_epsilon()
        return action
        
    def choose_action(self, observation, show_action=True, training=True):
        if np.random.random() >= self.epsilon:
            self.q_eval.reset_noise()
            state = T.from_numpy([observation]).float().to(self.device)
    
            # --- Use running BN stats & disable dropout while acting ---
            was_training = self.q_eval.training
            self.q_eval.eval()
            with T.no_grad():
                advantage = self.q_eval(state, adv_only=True)   # training flag not needed
                action = T.argmax(advantage, dim=1).item()
            self.q_eval.train(was_training)   # restore previous mode
            # -----------------------------------------------------------
        else:
            action = np.random.choice(self.action_space)
    
        self.decrement_epsilon()
        return action

    def choose_actions(self, observations, n_envs, show_action=True, training=True):
        if np.random.random() >= self.epsilon:
            self.q_eval.reset_noise()
            states = T.from_numpy(observations).float().to(self.device)
    
            # --- eval() just for inference, then restore ---
            was_training = self.q_eval.training
            self.q_eval.eval()
            with T.no_grad():
                advantages = self.q_eval(states, adv_only=True)
                actions = T.argmax(advantages, dim=1).cpu().numpy()
            self.q_eval.train(was_training)
            # ------------------------------------------------
        else:
            actions = np.random.randint(0, self.n_actions, size=n_envs)
    
        self.decrement_epsilon()
        return actions
    
    def pred_action(self, observation, training=False):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.device)
    
        # --- eval() during prediction, then restore ---
        was_training = self.q_eval.training
        self.q_eval.eval()
        with T.no_grad():
            advantage = self.q_eval(state, adv_only=True)
            action = T.argmax(advantage, dim=1).item()
        self.q_eval.train(was_training)
        # ---------------------------------------------
        return action

    def store_transition_new(self, state, action, reward, state_, done, rewardn):
        transition = [state, action, reward, state_, done, rewardn]
        self.memory.store(*transition)
        
    def store_transition_news(self, state, action, reward, state_, done, rewardn, env_id):
        transition = [state, action, reward, state_, done, rewardn, env_id]
        self.memory.store(*transition)

    def replace_target_network(self):
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            print('replace_target_network @ ', self.learn_step_counter)
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)
 
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self, fridx):
        self.q_eval.train(True)        # make sure BN/dropout behave for training
        # Get beta value for this frame
        beta = self.beta_by_frame(fridx)
        
        # Sample batch and compute loss in one operation
        samples = self.memory.sample_batch(beta, self.batch_size)
        weights = samples["weights"]  # Already on GPU
        indices = samples["indices"]
        
        # Fuse optimizer operations - clear gradients right before backward
        self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        if self._learn_counter == 2000:
            self.groupr.freeze = True
        
        # Compute loss in one go
        elem_loss, curr_q, target_q = self._compute_dqn_loss(samples, self.gamma_n_step)
        loss = (elem_loss * weights).mean()
        # Backprop
        loss.backward()
        
        # Clip gradients and update in one efficient block
        clip_grad_norm_(self.q_eval.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- SOFT UPDATE: move target toward online every learn step ----
        if (self._learn_counter % 4) == 0:
            self.soft_update(tau=0.02)   # try 0.003–0.01
        
        # Compute priorities more efficiently
        with T.no_grad():
#            td = (curr_q - target_q).abs().view(-1)
#            new_priorities = (td + 1e-8).sqrt()   # or td.log1p()
            td_errors = (curr_q - target_q).abs().clamp(max=10.0).view(-1)
            new_priorities = (td_errors + 1e-8)
        
        if (self._learn_counter % self.per_update_every) != 0:
            # store indices + priorities tensor for later (still no sync)
            self._per_pending.append((indices, new_priorities))
#            print(f"1028...storing {self._learn_counter}")
        else:
            # flush pending + current in one go
            pend = self._per_pending
            self._per_pending = []
            pend.append((indices, new_priorities))
        
            # one batched sync instead of per-step sync
            all_idx = []
            all_pr  = []
            for idx, prt in pend:
                all_idx.append(idx)
                all_pr.append(prt)
        
            # concatenate on GPU, then one transfer
            pr_cpu = T.cat(all_pr).detach().to("cpu", non_blocking=True).numpy()
            idx_cpu = np.concatenate(all_idx)  # indices are usually already CPU arrays
        
            self.memory.update_priorities(idx_cpu, pr_cpu)
#1100            self._learn_counter = 0
#            print(f"1047...syncing {self._learn_counter}")

        if self._learn_counter % 2000 == 0:
            with T.no_grad():
                a, b = self.groupr.coeffs()
                std = self.groupr.var.sqrt()
            print("step=", self._learn_counter, "a=", float(a), "b=", float(b), "std=", float(std), "freeze=", self.groupr.freeze)
#200            a, b = self.groupr.coeffs()
#200            print("step=", self._learn_counter, "rnorm:", float(a), float(b), "freeze:", self.groupr.freeze)
        
        self._learn_counter += 1
        
        # Store loss value (optional: consider using a deque with max length for efficiency)
        self.loss_c.append(loss.item())

        return loss.item()  # Optional: return loss for monitoring
        
    def learn_old(self, fridx):
        self.q_eval.train(True)        # make sure BN/dropout behave for training
        # Get beta value for this frame
        beta = self.beta_by_frame(fridx)
        
        # Sample batch and compute loss in one operation
        samples = self.memory.sample_batch(beta, self.batch_size)
        weights = samples["weights"]  # Already on GPU
        indices = samples["indices"]
        
        # Fuse optimizer operations - clear gradients right before backward
        self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Compute loss in one go
        elem_loss, curr_q, target_q = self._compute_dqn_loss(samples, self.gamma_n_step)
        loss = (elem_loss * weights).mean()
        
        # Backprop
        loss.backward()
        
        # Clip gradients and update in one efficient block
        clip_grad_norm_(self.q_eval.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- SOFT UPDATE: move target toward online every learn step ----
        self.soft_update(tau=0.005)   # try 0.003–0.01
        
        # Compute priorities more efficiently
        with T.no_grad():
            td_errors = (curr_q - target_q).abs()
            
            # ---- Fixed cap (simple & fast). Tune 3–10 depending on reward scale. ----
            td_errors.clamp_(max=10.0)
            
            # Process on GPU as much as possible before transfer
            td_errors_flat = td_errors.detach().cpu().numpy().flatten()
            new_priorities = td_errors_flat + 1e-8
        
        # Update priorities
        self.memory.update_priorities(indices, new_priorities)
        
        # Store loss value (optional: consider using a deque with max length for efficiency)
        self.loss_c.append(loss.item())
        
        return loss.item()  # Optional: return loss for monitoring

    def soft_update(self, tau):
        with T.no_grad():
            for tp, sp in zip(self.q_next.parameters(), self.q_eval.parameters()):
                tp.lerp_(sp, tau)


    def soft_update_old(self, tau=0.005):
        """Polyak averaging: q_next ← τ * q_eval + (1-τ) * q_next."""
        with T.no_grad():
            # parameters
            for tp, sp in zip(self.q_next.parameters(), self.q_eval.parameters()):
                tp.mul_(1.0 - tau).add_(sp, alpha=tau)
            # buffers (e.g., BatchNorm running stats)
            for (tn, tb), (sn, sb) in zip(self.q_next.named_buffers(), self.q_eval.named_buffers()):
                if tb.dtype.is_floating_point:
                    tb.mul_(1.0 - tau).add_(sb, alpha=tau)
                else:
                    tb.copy_(sb)  # ints like num_batches_tracked

    def sample_noise(self):
        # NoisyNet: reset noise
        self.q_eval.sample_noise()

        
