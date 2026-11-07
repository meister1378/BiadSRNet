"""BIADSRNet (BIASRNet + Diffusion) -- a small conditional diffusion
refinement stage that sits *after* a frozen BIASRNet backbone, sharpening
perceptual texture without changing the overall structure BIASRNet already
recovered.

Why a second, separate model instead of a bigger backbone: a deterministic
L1/SmoothL1-trained network (BIASRNet) tends to over-smooth fine texture --
pushing further on alignment/fusion architecture plateaus around the same
PSNR. A diffusion model instead hallucinates plausible fine texture, which is
a genuine win only when the deployment target is human viewing rather than a
downstream PSNR/detection metric, since the added texture is not guaranteed
to be physically true (relevant caveat for thermal imagery -- re-check this
before any safety-relevant deployment).

Method: rather than running the full T-step reverse diffusion process from
pure noise, seed the process at an intermediate noise level using the backbone's (BIASRNet's)
output as a proxy for x0, then run only the remaining `t_start -> 0` reverse
steps. This keeps the backbone's already-correct global structure intact and
lets diffusion spend its few steps only on fine detail -- far cheaper than
full generative sampling.

Training stays stage-wise: biasrnet/train.py trains the backbone first, then
train.py here trains only the diffusion UNet with that backbone frozen (see
train.py). This module (BIADSRNet) exists to combine the two for inference,
not as a jointly-trained model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biasrnet_model import BIASRNet  # local copy -- see biasrnet_model.py's docstring


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (c // self.num_heads) ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q * scale, k).softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class UNetDiffusion(nn.Module):
    """Input: 2-channel (noisy HR, BIASRNet-output conditioning) stack,
    [-1, 1] range. Output: predicted noise, 1-channel, same size.
    Fully convolutional -- works at any resolution divisible by 8."""

    def __init__(self, base_ch=48, time_dim=256):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.time_embed = SinusoidalPosEmb(base_ch)
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.init_conv = nn.Conv2d(2, c1, 3, padding=1)

        self.enc1a = ResBlock(c1, c1, time_dim)
        self.enc1b = ResBlock(c1, c1, time_dim)
        self.down1 = Downsample(c1, c2)

        self.enc2a = ResBlock(c2, c2, time_dim)
        self.enc2b = ResBlock(c2, c2, time_dim)
        self.down2 = Downsample(c2, c3)

        self.enc3a = ResBlock(c3, c3, time_dim)
        self.enc3b = ResBlock(c3, c3, time_dim)
        self.down3 = Downsample(c3, c4)

        self.mid1 = ResBlock(c4, c4, time_dim)
        self.mid_attn = SelfAttention2d(c4)
        self.mid2 = ResBlock(c4, c4, time_dim)

        self.up3 = Upsample(c4, c3)
        self.dec3a = ResBlock(c3 * 2, c3, time_dim)
        self.dec3b = ResBlock(c3 * 2, c3, time_dim)

        self.up2 = Upsample(c3, c2)
        self.dec2a = ResBlock(c2 * 2, c2, time_dim)
        self.dec2b = ResBlock(c2 * 2, c2, time_dim)

        self.up1 = Upsample(c2, c1)
        self.dec1a = ResBlock(c1 * 2, c1, time_dim)
        self.dec1b = ResBlock(c1 * 2, c1, time_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))

        h = self.init_conv(x)
        h = self.enc1a(h, t_emb); skip1a = h
        h = self.enc1b(h, t_emb); skip1b = h
        h = self.down1(h)

        h = self.enc2a(h, t_emb); skip2a = h
        h = self.enc2b(h, t_emb); skip2b = h
        h = self.down2(h)

        h = self.enc3a(h, t_emb); skip3a = h
        h = self.enc3b(h, t_emb); skip3b = h
        h = self.down3(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        h = self.up3(h)
        h = self.dec3a(torch.cat([h, skip3b], dim=1), t_emb)
        h = self.dec3b(torch.cat([h, skip3a], dim=1), t_emb)

        h = self.up2(h)
        h = self.dec2a(torch.cat([h, skip2b], dim=1), t_emb)
        h = self.dec2b(torch.cat([h, skip2a], dim=1), t_emb)

        h = self.up1(h)
        h = self.dec1a(torch.cat([h, skip1b], dim=1), t_emb)
        h = self.dec1b(torch.cat([h, skip1a], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


class GaussianDiffusion:
    """Standard DDPM (Ho et al. 2020) linear schedule, epsilon-prediction,
    plus `sample_from_cond` for the BSRD-style partial/intermediate-start
    reverse process seeded by a backbone's output."""

    def __init__(self, timesteps=1000, device='cuda'):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    @staticmethod
    def _extract(a, t, shape):
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omacp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_omacp * noise

    def p_losses(self, model, x0, cond, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(torch.cat([x_t, cond], dim=1), t)
        return F.mse_loss(pred_noise, noise)

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omacp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_omacp * noise) / sqrt_acp

    @torch.no_grad()
    def p_sample(self, model, x_t, cond, t):
        pred_noise = model(torch.cat([x_t, cond], dim=1), t)
        x0_hat = self.predict_start_from_noise(x_t, t, pred_noise).clamp(-1, 1)
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_hat + coef2 * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().reshape(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_from_cond(self, model, cond, t_start):
        """BSRD-style: seed x_{t_start} using `cond` (the backbone's output)
        as a proxy for x0, then only run the remaining t_start -> 0 reverse
        steps instead of the full T-step process."""
        b = cond.shape[0]
        device = cond.device
        t_init = torch.full((b,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(cond)
        x_t = self.q_sample(cond, t_init, noise)
        for step in reversed(range(t_start + 1)):
            t_batch = torch.full((b,), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, cond, t_batch)
        return x_t


class BIADSRNet(nn.Module):
    """BIADSRNet = BIASRNet + Diffusion: end-to-end inference wrapper
    combining the frozen BIASRNet backbone with the diffusion refinement
    stage under one name. Training stays stage-wise (biasrnet/train.py for
    the backbone, train.py here for the diffusion refiner with the backbone
    frozen) -- this class exists for inference/eval convenience, not as a
    jointly-trained module."""

    def __init__(self, backbone, unet, diffusion, t_start=200):
        super().__init__()
        self.backbone = backbone
        self.unet = unet
        self.diffusion = diffusion
        self.t_start = t_start

    @classmethod
    def from_checkpoints(cls, backbone_ckpt, diffusion_ckpt, device, n_frames=5,
                          base_ch=48, timesteps=1000, t_start=200):
        backbone = BIASRNet(n_frames=n_frames, deform_groups=1)
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        unet = UNetDiffusion(base_ch=base_ch)
        unet.load_state_dict(torch.load(diffusion_ckpt, map_location=device))
        unet.to(device).eval()

        diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
        return cls(backbone, unet, diffusion, t_start=t_start)

    @torch.no_grad()
    def forward(self, burst):
        backbone_out = self.backbone(burst).clamp(0, 1)
        cond = backbone_out * 2 - 1
        refined = self.diffusion.sample_from_cond(self.unet, cond, self.t_start)
        return ((refined + 1) / 2).clamp(0, 1)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetDiffusion().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'UNetDiffusion parameters: {n_params:.2f}M')

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    x0 = torch.rand(2, 1, 256, 256, device=device) * 2 - 1
    cond = torch.rand(2, 1, 256, 256, device=device) * 2 - 1
    t = torch.randint(0, 1000, (2,), device=device)
    loss = diffusion.p_losses(model, x0, cond, t)
    print('training loss (random init):', loss.item())

    out = diffusion.sample_from_cond(model, cond, t_start=20)
    print('sample_from_cond output:', out.shape, out.min().item(), out.max().item())
