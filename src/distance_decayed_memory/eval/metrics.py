"""Per-frame image metrics: PSNR, SSIM, and LPIPS (TRACK_A_PLAN.md §7).

Frames are ``uint8`` or float tensors shaped ``[N, H, W, C]``; every function
returns one value per frame. LPIPS needs the optional ``lpips`` package (the
``eval`` extra) and downloads its AlexNet weights on first use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_unit(frames: torch.Tensor) -> torch.Tensor:
    """``[N, H, W, C]`` in ``[0, 1]`` float, from uint8 or ``[-1, 1]`` floats."""
    if frames.dtype == torch.uint8:
        return frames.float() / 255.0
    return ((frames.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB per frame (inputs as for :func:`_to_unit`)."""
    error = (_to_unit(prediction) - _to_unit(target)).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / error.clamp_min(1e-10))


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Structural similarity per frame (Wang et al. 2004), averaged over channels.

    Uses a Gaussian window with valid (unpadded) filtering, the standard
    constants ``C1 = 0.01²`` and ``C2 = 0.03²`` for data in ``[0, 1]``.
    """
    x = _to_unit(prediction).permute(0, 3, 1, 2)
    y = _to_unit(target).permute(0, 3, 1, 2)
    channels = x.shape[1]
    kernel = _gaussian_window(window, sigma, x.device).expand(channels, 1, -1, -1)

    def blur(values: torch.Tensor) -> torch.Tensor:
        return F.conv2d(values, kernel, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return value.mean(dim=(1, 2, 3))


class LPIPS:
    """Learned perceptual distance (AlexNet features), lower is closer."""

    def __init__(self, device=None, network: str = "alex") -> None:
        try:
            import lpips
        except ImportError as error:  # pragma: no cover - depends on the env
            raise ImportError(
                "LPIPS needs the 'lpips' package: pip install '.[eval]'"
            ) from error
        self.model = lpips.LPIPS(net=network, verbose=False).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = _to_unit(prediction).permute(0, 3, 1, 2) * 2 - 1
        y = _to_unit(target).permute(0, 3, 1, 2) * 2 - 1
        return self.model(x, y).flatten()


class FrameMetrics:
    """PSNR, SSIM, and (when available and requested) LPIPS for frame batches."""

    def __init__(self, device=None, use_lpips: bool = True) -> None:
        self.lpips = LPIPS(device) if use_lpips else None

    @torch.no_grad()
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        values = {"psnr": psnr(prediction, target), "ssim": ssim(prediction, target)}
        if self.lpips is not None:
            values["lpips"] = self.lpips(prediction, target)
        return {name: value.float().cpu() for name, value in values.items()}
