import math

import torch
import torch.nn as nn

# Kornia version
# def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     r"""Convert an image from RGB to HSV.
#
#     .. image:: _static/image/rgb_to_hsv.png
#
#     The image data is assumed to be in the range of (0, 1).
#
#     Args:
#         image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
#         eps: scalar to enforce numarical stability.
#
#     Returns:
#         HSV version of the image with shape of :math:`(*, 3, H, W)`.
#         The H channel values are in the range 0..2pi. S and V are in the range 0..1.
#
#     .. note::
#        See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
#        color_conversions.html>`__.
#
#     Example:
#         >>> input = torch.rand(2, 3, 4, 5)
#         >>> output = rgb_to_hsv(input)  # 2x3x4x5
#     """
#     if not isinstance(image, torch.Tensor):
#         raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")
#
#     if len(image.shape) < 3 or image.shape[-3] != 3:
#         raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")
#
#     # The first or last occurrence is not guaranteed before 1.6.0
#     # https://github.com/pytorch/pytorch/issues/20414
#     maxc, _ = image.max(-3)
#     maxc_mask = image == maxc.unsqueeze(-3)
#     _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
#     minc: torch.Tensor = image.min(-3)[0]
#
#     v: torch.Tensor = maxc  # brightness
#
#     deltac: torch.Tensor = maxc - minc
#     s: torch.Tensor = deltac / (v + eps)
#
#     # avoid division by zero
#     deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)
#
#     maxc_tmp = maxc.unsqueeze(-3) - image
#     rc: torch.Tensor = maxc_tmp[..., 0, :, :]
#     gc: torch.Tensor = maxc_tmp[..., 1, :, :]
#     bc: torch.Tensor = maxc_tmp[..., 2, :, :]
#
#     h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)
#     # print('rgb_to_hsv', 'h', h.shape, max_indices.max(), max_indices.min())
#
#     h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
#     h = h.squeeze(-3)
#     h = h / deltac
#
#     h = (h / 6.0) % 1.0
#
#     h = 2 * math.pi * h
#
#     return torch.stack([h, s, v], dim=-3)

# torchvision version
def rgb_to_hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    h = 2 * math.pi * h

    return torch.stack((h, s, maxc), dim=-3)

# Kornia version
# def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
#     r"""Convert an image from HSV to RGB.
#
#     The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.
#
#     Args:
#         image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
#
#     Returns:
#         RGB version of the image with shape of :math:`(*, 3, H, W)`.
#
#     Example:
#         >>> input = torch.rand(2, 3, 4, 5)
#         >>> output = hsv_to_rgb(input)  # 2x3x4x5
#     """
#     if not isinstance(image, torch.Tensor):
#         raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")
#
#     if len(image.shape) < 3 or image.shape[-3] != 3:
#         raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")
#
#     # print('h', image[..., 0, :, :].max(), image[..., 0, :, :].min())
#     # print('s', image[..., 1, :, :].max(), image[..., 1, :, :].min())
#     # print('v', image[..., 2, :, :].max(), image[..., 2, :, :].min())
#     h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
#     s: torch.Tensor = image[..., 1, :, :]
#     v: torch.Tensor = image[..., 2, :, :]
#
#     # h = torch.clamp(h, 0., 1.)
#     # s = torch.clamp(s, 0., 1.)
#     # v = torch.clamp(v, 0., 1.)
#
#     # print('h', h.max(), h.min())
#
#     hi: torch.Tensor = torch.floor(h * 6) % 6
#     f: torch.Tensor = ((h * 6) % 6) - hi
#     # hi: torch.Tensor = torch.floor(h * 6)
#     # f: torch.Tensor = h * 6 - hi
#     one: torch.Tensor = torch.tensor(1.0).to(image.device)
#     p: torch.Tensor = v * (one - s)
#     q: torch.Tensor = v * (one - f * s)
#     t: torch.Tensor = v * (one - (one - f) * s)
#
#     hi = hi.long()
#
#     # r = torch.stack((v, q, p, p, t, v), dim=-3)
#     # g = torch.stack((t, v, v, q, p, p), dim=-3)
#     # b = torch.stack((p, p, t, v, v, q), dim=-3)
#     # print(r.shape, hi.min(), hi.max())
#     # r = torch.gather(r, -3, hi.unsqueeze(-3))
#     # g = torch.gather(g, -3, hi.unsqueeze(-3))
#     # b = torch.gather(b, -3, hi.unsqueeze(-3))
#     # out = torch.cat((r, g, b), dim=-3)
#
#     indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
#     out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
#     out = torch.gather(out, -3, indices)
#
#     return out

# torchvision version
def hsv_to_rgb(img):
    h, s, v = img.unbind(dim=-3)
    h = h / (2 * math.pi)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class RgbToHsv(nn.Module):
    r"""Convert an image from RGB to HSV.
    The image data is assumed to be in the range of (0, 1).
    Args:
        eps: scalar to enforce numarical stability.
    Returns:
        HSV version of the image.
    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hsv(image, self.eps)


class HsvToRgb(nn.Module):
    r"""Convert an image from HSV to RGB.
    H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.
    Returns:
        RGB version of the image.
    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsvToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hsv_to_rgb(image)