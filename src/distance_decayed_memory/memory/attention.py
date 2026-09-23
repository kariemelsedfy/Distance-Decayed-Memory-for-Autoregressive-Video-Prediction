"""Reference attention from query tokens to a policy's cache.

Queries come from the current chunk and keys from the cache, so a plain
concatenated-KV ``scaled_dot_product_attention`` suffices (§4.3). RoPE is
applied here at each token's (possibly fractional) position, and the optional
proportional-attention bias adds ``log v`` for a token of volume ``v``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from distance_decayed_memory.memory.rope import apply_rope


def cache_attention(
    q: torch.Tensor,
    q_pos: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_pos: torch.Tensor,
    weight: torch.Tensor | None = None,
    proportional: bool = True,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend ``q[..., Nq, D]`` over pre-RoPE ``k``/``v`` ``[..., N, D]``.

    ``mask`` (``[Nq, N]`` boolean, True = attend) is combined with the
    ``log v`` bias when both are given.
    """
    q = apply_rope(q, q_pos)
    k = apply_rope(k, k_pos)
    bias = None
    if proportional and weight is not None:
        bias = weight.log().to(q.dtype).expand(q.shape[-2], -1)
    if mask is not None:
        blocked = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
        blocked = blocked.masked_fill(~mask, float("-inf"))
        bias = blocked if bias is None else bias + blocked
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
