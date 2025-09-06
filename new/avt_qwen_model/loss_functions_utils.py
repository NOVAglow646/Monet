import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Literal


def collect_pooled_image_embeddings(
    inputs_embeds: torch.Tensor,
    seg: List[List[Tuple]],
    *,
    latent_size: int,
    method: Literal["gaussian", "chunk"] = "gaussian",
    sigma_scale: float = 0.5,
) -> List[torch.Tensor]:
    """
    Pool per-step image token embeddings to match latent_size.
    For each batch b and each step s in seg[b], produce latent_size pooled vectors.
    Return a list over batch; each entry is a 2D tensor [latent_size * t, d] aligned with ce_patch_vec[b].

    Args:
        inputs_embeds: Tensor [B, S, d], token embeddings after image features are scattered.
        seg: List[List[Tuple]], seg[b][s][0] gives positions (tensor/list/tuple of ints) of image tokens for step s.
             NOTE: seg[b] already excludes the question image; do NOT drop its first element.
        latent_size: number of latent tokens per step.
        method: "gaussian" (smooth kernel pooling) or "chunk" (contiguous chunk mean).
        sigma_scale: width of Gaussian kernel as a fraction of (k / latent_size). Ignored for "chunk".

    Returns:
        pooled_per_batch: List of Tensors, each shaped [latent_size * t, d].
    """
    assert inputs_embeds.dim() == 3, "inputs_embeds must be [B, S, d]"
    assert isinstance(seg, list) and isinstance(latent_size, int) and latent_size > 0
    B, S, d = inputs_embeds.shape
    device, dtype = inputs_embeds.device, inputs_embeds.dtype

    def _to_index_tensor(pos) -> torch.LongTensor:
        if torch.is_tensor(pos):
            return pos.to(device=device, dtype=torch.long)
        elif isinstance(pos, (list, tuple)):
            return torch.tensor(list(pos), device=device, dtype=torch.long)
        else:
            raise TypeError("Invalid seg position container; expect tensor/list/tuple of ints.")

    def _pool_step(Z: torch.Tensor, L: int) -> torch.Tensor:
        """Pool k image tokens [k, d] into L vectors [L, d]."""
        k = Z.size(0)
        if k == 0:
            return torch.zeros(L, d, device=Z.device, dtype=Z.dtype)
        if L == 1:
            return Z.mean(dim=0, keepdim=True)
        if k == 1:
            return Z.repeat(L, 1)

        if method == "chunk":
            # Split by contiguous chunks along token order
            # boundaries ~ round(linspace(0, k, L+1))
            boundaries = torch.linspace(0, k, steps=L + 1, device=Z.device)
            boundaries = torch.round(boundaries).to(torch.long).clamp_(0, k)
            outs = []
            for j in range(L):
                a = boundaries[j].item()
                b = boundaries[j + 1].item()
                if a >= b:
                    # fallback: nearest token to the center of this bin
                    center = ( (j + 0.5) * k / L ) - 0.5
                    nearest = int(round(center))
                    nearest = max(0, min(k - 1, nearest))
                    outs.append(Z[nearest:nearest + 1].mean(dim=0, keepdim=True))
                else:
                    outs.append(Z[a:b].mean(dim=0, keepdim=True))
            return torch.cat(outs, dim=0)

        # Default: gaussian kernel pooling along token index
        # centers on [0, k-1]; sigma proportional to (k / L)
        pos = torch.arange(k, device=Z.device, dtype=torch.float32)  # [k]
        centers = (torch.arange(L, device=Z.device, dtype=torch.float32) + 0.5) * (k / L) - 0.5  # [L]
        sigma = max(1e-6, float(sigma_scale) * (k / L))
        # weights: [L, k]
        w = torch.exp(- (pos.unsqueeze(0) - centers.unsqueeze(1)) ** 2 / (2.0 * sigma * sigma))
        w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-12))
        # pooled: [L, d] = [L,k] @ [k,d]
        pooled = w.to(Z.dtype) @ Z.to(Z.dtype)
        return pooled

    pooled_per_batch: List[torch.Tensor] = []
    for b in range(B):
        seg_b = seg[b] if b < len(seg) else []
        if len(seg_b) == 0:
            pooled_per_batch.append(torch.zeros(0, d, device=device, dtype=dtype))
            continue

        outs_b = []
        for s_idx in range(len(seg_b)):
            pos_container = seg_b[s_idx][0]
            idx = _to_index_tensor(pos_container)
            Z_img = inputs_embeds[b, idx, :] if idx.numel() > 0 else torch.empty(0, d, device=device, dtype=dtype)
            pooled = _pool_step(Z_img, latent_size)  # [latent_size, d]
            outs_b.append(pooled)

        pooled_b = torch.cat(outs_b, dim=0) if outs_b else torch.zeros(0, d, device=device, dtype=dtype)
        pooled_per_batch.append(pooled_b)

    return pooled_per_batch




def pooled_alignment_loss(
    pooled_images: List[torch.Tensor],
    ce_patch_vec: List[torch.Tensor],
    *,
    latent_size: int,
    loss_type: Literal["mse", "cosine", "smoothl1"] = "cosine",
    normalize: bool = False,
    stop_grad_image: bool = True,
    reduction: Literal["mean", "sum"] = "mean",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """
    Compute alignment loss between pooled image embeddings and latent embeddings for each sample.
    Both `pooled_images[b]` and `ce_patch_vec[b]` are expected to be 2D tensors [latent_size * t, d].

    Args:
        pooled_images: list over batch; each tensor [latent_size * t, d], from `collect_pooled_image_embeddings`.
        ce_patch_vec: list over batch; each tensor [latent_size * t, d], latent embeddings (flattened per step).
        latent_size: number of latent tokens per step (to validate shapes).
        loss_type: "mse" | "cosine" | "smoothl1".
        normalize: L2-normalize vectors before loss (useful for MSE as angular proxy).
        stop_grad_image: if True, detach pooled_images to only update latent side.
        reduction: "mean" or "sum".
        huber_delta: delta for SmoothL1 (Huber) loss.

    Returns:
        Scalar tensor loss.
    """
    assert isinstance(pooled_images, (list, tuple)) and isinstance(ce_patch_vec, (list, tuple))
    assert len(pooled_images) == len(ce_patch_vec), "Batch size mismatch between pooled_images and ce_patch_vec."
    assert isinstance(latent_size, int) and latent_size > 0

    loss_terms = []
    device = None
    dtype = None

    for b in range(len(pooled_images)):
        Zimg = pooled_images[b]
        Zlat = ce_patch_vec[b]
        assert torch.is_tensor(Zimg) and torch.is_tensor(Zlat), "Both tensors must be torch.Tensor."
        assert Zimg.dim() == 2 and Zlat.dim() == 2, "Expected 2D [latent_size*t, d] per sample."
        assert Zimg.shape == Zlat.shape, f"Shape mismatch at batch {b}: pooled {tuple(Zimg.shape)} vs latent {tuple(Zlat.shape)}"

        if device is None:
            device, dtype = Zlat.device, Zlat.dtype

        if stop_grad_image:
            Zimg = Zimg.detach()

        if normalize:
            Zimg = F.normalize(Zimg, dim=-1)
            Zlat = F.normalize(Zlat, dim=-1)

        if loss_type == "mse":
            term = F.mse_loss(Zimg, Zlat, reduction="mean")
        elif loss_type == "cosine":
            # cosine distance = 1 - cosine similarity
            sim = F.cosine_similarity(Zimg, Zlat, dim=-1)  # [N]
            term = (1.0 - sim).mean()
        else:  # "smoothl1"
            term = F.smooth_l1_loss(Zimg, Zlat, reduction="mean", beta=huber_delta)

        loss_terms.append(term)

    if not loss_terms:
        if device is None:
            device, dtype = torch.device("cpu"), torch.float32
        return torch.zeros((), device=device, dtype=dtype)

    loss_stack = torch.stack(loss_terms)
    return loss_stack.sum() if reduction == "sum" else loss_stack.mean()
