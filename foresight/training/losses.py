"""Active-patch event alignment in the frozen DINO space."""

from torch.nn import functional as F


def event_alignment_loss(events, targets, activity, *, beta=1.0, cosine_weight=0.2):
    if events.shape != targets.shape or activity.shape != targets.shape[:-1]:
        raise ValueError("Event alignment requires matching [B,T,N,D] features and [B,T,N] activity")
    targets = targets.detach().float()
    error = F.smooth_l1_loss(events.float(), targets, beta=beta, reduction="none").mean(-1)
    error = error + cosine_weight * (1 - F.cosine_similarity(events.float(), targets, dim=-1))
    mask = activity.to(error.dtype)
    # Each sample/time contributes equally; an empty active set contributes zero.
    return ((error * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
