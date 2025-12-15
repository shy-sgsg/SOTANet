import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a smooth L1 variant).

    L_char(x, y) = mean( sqrt((x-y)^2 + eps^2) )
    """

    def __init__(self, eps: float = 1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = float(eps)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        loss = torch.sqrt(diff * diff + (self.eps ** 2))
        return loss.mean()


class HuberLoss(nn.Module):
    """Huber loss (a.k.a. smooth L1) implementation.

    L_delta(a) = 0.5 * a^2            if |a| <= delta
               = delta * |a| - 0.5*delta^2  otherwise
    """

    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.delta = float(delta)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_err = torch.abs(input - target)
        mask = abs_err <= self.delta
        loss = torch.where(mask, 0.5 * (abs_err ** 2), self.delta * abs_err - 0.5 * (self.delta ** 2))
        return loss.mean()


def get_rec_loss(name: str = "L1", eps: float = 1e-3, delta: float = 1.0):
    """Factory: return reconstruction loss module by name.

    name: one of 'L1', 'Charbonnier', 'Huber' (case-insensitive)
    eps: Charbonnier epsilon
    delta: Huber delta
    """
    n = name.lower()
    if n == "l1":
        return nn.L1Loss()
    elif n == "charbonnier":
        return CharbonnierLoss(eps=eps)
    elif n == "huber":
        return HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown reconstruction loss '{name}'. Choose one of L1|Charbonnier|Huber.")
