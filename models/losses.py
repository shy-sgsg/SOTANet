import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GradientLoss(nn.Module):
    """Gradient (edge) loss using Sobel operators.

    Computes horizontal and vertical gradients for each channel using depthwise
    convolution with Sobel kernels, then applies a reconstruction loss (L1/Charbonnier/Huber)
    to the gradient maps and sums the two components.
    """

    def __init__(self, loss_type: str = "Charbonnier", eps: float = 1e-3, delta: float = 1.0):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.delta = delta
        # reuse factory to get underlying criterion (it will return an nn.Module)
        self.criterion = get_rec_loss(loss_type, eps=eps, delta=delta)

        # base sobel kernels (1 x 1 x 3 x 3)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        # register as buffers for device/dtype tracking (we will expand to match channels in forward)
        self.register_buffer("_sobel_x_base", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("_sobel_y_base", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss between input and target.

        Both `input` and `target` are expected to be (B, C, H, W).
        """
        if input.dim() != 4 or target.dim() != 4:
            raise ValueError("GradientLoss expects 4D tensors (B, C, H, W)")

        C = input.size(1)
        device = input.device
        dtype = input.dtype

        # expand sobel kernels to (C, 1, 3, 3) for depthwise conv (groups=C)
        sobel_x = self._sobel_x_base.to(device=device, dtype=dtype).repeat(C, 1, 1, 1)
        sobel_y = self._sobel_y_base.to(device=device, dtype=dtype).repeat(C, 1, 1, 1)

        # compute gradients per channel using grouped conv
        G_in_x = F.conv2d(input, sobel_x, padding=1, groups=C)
        G_in_y = F.conv2d(input, sobel_y, padding=1, groups=C)
        G_tgt_x = F.conv2d(target, sobel_x, padding=1, groups=C)
        G_tgt_y = F.conv2d(target, sobel_y, padding=1, groups=C)

        # compute loss on gradient maps
        loss_x = self.criterion(G_in_x, G_tgt_x)
        loss_y = self.criterion(G_in_y, G_tgt_y)

        return loss_x + loss_y
