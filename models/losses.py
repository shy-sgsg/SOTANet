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


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (returns 1 - mean_ssim).

    Implementation adapted to be lightweight and dependency-free. Computes
    SSIM per-channel and averages across channels and batch.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = float(data_range)
        # create gaussian window
        self.register_buffer("_window", self.create_window(window_size, sigma))

    def create_window(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = g[:, None] @ g[None, :]
        return w.unsqueeze(0).unsqueeze(0)  # 1x1xWxH

    def _pad(self, x):
        pad = self.window_size // 2
        return F.pad(x, (pad, pad, pad, pad), mode="reflect")

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # expect tensors in [0,1] float
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError("SSIMLoss expects 4D tensors (B,C,H,W)")

        B, C, H, W = img1.shape
        window = self._window.to(device=img1.device, dtype=img1.dtype)
        K1 = 0.01
        K2 = 0.03
        L = self.data_range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        # compute per-channel SSIM
        mu1 = F.conv2d(self._pad(img1.view(B * C, 1, H, W)), window, padding=0, groups=1)
        mu2 = F.conv2d(self._pad(img2.view(B * C, 1, H, W)), window, padding=0, groups=1)
        mu1 = mu1.view(B, C, H, W)
        mu2 = mu2.view(B, C, H, W)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(self._pad((img1 * img1).view(B * C, 1, H, W)), window, padding=0).view(B, C, H, W) - mu1_sq
        sigma2_sq = F.conv2d(self._pad((img2 * img2).view(B * C, 1, H, W)), window, padding=0).view(B, C, H, W) - mu2_sq
        sigma12 = F.conv2d(self._pad((img1 * img2).view(B * C, 1, H, W)), window, padding=0).view(B, C, H, W) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        # clamp to [0,1]
        ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        # mean over spatial and channels and batch
        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val


class VGGFeatureLoss(nn.Module):
    """Perceptual loss using VGG features (shallow layers).

    Uses torchvision's VGG16 pretrained network and extracts features at
    specified layer names (e.g., 'relu1_2', 'relu2_2'). Computes L1 loss
    between feature maps.
    """

    def __init__(self, layers=("relu1_2", "relu2_2"), device="cpu"):
        super(VGGFeatureLoss, self).__init__()
        try:
            import torchvision.models as models
        except Exception as e:
            raise ImportError("torchvision is required for VGGFeatureLoss") from e

        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.layers = layers
        # mapping layer names to indices in vgg.features
        self.layer_name_mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu4_1": 18,
            "relu4_2": 20,
            "relu4_3": 22,
            "relu5_1": 25,
        }

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # expect input and target in [0,1], 3-channel RGB
        if input.dim() != 4 or target.dim() != 4:
            raise ValueError("VGGFeatureLoss expects 4D tensors (B,C,H,W)")
        # normalize to VGG expected mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=input.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input.device).view(1, 3, 1, 1)
        inp = (input - mean) / std
        tgt = (target - mean) / std

        feats_in = []
        feats_tgt = []
        x_in = inp
        x_tgt = tgt
        loss = 0.0
        max_idx = max(self.layer_name_mapping[l] for l in self.layers)
        for i, layer in enumerate(self.vgg):
            x_in = layer(x_in)
            x_tgt = layer(x_tgt)
            if i in [self.layer_name_mapping[l] for l in self.layers]:
                loss = loss + F.l1_loss(x_in, x_tgt)
            if i >= max_idx:
                break

        return loss
