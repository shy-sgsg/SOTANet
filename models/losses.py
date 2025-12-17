import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import uuid


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
    """Gradient (edge) loss using Sobel operators with optional downsampling
    and local-density suppression to focus on large-scale structures.

    Features added:
    - `downsample`: compute gradients on a downsampled version of the images
      (low-pass effect to suppress high-frequency small-scale edges).
    - `mag_thresh`: a per-pixel gradient magnitude threshold to form a binary
      edge mask.
    - `density_window` and `density_thresh`: compute local density (fraction of
      edge pixels in a window). If density is too high (e.g. dense village area)
      that region is suppressed from contributing to the gradient loss.
    """

    def __init__(
        self,
        loss_type: str = "Charbonnier",
        eps: float = 1e-3,
        delta: float = 1.0,
        downsample: int = 1,
        density_window: int = 0,
        density_thresh: float = 0.5,
        mag_thresh: float = 0.05,
    ):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
        self.eps = float(eps)
        self.delta = float(delta)
        self.downsample = int(downsample)
        self.density_window = int(density_window)
        self.density_thresh = float(density_thresh)
        self.mag_thresh = float(mag_thresh)

        # base sobel kernels (1 x 1 x 3 x 3)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        # register as buffers for device/dtype tracking (we will expand to match channels in forward)
        self.register_buffer("_sobel_x_base", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("_sobel_y_base", sobel_y.unsqueeze(0).unsqueeze(0))

    def _elementwise_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Compute elementwise loss map according to the configured loss_type."""
        n = self.loss_type.lower()
        if n == "l1":
            return torch.abs(diff)
        elif n == "charbonnier":
            return torch.sqrt(diff * diff + (self.eps ** 2))
        elif n == "huber":
            abs_err = torch.abs(diff)
            mask = abs_err <= self.delta
            loss_map = torch.where(mask, 0.5 * (abs_err ** 2), self.delta * abs_err - 0.5 * (self.delta ** 2))
            return loss_map
        else:
            raise ValueError(f"Unknown grad loss type '{self.loss_type}'")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss between input and target.

        Both `input` and `target` are expected to be (B, C, H, W).
        """
        if input.dim() != 4 or target.dim() != 4:
            raise ValueError("GradientLoss expects 4D tensors (B, C, H, W)")

        # optionally downsample (low-pass) to focus on large-scale structures
        if self.downsample is not None and int(self.downsample) > 1:
            scale = 1.0 / float(self.downsample)
            # use bilinear interpolation for smooth downsampling
            input_ds = F.interpolate(input, scale_factor=scale, mode="bilinear", align_corners=False)
            target_ds = F.interpolate(target, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            input_ds = input
            target_ds = target

        B, C, H, W = input_ds.shape
        device = input_ds.device
        dtype = input_ds.dtype

        # expand sobel kernels to (C, 1, 3, 3) for depthwise conv (groups=C)
        sobel_x = self._sobel_x_base.to(device=device, dtype=dtype).repeat(C, 1, 1, 1)
        sobel_y = self._sobel_y_base.to(device=device, dtype=dtype).repeat(C, 1, 1, 1)

        # compute gradients per channel using grouped conv
        G_in_x = F.conv2d(input_ds, sobel_x, padding=1, groups=C)
        G_in_y = F.conv2d(input_ds, sobel_y, padding=1, groups=C)
        G_tgt_x = F.conv2d(target_ds, sobel_x, padding=1, groups=C)
        G_tgt_y = F.conv2d(target_ds, sobel_y, padding=1, groups=C)

        # compute magnitude map (per-pixel) used for density computation
        grad_mag = torch.sqrt(G_in_x * G_in_x + G_in_y * G_in_y)

        # binary edge mask per-pixel using magnitude threshold
        if self.mag_thresh is not None and self.mag_thresh > 0.0:
            edge_mask = (grad_mag > self.mag_thresh).float()
        else:
            edge_mask = torch.ones_like(grad_mag)

        # if density suppression is enabled, compute local density and build suppression mask
        if self.density_window is not None and int(self.density_window) > 0:
            k = int(self.density_window)
            pad = k // 2
            # compute local density as fraction of edge pixels in window
            # reshape to (B*C,1,H,W) so avg_pool2d returns fraction
            edges_bc = edge_mask.view(B * C, 1, H, W)
            density = F.avg_pool2d(edges_bc, kernel_size=k, stride=1, padding=pad)
            density = density.view(B, C, H, W)
            # suppression: if density > density_thresh (too dense), we suppress (0), else keep (1)
            suppression_mask = (density <= self.density_thresh).float()
        else:
            suppression_mask = torch.ones_like(grad_mag)

        # compute elementwise loss maps for x and y gradients
        diff_x = G_in_x - G_tgt_x
        diff_y = G_in_y - G_tgt_y

        loss_map_x = self._elementwise_loss(diff_x)
        loss_map_y = self._elementwise_loss(diff_y)

        # apply suppression mask (and only average over kept pixels to avoid scale changes)
        mask_sum = suppression_mask.sum()
        if mask_sum.item() > 0:
            loss_x = (loss_map_x * suppression_mask).sum() / (mask_sum + 1e-12)
            loss_y = (loss_map_y * suppression_mask).sum() / (mask_sum + 1e-12)
        else:
            # fallback: if everything suppressed, just use plain mean
            loss_x = loss_map_x.mean()
            loss_y = loss_map_y.mean()

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


class CannyEdgeLoss(nn.Module):
    """Edge loss using Median blur + Canny on the real image and a differentiable
    soft-edge map for the generated image.

    - Real edges: computed on CPU using OpenCV's `medianBlur` and `Canny`.
      Optionally apply area-opening (remove small connected components) and
      compute a distance transform (DT) so the loss penalizes distance to
      true edges (Chamfer-like).
    - Generated edges: computed as a soft edge map via differentiable Sobel
      gradient magnitude followed by a smooth threshold (sigmoid). This allows
      gradients to flow back to the generator.

    Supports multi-scale processing: for each scale (e.g. 4,2,1) compute a DT
    on the downsampled real image and compare with corresponding downsampled
    soft edges from the generated image. Scales are weighted.
    """

    def __init__(
        self,
        scales=(4, 2, 1),
        weights=None,
        median_ksize: int = 3,
        canny_thresholds=((50, 150),),
        min_area: int = 30,
        soft_thresh: float = 0.02,
        soft_alpha: float = 5.0,
        device="cpu",
    ):
        super(CannyEdgeLoss, self).__init__()
        try:
            import cv2
            import numpy as np
        except Exception as e:
            raise ImportError("CannyEdgeLoss requires OpenCV (cv2) and numpy installed on the host") from e

        self.cv2 = cv2
        self.np = np
        self.scales = tuple(int(s) for s in scales)
        if weights is None:
            # default coarse-to-fine weights
            total = sum([0.6, 0.3, 0.1])
            self.weights = [0.6 / total, 0.3 / total, 0.1 / total][: len(self.scales)]
        else:
            self.weights = [float(w) for w in weights]
        self.median_ksize = int(median_ksize)
        # list/tuple of (threshold1, threshold2) pairs for multi-threshold Canny
        self.canny_thresholds = [tuple(t) for t in canny_thresholds]
        self.min_area = int(min_area)
        self.soft_thresh = float(soft_thresh)
        self.soft_alpha = float(soft_alpha)
        self.device = device
        # visualization
        self.save_vis = False
        self.vis_dir = None

    def enable_visualization(self, vis_dir: str):
        try:
            os.makedirs(vis_dir, exist_ok=True)
            self.save_vis = True
            self.vis_dir = vis_dir
        except Exception:
            self.save_vis = False
            self.vis_dir = None

    def _compute_real_dt_per_sample(self, img_np):
        """Given one RGB image as numpy float32 in [0,1], compute list of
        tuples (edges_uint8, dt_float32) for each configured scale.
        edges_uint8: HxW uint8 binary edge image (0/255)
        dt_float32: HxW float32 distance transform normalized by max dim
        """
        cv2 = self.cv2
        np = self.np
        results = []
        # convert to gray uint8
        gray = (img_np * 255.0).astype(np.uint8)
        if gray.ndim == 3 and gray.shape[2] == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        for s in self.scales:
            if s > 1:
                h = max(1, gray.shape[0] // s)
                w = max(1, gray.shape[1] // s)
                gray_s = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            else:
                gray_s = gray

            # median blur
            if self.median_ksize and self.median_ksize > 1:
                k = self.median_ksize
                # ensure odd kernel
                if k % 2 == 0:
                    k += 1
                gray_blur = cv2.medianBlur(gray_s, k)
            else:
                gray_blur = gray_s

            # combine multiple Canny thresholds (logical OR)
            edges = np.zeros_like(gray_blur, dtype=np.uint8)
            for (t1, t2) in self.canny_thresholds:
                e = cv2.Canny(gray_blur, int(t1), int(t2))
                edges = cv2.bitwise_or(edges, e)

            # remove small connected components (area opening)
            if self.min_area and self.min_area > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
                # background label=0
                new_edges = np.zeros_like(edges)
                for lab in range(1, num_labels):
                    area = stats[lab, cv2.CC_STAT_AREA]
                    if area >= self.min_area:
                        new_edges[labels == lab] = 255
                edges = new_edges

            # compute distance transform: distance to nearest edge (in pixels)
            # distanceTransform computes distance to zero pixels, so invert edges
            inv = cv2.bitwise_not(edges)
            # convert to binary (0 background, 255 foreground), then to uint8
            inv_bin = (inv > 0).astype(np.uint8) * 255
            # compute distance transform (float32)
            dt = cv2.distanceTransform(inv_bin, distanceType=cv2.DIST_L2, maskSize=5)
            # normalize dt by max dim to keep scale-invariant
            max_dim = max(dt.shape)
            if max_dim > 0:
                dt = dt.astype(np.float32) / float(max_dim)
            else:
                dt = dt.astype(np.float32)

            # edges currently 0/255 uint8
            edges_uint8 = edges.astype(np.uint8)
            results.append((edges_uint8, dt))

        return results

    def forward(self, gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Compute Canny-based chamfer-like loss between `gen` and `real`.

        gen, real: (B, C, H, W) in [0,1] float on device (GPU/CPU). Real DTs
        are computed on CPU with OpenCV; generated soft edges are computed on
        `self.device` via differentiable ops so gradients flow to generator.
        """
        cv2 = self.cv2
        np = self.np

        if gen.dim() != 4 or real.dim() != 4:
            raise ValueError("CannyEdgeLoss expects 4D tensors (B,C,H,W)")

        B, C, H, W = gen.shape

        # compute soft edge map from generated image using Sobel magnitude
        # first, convert gen to grayscale (weighted RGB) on device
        if C == 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=gen.device, dtype=gen.dtype).view(1, 3, 1, 1)
            gen_gray = (gen * weights).sum(dim=1, keepdim=True)
            real_gray = (real * weights).sum(dim=1, keepdim=True)
        else:
            gen_gray = gen.mean(dim=1, keepdim=True)
            real_gray = real.mean(dim=1, keepdim=True)

        # compute Sobel gradients (differentiable)
        sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=gen.device, dtype=gen.dtype)
        sobel_y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=gen.device, dtype=gen.dtype)
        Cg = gen_gray.shape[1]
        # use groups=1 because gen_gray has single channel
        Gx = F.conv2d(F.pad(gen_gray, (1, 1, 1, 1), mode="reflect"), sobel_x.to(gen.device), padding=0)
        Gy = F.conv2d(F.pad(gen_gray, (1, 1, 1, 1), mode="reflect"), sobel_y.to(gen.device), padding=0)
        grad_mag = torch.sqrt(Gx * Gx + Gy * Gy + 1e-12)

        # soft edge via sigmoid around soft_thresh
        soft = torch.sigmoid((grad_mag - self.soft_thresh) * self.soft_alpha)
        # soft in [0,1], shape (B,1,H,W)

        # prepare vis directory if needed
        do_vis = getattr(self, "save_vis", False)
        vis_dir = getattr(self, "vis_dir", None)
        if do_vis and vis_dir is not None:
            try:
                os.makedirs(vis_dir, exist_ok=True)
            except Exception:
                do_vis = False

        total_loss = torch.tensor(0.0, device=gen.device, dtype=gen.dtype)
        # process per-sample on CPU for real DTs
        for b in range(B):
            real_b = real[b].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
            dts_and_edges = self._compute_real_dt_per_sample(real_b)

            for i, (edges_uint8, dt) in enumerate(dts_and_edges):
                w = float(self.weights[i]) if i < len(self.weights) else 1.0
                # dt is (H_s, W_s) float32 normalized
                dt_t = torch.from_numpy(dt).to(gen.device).unsqueeze(0).unsqueeze(0)  # 1x1xH_sxW_s
                # resize soft[b] to dt size
                soft_b = soft[b : b + 1]
                soft_resized = F.interpolate(soft_b, size=(dt.shape[0], dt.shape[1]), mode="bilinear", align_corners=False)
                # multiply and sum (mean over pixels)
                loss_map = soft_resized * dt_t
                total_loss = total_loss + w * loss_map.mean()

                # optionally save visualizations
                if do_vis and vis_dir is not None:
                    try:
                        # prepare file base name
                        tstamp = int(time.time())
                        uid = uuid.uuid4().hex[:6]
                        base = f"vis_b{b}_s{i}_{tstamp}_{uid}"
                        # save real edges (edges_uint8), filtered edges, dt (normalized), soft edge map, and generated grad mag
                        # edges_uint8 is 0/255 single channel
                        fn_edges = os.path.join(vis_dir, base + "_real_edges.png")
                        cv2.imwrite(fn_edges, edges_uint8)

                        # save dt as heatmap-like grayscale (normalize 0..1 to 0..255)
                        dt_vis = (dt - dt.min())
                        if dt_vis.max() > 0:
                            dt_vis = dt_vis / (dt_vis.max())
                        dt_vis_u8 = (dt_vis * 255.0).astype(self.np.uint8)
                        fn_dt = os.path.join(vis_dir, base + "_real_dt.png")
                        cv2.imwrite(fn_dt, dt_vis_u8)

                        # save soft edge map (resized) as u8
                        soft_np = (soft_resized.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0).astype(self.np.uint8)
                        fn_soft = os.path.join(vis_dir, base + "_gen_soft.png")
                        cv2.imwrite(fn_soft, soft_np)

                        # save generated gradient magnitude (before sigmoid) resized to dt size
                        # compute grad mag for this sample
                        Gx_b = F.conv2d(F.pad(gen_gray[b : b + 1], (1, 1, 1, 1), mode="reflect"), sobel_x.to(gen.device), padding=0)
                        Gy_b = F.conv2d(F.pad(gen_gray[b : b + 1], (1, 1, 1, 1), mode="reflect"), sobel_y.to(gen.device), padding=0)
                        gradmag_b = torch.sqrt(Gx_b * Gx_b + Gy_b * Gy_b + 1e-12)
                        gradmag_res = F.interpolate(gradmag_b, size=(dt.shape[0], dt.shape[1]), mode="bilinear", align_corners=False)
                        gradmag_np = (gradmag_res.squeeze(0).squeeze(0).detach().cpu().numpy())
                        # normalize for visualization
                        gm = gradmag_np - gradmag_np.min()
                        if gm.max() > 0:
                            gm = gm / gm.max()
                        gm_u8 = (gm * 255.0).astype(self.np.uint8)
                        fn_gm = os.path.join(vis_dir, base + "_gen_gradmag.png")
                        cv2.imwrite(fn_gm, gm_u8)

                        # save generated image (RGB) resized to dt size
                        gen_rgb = gen[b].detach().cpu().numpy().transpose(1, 2, 0)
                        gen_rgb_rs = cv2.resize((gen_rgb * 255.0).astype(self.np.uint8), (dt.shape[1], dt.shape[0]), interpolation=cv2.INTER_AREA)
                        fn_gen = os.path.join(vis_dir, base + "_gen_image.png")
                        cv2.imwrite(fn_gen, gen_rgb_rs)
                    except Exception:
                        # best-effort; do not raise
                        pass

        # average over batch
        total_loss = total_loss / float(B)
        return total_loss
