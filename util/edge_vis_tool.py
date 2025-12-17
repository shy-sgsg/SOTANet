#!/usr/bin/env python3
"""Utility to run CannyEdgeLoss on a single image pair and save visualizations.

Usage examples:
  python util/edge_vis_tool.py --gen path/to/gen.png --real path/to/real.png --out results/edge_vis

This script expects the repository's Python path to include the project root so
that `models.losses.CannyEdgeLoss` can be imported. It uses OpenCV + numpy + torch.
"""

import argparse
import os
import sys

# make sure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch

from models.losses import CannyEdgeLoss


def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # convert to RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def to_tensor(img):
    # img: HWC float32 [0,1]
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # 1xCxHxW
    return t


def parse_thresholds(s: str):
    # format: "50:150;30:90" or "50:150"
    pairs = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if ":" in seg:
            a, b = seg.split(":")
            pairs.append((int(a), int(b)))
        else:
            v = int(seg)
            pairs.append((v // 3, v))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, help="path to generated image (PNG/JPG)")
    parser.add_argument("--real", required=True, help="path to real image (PNG/JPG)")
    parser.add_argument("--out", required=True, help="output directory to save visualizations")
    parser.add_argument("--canny_scales", default="4,2,1")
    parser.add_argument("--canny_weights", default="0.6,0.3,0.1")
    parser.add_argument("--canny_thresholds", default="50:150")
    parser.add_argument("--canny_median_ksize", type=int, default=3)
    parser.add_argument("--canny_min_area", type=int, default=30)
    parser.add_argument("--lambda_Canny", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gen_img = imread_rgb(args.gen)
    real_img = imread_rgb(args.real)

    # resize gen to match real if needed
    if gen_img.shape[0:2] != real_img.shape[0:2]:
        gen_img = cv2.resize(gen_img, (real_img.shape[1], real_img.shape[0]), interpolation=cv2.INTER_AREA)

    gen_t = to_tensor(gen_img).to(args.device)
    real_t = to_tensor(real_img).to(args.device)

    scales = [int(s) for s in args.canny_scales.split(",") if s.strip()]
    weights = [float(w) for w in args.canny_weights.split(",") if w.strip()]
    thresholds = parse_thresholds(args.canny_thresholds)

    loss_fn = CannyEdgeLoss(scales=tuple(scales), weights=weights, median_ksize=args.canny_median_ksize, canny_thresholds=thresholds, min_area=args.canny_min_area, device=args.device)
    loss_fn.enable_visualization(args.out)

    loss = loss_fn(gen_t, real_t)
    print(f"CannyEdgeLoss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
