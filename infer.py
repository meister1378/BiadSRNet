"""Run a trained BIADSRNet (frozen BIASRNet backbone + diffusion refinement)
on a real low-resolution burst (no synthetic degradation -- production
inference path).

Takes a directory of >= n_frames consecutive grayscale LR frames, aligns the
neighbors onto the center frame with Farneback optical flow, runs BIASRNet
then the diffusion refinement stage, and writes the restored x4 image (plus
the plain-BIASRNet and bicubic outputs for reference).

Usage:
    python infer.py --frames-dir /path/to/lr_frames --center 7 \
        --backbone-ckpt /path/to/biasrnet_bestPSNR.pth \
        --diffusion-ckpt ./checkpoints/diffrefine_final.pth --out-dir ./out
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from align import align_to_center
from model import BIADSRNet


def load_frames(frames_dir):
    files = sorted(os.listdir(frames_dir))
    return [cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', required=True, help='directory of consecutive grayscale LR frames')
    p.add_argument('--backbone-ckpt', required=True)
    p.add_argument('--diffusion-ckpt', required=True)
    p.add_argument('--center', type=int, default=None,
                   help='index (into the sorted frame list) of the frame to restore; default = middle frame')
    p.add_argument('--n-frames', type=int, default=5)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--t-start', type=int, default=200)
    p.add_argument('--out-dir', default='./out')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    frames = load_frames(args.frames_dir)
    half = args.n_frames // 2
    center = args.center if args.center is not None else len(frames) // 2
    if center - half < 0 or center + half >= len(frames):
        raise ValueError(f'center={center} needs {half} frames of margin on each side '
                          f'(have {len(frames)} frames)')
    indices = list(range(center - half, center + half + 1))
    window = [np.ascontiguousarray(frames[i]) for i in indices]

    aligned = align_to_center(window, half)
    burst = np.stack(aligned, axis=0).astype(np.float32) / 255.0
    burst_t = torch.from_numpy(burst).float().unsqueeze(0).to(device)

    biadsr = BIADSRNet.from_checkpoints(args.backbone_ckpt, args.diffusion_ckpt, device,
                                         n_frames=args.n_frames, t_start=args.t_start)

    with torch.no_grad():
        backbone_out = biadsr.backbone(burst_t).clamp(0, 1)
        refined = biadsr(burst_t)
        center_t = burst_t[:, half:half + 1]
        bicubic = F.interpolate(center_t, scale_factor=args.scale, mode='bicubic', align_corners=False)

    def to_u8(t):
        return (t.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.out_dir, 'restored_biadsrnet.png'), to_u8(refined))
    cv2.imwrite(os.path.join(args.out_dir, 'restored_biasrnet.png'), to_u8(backbone_out))
    cv2.imwrite(os.path.join(args.out_dir, 'bicubic_baseline.png'), to_u8(bicubic))
    print(f'wrote restored_biadsrnet.png + restored_biasrnet.png + bicubic_baseline.png '
          f'to {args.out_dir} (center frame index {center})')


if __name__ == '__main__':
    main()
