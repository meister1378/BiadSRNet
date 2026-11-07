# BIADSRNet

BIADSRNet (Burst Intermediate Alignment Diffusion Super-Resolution Net --
BIASRNet + Diffusion): a small conditional diffusion refinement
stage bolted onto a frozen BIASRNet backbone (see the separate BIASRNet
repo), for perceptual texture sharpening when the deployment target is human viewing
rather than a pixel-fidelity metric (PSNR). Every architecture change to the
backbone's own alignment/fusion plateaued around the same PSNR; diffusion
refinement is a different axis -- it hallucinates plausible fine texture a
deterministic network tends to over-smooth.

**Caveat:** the added texture is not guaranteed to be physically real --
re-check this before any safety-relevant deployment on thermal imagery.

Method: rather than running the full T-step reverse diffusion process from
pure noise, seed the process at an intermediate noise level using the
backbone's output as a proxy for x0, then run only the remaining
`t_start -> 0` reverse steps. See `model.py` for the implementation.

This repo vendors a copy of the BIASRNet model definition
(`biasrnet_model.py`) so it can run standalone -- see that file's
docstring.

## Files

- `biasrnet_model.py` -- vendored copy of the BIASRNet backbone (frozen at
  inference time here)
- `model.py` -- diffusion UNet, DDPM (`GaussianDiffusion`), and the
  `BIADSRNet` wrapper that combines the frozen backbone + diffusion stage
- `align.py` -- classical Farneback pre-alignment for a real LR burst
- `infer.py` -- runs a trained BIADSRNet on a real LR burst (no synthetic
  degradation -- this is the production inference path)

This repo ships inference code only. You'll need trained checkpoints
(`.pth`, backbone + diffusion stage) to run it.

## Inference

```bash
python infer.py --frames-dir /path/to/real_lr_frames \
    --backbone-ckpt /path/to/biasrnet_bestPSNR.pth \
    --diffusion-ckpt ./checkpoints/diffrefine_final.pth --out-dir ./out
```

`--frames-dir` should contain real, already-low-resolution consecutive
grayscale frames.

## Requirements

```
torch
torchvision
opencv-python
numpy
```

## Data

Trained and evaluated on two public, permissively-licensed real thermal
video sources:
- [IRVAL](https://huggingface.co/datasets/zytlhy/IRVAL) (CC-BY-4.0)
- [issai/OpenThermalAction](https://huggingface.co/datasets/issai/OpenThermalAction) (MIT)

No other dataset is used or redistributed by this code.
