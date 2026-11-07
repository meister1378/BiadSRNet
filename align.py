"""Classical pre-alignment for a real LR burst, used at inference time."""
import cv2
import numpy as np


def align_to_center(lr_frames, center_idx):
    """Farneback-flow-warp every frame in lr_frames onto the center frame."""
    center = lr_frames[center_idx]
    h, w = center.shape
    aligned = []
    for i, f in enumerate(lr_frames):
        if i == center_idx:
            aligned.append(center)
            continue
        flow = cv2.calcOpticalFlowFarneback(f.astype(np.uint8), center.astype(np.uint8), None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped = cv2.remap(f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(warped)
    return aligned
