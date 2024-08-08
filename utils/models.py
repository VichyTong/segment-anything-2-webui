import torch
from sam2.build_sam import build_sam2_video_predictor


CHECKPOINT_NAMES = ["tiny", "small", "base_plus", "large"]
CHECKPOINTS = {
    "tiny": ["sam2_hiera_t.yaml", "checkpoints/sam2_hiera_tiny.pt"],
    "small": ["sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt"],
    "base_plus": ["sam2_hiera_b+.yaml", "checkpoints/sam2_hiera_base_plus.pt"],
    "large": ["sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt"],
}


def load_models():
    vedio_predictors = {}
    for key, (config, checkpoint) in CHECKPOINTS.items():
        vedio_predictors[key] = build_sam2_video_predictor(config, checkpoint)
    return vedio_predictors
