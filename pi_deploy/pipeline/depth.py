import torch
import numpy as np


class DepthEstimator:
    def __init__(self):
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.hub.load(
            'intel-isl/MiDaS', 'MiDaS_small', trust_repo=True
        )
        self.model.eval()
        transforms = torch.hub.load(
            'intel-isl/MiDaS', 'transforms', trust_repo=True
        )
        self.transform = transforms.small_transform
        print("DepthEstimator (MiDaS v2.1 Small) loaded.")

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Takes an RGB numpy array (H, W, 3).
        Returns raw MiDaS inverse-depth map (H, W) — higher = closer.
        """
        input_batch = self.transform(image_rgb)
        with torch.no_grad():
            pred = self.model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        return pred.cpu().numpy()
