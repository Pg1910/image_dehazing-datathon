import numpy as np

class DummyDehaze:
    name = "dummy"

    def load(self, device: str = "cpu"):
        return self

    def predict_tile(self, tile_rgb_u8: np.ndarray) -> np.ndarray:
        # simple contrast stretch as placeholder
        x = tile_rgb_u8.astype(np.float32)
        x = (x - x.min()) / max(1e-6, (x.max() - x.min()))
        return (x * 255.0).clip(0, 255).astype(np.uint8)