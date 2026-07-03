import torch
from typing import Tuple, Dict, Any


class Float32ColorCorrect:
    """Professional color correction in 32-bit float space."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "lift_r": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "lift_g": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "lift_b": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "correct"
    CATEGORY = "FXTD Studios/Radiance/Color"

    def correct(self, image: torch.Tensor, exposure: float = 0.0,
                contrast: float = 1.0, brightness: float = 0.0,
                saturation: float = 1.0, lift_r: float = 0.0,
                lift_g: float = 0.0, lift_b: float = 0.0,
                gain_r: float = 1.0, gain_g: float = 1.0,
                gain_b: float = 1.0) -> Tuple[torch.Tensor]:

        img = image.clone().float()

        if exposure != 0.0:
            img = img * (2.0 ** exposure)

        if img.shape[-1] >= 3:
            img[..., 0] = img[..., 0] * gain_r + lift_r
            img[..., 1] = img[..., 1] * gain_g + lift_g
            img[..., 2] = img[..., 2] * gain_b + lift_b

        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5

        if brightness != 0.0:
            img = img + brightness

        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)

        return (img,)


NODE_CLASS_MAPPINGS = {
    "Float32ColorCorrect": Float32ColorCorrect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Float32ColorCorrect": "Float32 Color Correct"
}
