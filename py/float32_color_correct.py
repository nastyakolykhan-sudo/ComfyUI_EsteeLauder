import torch
from typing import Tuple, Dict, Any

LUMA_SPACES = ["Rec.709 / sRGB", "Rec.2020", "DCI-P3", "ACES"]
LUMA_COEFFICIENTS = {
    "Rec.709 / sRGB": (0.2126, 0.7152, 0.0722),
    "Rec.2020":       (0.2627, 0.6780, 0.0593),
    "DCI-P3":         (0.2289, 0.6917, 0.0793),
    "ACES":           (0.2126, 0.7152, 0.0722),
}


class Float32ColorCorrect:
    """Professional color correction in 32-bit float space."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image":      ("IMAGE",),
                "exposure":   ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0,  "step": 0.1}),
                "contrast":   ("FLOAT", {"default": 1.0, "min":   0.0, "max":  4.0,  "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min":  -1.0, "max":  1.0,  "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min":   0.0, "max":  3.0,  "step": 0.05}),
                "gamma":      ("FLOAT", {"default": 2.0, "min":   0.1, "max": 10.0,  "step": 0.1}),
            },
            "optional": {
                "lift_r":       ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "lift_g":       ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "lift_b":       ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "gain_r":       ("FLOAT", {"default": 1.0, "min":  0.0, "max": 2.0, "step": 0.01}),
                "gain_g":       ("FLOAT", {"default": 1.0, "min":  0.0, "max": 2.0, "step": 0.01}),
                "gain_b":       ("FLOAT", {"default": 1.0, "min":  0.0, "max": 2.0, "step": 0.01}),
                "luma_space":   (LUMA_SPACES, {"default": "Rec.709 / sRGB"}),
                "clamp_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "correct"
    CATEGORY = "FXTD Studios/Radiance/Color"

    def correct(self, image: torch.Tensor, exposure: float = 0.0,
                contrast: float = 1.0, brightness: float = 0.0,
                saturation: float = 1.0, gamma: float = 2.0,
                lift_r: float = 0.0, lift_g: float = 0.0, lift_b: float = 0.0,
                gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
                luma_space: str = "Rec.709 / sRGB",
                clamp_output: bool = False) -> Tuple[torch.Tensor]:

        img = image.clone().float()

        if exposure != 0.0:
            img = img * (2.0 ** exposure)

        if img.shape[-1] >= 3:
            img[..., 0] = img[..., 0] * gain_r + lift_r
            img[..., 1] = img[..., 1] * gain_g + lift_g
            img[..., 2] = img[..., 2] * gain_b + lift_b

        if gamma != 1.0:
            img = img.clamp(min=0.0) ** (1.0 / gamma)

        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5

        if brightness != 0.0:
            img = img + brightness

        if saturation != 1.0 and img.shape[-1] >= 3:
            lr, lg, lb = LUMA_COEFFICIENTS.get(luma_space, (0.2126, 0.7152, 0.0722))
            luma = lr * img[..., 0] + lg * img[..., 1] + lb * img[..., 2]
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)

        if clamp_output:
            img = img.clamp(0.0, 1.0)

        return (img,)


NODE_CLASS_MAPPINGS = {
    "Float32ColorCorrect": Float32ColorCorrect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Float32ColorCorrect": "Float32 Color Correct"
}
