import os
import re
import torch
import numpy as np
from PIL import Image


class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False


def log(message: str, message_type: str = 'info'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# ComfyUI_EsteeLauder -> {message}")


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(t_image: torch.Tensor) -> Image:
    if t_image.dtype != torch.float32:
        t_image = t_image.float()
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def image2mask(image: Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])


def file_is_extension(filename: str, ext_list: tuple) -> bool:
    true_ext = os.path.splitext(filename)[1]
    return true_ext.lower() in ext_list


def collect_files(root_dir: str, suffixes: tuple, default_dir: str = ""):
    result = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file_is_extension(file, suffixes):
                full_path = os.path.join(dirpath, file)
                if dirpath == default_dir:
                    relative_path = os.path.relpath(full_path, root_dir)
                    result.update({relative_path: full_path})
                else:
                    result.update({full_path: full_path})
    return result


def get_font_dict() -> dict:
    """Load fonts from our pack's fonts/ directory."""
    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_root = os.path.join(pack_root, 'fonts')
    font_dict = {}
    for brand_dir in ['Clinique', 'TheOrdinary']:
        d = os.path.join(fonts_root, brand_dir)
        if os.path.isdir(d):
            font_dict.update(collect_files(root_dir=d, suffixes=('.ttf', '.otf')))
    return font_dict
