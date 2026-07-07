"""
Impact Pack nodes (minimal subset)
-----------------------------------
Sourced from ltdrdata/ComfyUI-Impact-Pack. Ported to ComfyUI_EsteeLauder_prod.
Nodes: ImpactCompare, ImpactSwitch, ImpactStringSelector,
       ImageListToImageBatch, ImpactImageBatchToImageList, MaskListToMaskBatch
"""

import sys
import logging
import inspect

import torch
import comfy.utils


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


def make_3d_mask(mask):
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    return mask


class ImpactCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmp": (['a = b', 'a <> b', 'a > b', 'a < b', 'a >= b', 'a <= b', 'tt', 'ff'],),
                "a": (any_typ,),
                "b": (any_typ,),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"
    RETURN_TYPES = ("BOOLEAN",)

    def doit(self, cmp, a, b):
        if cmp == "a = b":
            return (a == b,)
        elif cmp == "a <> b":
            return (a != b,)
        elif cmp == "a > b":
            return (a > b,)
        elif cmp == "a < b":
            return (a < b,)
        elif cmp == "a >= b":
            return (a >= b,)
        elif cmp == "a <= b":
            return (a <= b,)
        elif cmp == 'tt':
            return (True,)
        else:
            return (False,)


class GeneralSwitch:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
                "sel_mode": ("BOOLEAN", {"default": False, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False}),
            },
            "optional": {
                "input1": (any_typ, {"lazy": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = (any_typ, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Util"

    def check_lazy_status(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"
        if input_name in kwargs:
            return [input_name]
        return []

    @staticmethod
    def doit(*args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"
        selected_label = input_name
        node_id = kwargs['unique_id']

        if 'extra_pnginfo' in kwargs and kwargs['extra_pnginfo'] is not None:
            nodelist = kwargs['extra_pnginfo']['workflow']['nodes']
            for node in nodelist:
                if str(node['id']) == node_id:
                    for slot in node['inputs']:
                        if slot['name'] == input_name and 'label' in slot:
                            selected_label = slot['label']
                    break
        else:
            logging.info("[EsteeLauder_prod] ImpactSwitch: API mode, label lookup skipped.")

        if input_name in kwargs:
            return kwargs[input_name], selected_label, selected_index
        else:
            logging.info("ImpactSwitch: invalid select index (ignored)")
            return None, "", selected_index


class StringSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "strings": ("STRING", {"multiline": True}),
            "multiline": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "select": ("INT", {"min": 0, "max": sys.maxsize, "step": 1, "default": 0}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Util"

    def doit(self, strings, multiline, select):
        lines = strings.split('\n')

        if multiline:
            result = []
            current_string = ""
            for line in lines:
                if line.startswith("#"):
                    if current_string:
                        result.append(current_string.strip())
                        current_string = ""
                current_string += line + "\n"
            if current_string:
                result.append(current_string.strip())
            if len(result) == 0:
                selected = strings
            else:
                selected = result[select % len(result)]
            if selected.startswith('#'):
                selected = selected[1:]
        else:
            if len(lines) == 0:
                selected = strings
            else:
                selected = lines[select % len(lines)]

        return (selected,)


class ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Operation"

    def doit(self, images):
        if len(images) == 0:
            return ()
        if len(images) == 1:
            img = images[0]
            if img.ndim == 3:
                img = img.unsqueeze(0)
            return (img,)

        first = images[0]
        if first.ndim == 3:
            first = first.unsqueeze(0)
        H, W, C = first.shape[1], first.shape[2], first.shape[3]
        dev = first.device

        # Pre-allocate; count total frames
        parts = [first]
        for img in images[1:]:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(dev)
            if img.shape[1] != H or img.shape[2] != W:
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), W, H, "lanczos", "center"
                ).movedim(1, -1)
            if img.shape[3] != C:
                C = min(C, img.shape[3])
            parts.append(img)

        result = torch.cat([p[:, :, :, :C] for p in parts], dim=0)
        return (result,)


class ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Util"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images,)


class MaskListToMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask):
        if len(mask) == 0:
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu").unsqueeze(0)
            return (empty_mask,)

        masks_3d = [make_3d_mask(m) for m in mask]
        target_shape = masks_3d[0].shape[1:]
        upscaled_masks = []
        for m in masks_3d:
            if m.shape[1:] != target_shape:
                m = m.unsqueeze(1).repeat(1, 3, 1, 1)
                m = comfy.utils.common_upscale(m, target_shape[1], target_shape[0], "lanczos", "center")
                m = m[:, 0, :, :]
            upscaled_masks.append(m)
        result = torch.cat(upscaled_masks, dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ImpactCompare": ImpactCompare,
    "ImpactSwitch": GeneralSwitch,
    "ImpactStringSelector": StringSelector,
    "ImageListToImageBatch": ImageListToImageBatch,
    "ImpactImageBatchToImageList": ImageBatchToImageList,
    "MaskListToMaskBatch": MaskListToMaskBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImpactCompare": "Impact: Compare",
    "ImpactSwitch": "Impact: Switch",
    "ImpactStringSelector": "Impact: String Selector",
    "ImageListToImageBatch": "Impact: Image List → Batch",
    "ImpactImageBatchToImageList": "Impact: Image Batch → List",
    "MaskListToMaskBatch": "Impact: Mask List → Batch",
}
