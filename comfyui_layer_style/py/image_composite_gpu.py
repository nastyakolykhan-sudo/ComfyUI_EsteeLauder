from .imagefunc import *
import torch.nn.functional as F

MAX_RESOLUTION = 8192


class LS_ImageCompositeGPU:

    def __init__(self):
        self.NODE_NAME = 'ImageCompositeGPU'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image":      ("IMAGE",),
                "layer_image":           ("IMAGE",),
                "invert_mask":           ("BOOLEAN", {"default": False}),
                "x_percent":             ("FLOAT",   {"default": 50.0, "min": -999, "max": 999, "step": 0.01}),
                "y_percent":             ("FLOAT",   {"default": 50.0, "min": -999, "max": 999, "step": 0.01}),
                "scale":                 ("FLOAT",   {"default": 1.0, "min": 0.001, "max": 16.0, "step": 0.001}),
                # GrowMaskWithBlur params (identical names/ranges)
                "expand":                ("INT",     {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT",  {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners":       ("BOOLEAN", {"default": True}),
                "flip_input":            ("BOOLEAN", {"default": False}),
                "blur_radius":           ("FLOAT",   {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "lerp_alpha":            ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor":          ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask":       ("MASK",),
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image",  "mask")
    FUNCTION = "composite_gpu"
    CATEGORY = '😺dzNodes/LayerUtility'

    def composite_gpu(self, background_image, layer_image, invert_mask,
                      x_percent, y_percent, scale,
                      expand, incremental_expandrate, tapered_corners, flip_input,
                      blur_radius, lerp_alpha, decay_factor,
                      mask=None, fill_holes=False):

        import kornia.morphology as morph

        device = background_image.device
        bg  = background_image.float().to(device)  # (B1, H, W, C)
        lyr = layer_image.float().to(device)        # (B2, lH, lW, lC)
        B1, bg_H, bg_W, _C = bg.shape
        B2, lH, lW, lC     = lyr.shape
        B = max(B1, B2)

        # ── Build mask ────────────────────────────────────────────────────────
        if mask is not None:
            msk = mask.float().to(device)
            if msk.dim() == 2:
                msk = msk.unsqueeze(0)
        elif lC == 4:
            msk = lyr[..., 3]
        else:
            msk = torch.ones(B2, lH, lW, device=device)

        if invert_mask:
            msk = 1.0 - msk
        if flip_input:
            msk = 1.0 - msk

        # ── Grow mask (GrowMaskWithBlur logic, per-frame) ─────────────────────
        if expand != 0 or incremental_expandrate != 0.0:
            growmask = msk.reshape(-1, msk.shape[-2], msk.shape[-1])
            out = []
            previous_output = None
            current_expand = float(expand)

            for m in growmask:
                output = m.unsqueeze(0).unsqueeze(0)
                if abs(round(current_expand)) > 0 and output.max() > 0:
                    kernel = torch.tensor(
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]] if tapered_corners
                        else [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=torch.float32, device=device,
                    )
                    for _ in range(abs(round(current_expand))):
                        if current_expand < 0:
                            output = morph.erosion(output, kernel)
                        else:
                            output = morph.dilation(output, kernel)

                output = output.squeeze(0).squeeze(0)

                if current_expand < 0:
                    current_expand -= abs(incremental_expandrate)
                else:
                    current_expand += abs(incremental_expandrate)

                if fill_holes:
                    import scipy.ndimage
                    filled = scipy.ndimage.binary_fill_holes((output > 0).cpu().numpy())
                    output = torch.from_numpy(filled.astype(np.float32)).to(device)

                if lerp_alpha < 1.0 and previous_output is not None:
                    output = lerp_alpha * output + (1 - lerp_alpha) * previous_output
                if decay_factor < 1.0 and previous_output is not None:
                    output = output + decay_factor * previous_output
                    if output.max() > 0:
                        output = output / output.max()

                previous_output = output
                out.append(output)

            msk = torch.stack(out, dim=0)

        # ── Blur mask ─────────────────────────────────────────────────────────
        if blur_radius != 0:
            k = max(1, int(blur_radius * 3)) * 2 + 1
            coords = torch.arange(k, dtype=torch.float32, device=device) - k // 2
            g = torch.exp(-0.5 * (coords / max(blur_radius, 1e-6)) ** 2)
            g = g / g.sum()
            blur_kernel = (g[:, None] * g[None, :]).view(1, 1, k, k)
            msk = F.conv2d(msk.unsqueeze(1), blur_kernel, padding=k // 2).squeeze(1).clamp(0, 1)

        # ── Resize mask to match layer spatial dims if needed ─────────────────
        if msk.shape[-2] != lH or msk.shape[-1] != lW:
            msk = F.interpolate(
                msk.unsqueeze(1), size=(lH, lW), mode='bilinear', antialias=True,
            ).squeeze(1)

        # ── Broadcast to full batch ───────────────────────────────────────────
        idx_bg = torch.arange(B, device=device) % B1
        idx_ly = torch.arange(B, device=device) % B2
        idx_lm = torch.arange(B, device=device) % msk.shape[0]

        bg_b  = bg[idx_bg].clone()
        lyr_b = lyr[idx_ly]
        msk_b = msk[idx_lm]

        # ── Paste position (center at x_percent/y_percent of background) ──────
        x_pos = int(bg_W * x_percent / 100 - lW / 2)
        y_pos = int(bg_H * y_percent / 100 - lH / 2)

        src_y1 = max(0, -y_pos);  dst_y1 = max(0, y_pos)
        src_x1 = max(0, -x_pos);  dst_x1 = max(0, x_pos)
        dst_y2 = min(bg_H, y_pos + lH);  dst_x2 = min(bg_W, x_pos + lW)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        # ── Alpha composite (all frames at once on GPU) ───────────────────────
        out_mask = torch.zeros(B, bg_H, bg_W, device=device)

        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            m = msk_b[:, src_y1:src_y2, src_x1:src_x2].clamp(0, 1).unsqueeze(-1)
            bg_b[:, dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
                bg_b[:, dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1.0 - m)
                + lyr_b[:, src_y1:src_y2, src_x1:src_x2, :3] * m
            )
            out_mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = msk_b[:, src_y1:src_y2, src_x1:src_x2].clamp(0, 1)

        out_image = bg_b[:, :, :, :3].clamp(0.0, 1.0)

        # ── Scale whole output ────────────────────────────────────────────────
        if scale != 1.0:
            new_H = max(1, int(bg_H * scale))
            new_W = max(1, int(bg_W * scale))
            out_image = F.interpolate(
                out_image.permute(0, 3, 1, 2),
                size=(new_H, new_W), mode='bilinear', antialias=True,
            ).permute(0, 2, 3, 1)
            out_mask = F.interpolate(
                out_mask.unsqueeze(1),
                size=(new_H, new_W), mode='bilinear', antialias=True,
            ).squeeze(1)

        return (out_image, out_mask)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageCompositeGPU": LS_ImageCompositeGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageCompositeGPU": "LayerUtility: Image Composite GPU",
}
