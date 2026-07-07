from .imagefunc import *
import math
import torch.nn.functional as F


def composite_layer(background_image: Image, layer_image: Image,
                    x_percent: float, y_percent: float, scale: float = 1.0,
                    rotate: float = 0, anti_aliasing: int = 1, opacity: int = 100) -> list:

    orig_layer_width, orig_layer_height = layer_image.size
    target_layer_width = int(orig_layer_width * scale)
    target_layer_height = int(orig_layer_height * scale)
    # scale
    _layer = layer_image.resize((target_layer_width, target_layer_height))
    _mask = _layer.split()[3]
    # rotate
    _layer, _mask, _ = image_rotate_extend_with_alpha(_layer.convert('RGB'), rotate, _mask, 'lanczos', anti_aliasing)

    # 处理位置
    x = int(background_image.width * x_percent / 100 - _layer.width / 2)
    y = int(background_image.height * y_percent / 100 - _layer.height / 2)

    # composit layer
    _comp = copy.copy(background_image)
    _compmask = Image.new("L", _comp.size, color='black')
    _comp.paste(_layer, (x, y))
    _compmask.paste(_mask, (x, y))

    # composition background
    bg = background_image.copy()
    bg.paste(_comp, mask=_compmask)

    # draw masks
    whiteimage = Image.new("L",  _mask.size, 'white')
    layer_mask = Image.merge("RGBA", (whiteimage, whiteimage, whiteimage, _mask))
    mask = Image.new("RGBA", bg.size, 'black')
    mask.alpha_composite(layer_mask, (x, y))
    bbox = Image.new("RGBA", bg.size, 'black')
    bbox.alpha_composite(whiteimage.convert("RGBA"), (x, y))

    return [bg.convert("RGB"), mask.convert("L"), bbox.convert("L")]

def sdf_rounded_rect_4corner(px, py, x1, y1, x2, y2, r):
    """
    r = [r_tl, r_tr, r_br, r_bl] 四个角不同半径
    顺序：左上、右上、右下、左下
    """
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5
    hh = (y2 - y1) * 0.5

    dx = px - cx
    dy = py - cy

    # 按象限选择对应圆角半径
    r_tl, r_tr, r_br, r_bl = r

    # 每个象限对应的半径
    r_corner = np.where(
        (dx < 0) & (dy < 0), r_tl,
        np.where(
            (dx > 0) & (dy < 0), r_tr,
            np.where(
                (dx > 0) & (dy > 0), r_br,
                r_bl
            )
        )
    )

    # 剩余半宽高
    ex = hw - r_corner
    ey = hh - r_corner

    dx2 = np.abs(dx) - ex
    dy2 = np.abs(dy) - ey

    ox = np.maximum(dx2, 0)
    oy = np.maximum(dy2, 0)

    outside = np.hypot(ox, oy)
    inside = np.minimum(np.maximum(dx2, dy2), 0)

    return outside + inside - r_corner


def smoothstep(t):
    return t * t * (3 - 2 * t)


def rounded_rect_gradient_mask_numpy(image, outer_box, inner_box, outer_radius):
    w, h = image.size

    if outer_box[0] <= 0:
        outer_box = [-outer_radius, outer_box[1], outer_box[2], outer_box[3]]
    if outer_box[1] <= 0:
        outer_box = [outer_box[0], -outer_radius, outer_box[2], outer_box[3]]
    if outer_box[2] >= w:
        outer_box = [outer_box[0], outer_box[1], w+outer_radius, outer_box[3]]
    if outer_box[3] >= h:
        outer_box = [outer_box[0], outer_box[1], outer_box[2], h+outer_radius]

    xs = np.arange(w) + 0.5
    ys = np.arange(h) + 0.5
    px, py = np.meshgrid(xs, ys)

    ox1, oy1, ox2, oy2 = outer_box
    ix1, iy1, ix2, iy2 = inner_box

    # ---- 自动禁用圆角边 ----
    inner_radius = outer_radius // 2
    # 四个角默认使用同一半径
    ro = np.array([outer_radius, outer_radius, outer_radius, outer_radius], dtype=np.float32)
    ri = np.array([inner_radius, inner_radius, inner_radius, inner_radius], dtype=np.float32)

    # 左边重叠：左上、左下角 = 0
    if ox1 == ix1:
        ro[0] = ro[3] = 0
        ri[0] = ri[3] = 0

    # 右边重叠：右上、右下角 = 0
    if ox2 == ix2:
        ro[1] = ro[2] = 0
        ri[1] = ri[2] = 0

    # 上边重叠：左上、右上角 = 0
    if oy1 == iy1:
        ro[0] = ro[1] = 0
        ri[0] = ri[1] = 0

    # 下边重叠：左下、右下角 = 0
    if oy2 == iy2:
        ro[3] = ro[2] = 0
        ri[3] = ri[2] = 0

    # ---- 计算 SDF ----
    sd_outer = sdf_rounded_rect_4corner(px, py, ox1, oy1, ox2, oy2, ro)
    sd_inner = sdf_rounded_rect_4corner(px, py, ix1, iy1, ix2, iy2, ri)

    mask = np.zeros((h, w), dtype=np.float32)

    inside_inner = sd_inner < 0
    outside_outer = sd_outer > 0
    transition = ~(inside_inner | outside_outer)

    mask[inside_inner] = 1.0
    mask[outside_outer] = 0.0

    sd_i = sd_inner[transition]
    sd_o = sd_outer[transition]

    t = 1.0 - (sd_i / (sd_i - sd_o + 1e-9))
    t = np.clip(t, 0, 1)

    t = smoothstep(t)

    mask[transition] = t

    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def _gpu_rotate_extend(lyr_nchw, msk_nchw, angle_deg, device):
    """Rotate with canvas extension on GPU so no corners are clipped."""
    import kornia.geometry.transform as KGT
    B, C, H, W = lyr_nchw.shape
    angle_rad = math.radians(angle_deg)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_W = int(math.ceil(W * cos_a + H * sin_a)) + 2
    new_H = int(math.ceil(W * sin_a + H * cos_a)) + 2
    pad_left  = (new_W - W) // 2
    pad_right = new_W - W - pad_left
    pad_top   = (new_H - H) // 2
    pad_bottom = new_H - H - pad_top

    lyr_pad = F.pad(lyr_nchw, (pad_left, pad_right, pad_top, pad_bottom))
    msk_pad = F.pad(msk_nchw, (pad_left, pad_right, pad_top, pad_bottom))

    angle_t = torch.tensor([angle_deg], dtype=torch.float32, device=device).expand(B)
    lyr_rot = KGT.rotate(lyr_pad, angle_t, mode='bilinear', padding_mode='zeros')
    msk_rot = KGT.rotate(msk_pad, angle_t, mode='bilinear', padding_mode='zeros')
    return lyr_rot, msk_rot


class LS_ImageCompositeHandleMask:

    def __init__(self):
        self.NODE_NAME = 'ImageCompositeHandleMask'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        mirror_mode = ['None', 'horizontal', 'vertical']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        handle_detect_list = ['mask_area', 'layer_bbox']
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1e4, "step": 0.001}),
                "rotate": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.01}),
                "mirror": (mirror_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "handle_detect": (handle_detect_list,),
                "top_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "bottom_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "left_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "right_handle": ("FLOAT", {"default": 0.3, "min": 0, "max": 5, "step": 0.01}),
                "handle_mask_outradius": ("INT", {"default": 128, "min": 8, "max": 9999, "step": 1}),
                "top_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "BOX", "STRING")
    RETURN_NAMES = ("image", "mask", "layer_bbox_mask", "handle_mask", "handle_crop_box", "handle_overrange")
    FUNCTION = 'image_composite_handle_mask'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_composite_handle_mask(self, background_image, layer_image, invert_mask, opacity,
                                    x_percent, y_percent, scale, rotate, mirror, anti_aliasing,
                                    handle_detect, top_handle, bottom_handle, left_handle, right_handle,
                                    handle_mask_outradius,
                                    top_reserve, bottom_reserve, left_reserve, right_reserve,
                                    round_to_multiple, layer_mask=None,):

        handle_overrange = "None"
        device = background_image.device

        # ── Tensors to float32 on device ─────────────────────────────────────
        bg  = background_image.float()   # (B1, H, W, C)
        lyr = layer_image.float()        # (B2, lH, lW, lC)
        B1, bg_H, bg_W, _C = bg.shape
        B2, lH, lW, lC     = lyr.shape
        B = max(B1, B2)

        # ── Build layer mask ─────────────────────────────────────────────────
        if layer_mask is not None:
            lmask = layer_mask.float().to(device)
            if lmask.dim() == 2:
                lmask = lmask.unsqueeze(0)
            if invert_mask:
                lmask = 1.0 - lmask
        else:
            if lC == 4:
                lmask = lyr[..., 3]                                    # (B2, lH, lW)
            else:
                lmask = torch.ones(B2, lH, lW, device=device)

        # Preserve original opacity check semantics (INT widget: only fires for opacity==0)
        if 0 <= opacity < 1.0:
            lmask = lmask * opacity

        # ── Mirror (batch GPU op) ────────────────────────────────────────────
        if mirror == 'horizontal':
            lyr   = torch.flip(lyr,   dims=[2])
            lmask = torch.flip(lmask, dims=[2])
        elif mirror == 'vertical':
            lyr   = torch.flip(lyr,   dims=[1])
            lmask = torch.flip(lmask, dims=[1])

        # ── Scale (F.interpolate, batch) ─────────────────────────────────────
        if scale != 1.0:
            new_lH = max(1, int(lH * scale))
            new_lW = max(1, int(lW * scale))
            lyr = F.interpolate(
                lyr.permute(0, 3, 1, 2),
                size=(new_lH, new_lW), mode='bilinear', antialias=True,
            ).permute(0, 2, 3, 1)
            lmask = F.interpolate(
                lmask.unsqueeze(1),
                size=(new_lH, new_lW), mode='bilinear', antialias=True,
            ).squeeze(1)
            lH, lW = new_lH, new_lW

        # ── Rotate with canvas extension (kornia, batch) ─────────────────────
        if rotate != 0:
            lyr_nchw = lyr.permute(0, 3, 1, 2)            # (B2, C, lH, lW)
            msk_nchw = lmask.unsqueeze(1)                  # (B2, 1, lH, lW)
            lyr_nchw, msk_nchw = _gpu_rotate_extend(lyr_nchw, msk_nchw, rotate, device)
            lyr   = lyr_nchw.permute(0, 2, 3, 1)          # (B2, new_lH, new_lW, C)
            lmask = msk_nchw.squeeze(1)                    # (B2, new_lH, new_lW)
            lH, lW = lyr.shape[1], lyr.shape[2]

        # ── Broadcast to full batch ───────────────────────────────────────────
        idx_bg = torch.arange(B, device=device) % B1
        idx_ly = torch.arange(B, device=device) % B2
        idx_lm = torch.arange(B, device=device) % lmask.shape[0]

        bg_b   = bg[idx_bg].clone()    # (B, bg_H, bg_W, C)
        lyr_b  = lyr[idx_ly]           # (B, lH, lW, lC)
        msk_b  = lmask[idx_lm]         # (B, lH, lW)

        # ── Compute paste position ────────────────────────────────────────────
        x = int(bg_W * x_percent / 100 - lW / 2)
        y = int(bg_H * y_percent / 100 - lH / 2)

        src_y1 = max(0, -y);  dst_y1 = max(0, y)
        src_x1 = max(0, -x);  dst_x1 = max(0, x)
        dst_y2 = min(bg_H, y + lH);  dst_x2 = min(bg_W, x + lW)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        # ── Alpha composite (batch GPU op) ───────────────────────────────────
        ret_masks      = torch.zeros(B, bg_H, bg_W, device=device)
        bbox_masks     = torch.zeros(B, bg_H, bg_W, device=device)

        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            m = msk_b[:, src_y1:src_y2, src_x1:src_x2].unsqueeze(-1).clamp(0, 1)  # (B, h, w, 1)
            layer_rgb = lyr_b[:, src_y1:src_y2, src_x1:src_x2, :3]               # (B, h, w, 3)
            bg_slice  = bg_b[:, dst_y1:dst_y2, dst_x1:dst_x2, :3]
            bg_b[:, dst_y1:dst_y2, dst_x1:dst_x2, :3] = bg_slice * (1.0 - m) + layer_rgb * m

            ret_masks[:, dst_y1:dst_y2, dst_x1:dst_x2] = \
                msk_b[:, src_y1:src_y2, src_x1:src_x2].clamp(0, 1)
            bbox_masks[:, dst_y1:dst_y2, dst_x1:dst_x2] = 1.0

        # ── mask_area on GPU (computed from frame 0; geometry is fixed) ───────
        det_mask = ret_masks[0] if handle_detect == "mask_area" else bbox_masks[0]
        rows = det_mask.any(dim=1)   # (bg_H,)
        cols = det_mask.any(dim=0)   # (bg_W,)
        if rows.any() and cols.any():
            my1 = int(rows.nonzero(as_tuple=False)[0,  0].item())
            my2 = int(rows.nonzero(as_tuple=False)[-1, 0].item()) + 1
            mx1 = int(cols.nonzero(as_tuple=False)[0,  0].item())
            mx2 = int(cols.nonzero(as_tuple=False)[-1, 0].item()) + 1
            det_x, det_y, det_w, det_h = mx1, my1, mx2 - mx1, my2 - my1
        else:
            det_x, det_y, det_w, det_h = 0, 0, bg_W, bg_H

        # ── Crop-box geometry (same as original) ─────────────────────────────
        x1 = det_x - left_reserve
        y1 = det_y - top_reserve
        x2 = x1 + det_w + right_reserve
        y2 = y1 + det_h + bottom_reserve
        if x1 < 0: x1 = 0
        if x2 > bg_W: x2 = bg_W
        if y1 < 0: y1 = 0
        if y2 > bg_H: y2 = bg_H
        mask_box = (x1, y1, x2, y2)

        side_length = ((x2 - x1) + (y2 - y1)) // 2
        handle_x1 = int(x1 - left_handle  * side_length - 1)
        handle_x2 = int(x2 + right_handle * side_length + 1)
        handle_y1 = int(y1 - top_handle   * side_length - 1)
        handle_y2 = int(y2 + bottom_handle* side_length + 1)

        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            handle_width  = num_round_up_to_multiple(handle_x2 - handle_x1, multiple)
            handle_height = num_round_up_to_multiple(handle_y2 - handle_y1, multiple)
            handle_x1 = handle_x1 - (handle_width  - (handle_x2 - handle_x1)) // 2
            handle_y1 = handle_y1 - (handle_height - (handle_y2 - handle_y1)) // 2
            handle_x2 = handle_x1 + handle_width
            handle_y2 = handle_y1 + handle_height

        if handle_x1 < 0:
            handle_x1 = 0
            if round_to_multiple != 'None':
                handle_x2 = num_round_up_to_multiple(handle_x2, multiple)
        if handle_x2 > bg_W:
            if round_to_multiple != 'None':
                handle_x1 = handle_x2 - num_round_up_to_multiple(handle_x2 - handle_x1, multiple)
            handle_x2 = bg_W
        if handle_y1 < 0:
            handle_y1 = 0
            if round_to_multiple != 'None':
                handle_y2 = num_round_up_to_multiple(handle_y2, multiple)
        if handle_y2 > bg_H:
            if round_to_multiple != 'None':
                handle_y1 = handle_y2 - num_round_up_to_multiple(handle_y2 - handle_y1, multiple)
            handle_y2 = bg_H

        crop_box = (handle_x1, handle_y1, handle_x2, handle_y2)

        # ── Handle overrange check ────────────────────────────────────────────
        if handle_x1 <= 0 or handle_x2 >= bg_W or handle_y1 <= 0 or handle_y2 >= bg_H:
            top = "top,"   if handle_y1 <= 0   else ""
            bot = "bottom," if handle_y2 >= bg_H else ""
            lft = "left,"  if handle_x1 <= 0   else ""
            rgt = "right"  if handle_x2 >= bg_W else ""
            handle_overrange = f"{top}{bot}{lft}{rgt}"
            log(f"{self.NODE_NAME} handle overrange: {handle_overrange}")

        # ── rounded_rect_gradient_mask: computed ONCE, expanded to all frames ─
        dummy_pil = Image.new('L', (bg_W, bg_H))
        handle_mask_pil = rounded_rect_gradient_mask_numpy(
            dummy_pil, list(crop_box), list(mask_box), handle_mask_outradius
        )
        handle_mask_t = image2mask(handle_mask_pil).to(device)   # (1, bg_H, bg_W)
        handle_mask_t = handle_mask_t.expand(B, -1, -1)          # (B, bg_H, bg_W)

        log(f"{self.NODE_NAME} Processed {B} image(s).", message_type='finish')

        return (
            bg_b[:, :, :, :3].clamp(0.0, 1.0),
            ret_masks,
            bbox_masks,
            handle_mask_t,
            list(crop_box),
            handle_overrange,
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageCompositeHandleMask": LS_ImageCompositeHandleMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageCompositeHandleMask": "LayerUtility: Image Composite Handle Mask",
}
