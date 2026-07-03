import os

# Automatically write resource_dir.ini into sibling LayerStyle packs
# so they pick up brand fonts without manual configuration.

_here = os.path.dirname(os.path.abspath(__file__))
_fonts_clinique = os.path.join(_here, "fonts", "Clinique")
_fonts_ordinary = os.path.join(_here, "fonts", "TheOrdinary")
_font_dir_line = f"FONT_dir={_fonts_clinique}, {_fonts_ordinary}\n"

_custom_nodes = os.path.dirname(_here)
_target_packs = ["comfyui_layerstyle_advance", "comfyui_layer_style"]

for pack in _target_packs:
    ini_path = os.path.join(_custom_nodes, pack, "resource_dir.ini")
    pack_dir = os.path.dirname(ini_path)
    if os.path.isdir(pack_dir):
        with open(ini_path, "w") as f:
            f.write(_font_dir_line)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
