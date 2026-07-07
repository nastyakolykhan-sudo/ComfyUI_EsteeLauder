"""
WAS Suite nodes (minimal subset)
---------------------------------
Sourced from WASasquatch/was-node-suite-comfyui. Ported to ComfyUI_EsteeLauder_prod.
Nodes: Text Multiline, StringReplace
"""

import re


# ── Text Multiline ────────────────────────────────────────────────────────────

class WAS_Text_Multiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_multiline"
    CATEGORY = "WAS Suite/Text"

    def text_multiline(self, text):
        import io
        new_text = []
        for line in io.StringIO(text):
            if not line.strip().startswith('#'):
                new_text.append(line.replace("\n", ''))
        return ("\n".join(new_text),)


# ── StringReplace ─────────────────────────────────────────────────────────────

class WAS_StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "find": ("STRING", {"default": ""}),
                "replace": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_replace"
    CATEGORY = "WAS Suite/Text"

    def string_replace(self, string, find, replace):
        return (string.replace(find, replace),)


# ── StringConcatenate ─────────────────────────────────────────────────────────

class WAS_StringConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"default": "", "forceInput": True}),
                "string_b": ("STRING", {"default": "", "forceInput": True}),
                "delimiter": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concatenate"
    CATEGORY = "WAS Suite/Text"

    def concatenate(self, string_a, string_b, delimiter=""):
        return (string_a + delimiter + string_b,)


NODE_CLASS_MAPPINGS = {
    "Text Multiline": WAS_Text_Multiline,
    "StringReplace": WAS_StringReplace,
    "StringConcatenate": WAS_StringConcatenate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Text Multiline": "WAS: Text Multiline",
    "StringReplace": "WAS: String Replace",
    "StringConcatenate": "WAS: String Concatenate",
}
