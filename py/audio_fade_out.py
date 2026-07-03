import torch


class AudioFadeOut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio to apply a fade-out to."}),
                "duration_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1,
                    "tooltip": "The duration of the fade-out effect in seconds."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade"
    CATEGORY = "⚪Lum3on/AudioTools/Effects"

    def fade(self, audio: dict, duration_seconds: float):
        w_batch, sr = audio["waveform"], audio["sample_rate"]
        processed_list = []
        for w in w_batch:
            w_copy = w.clone()
            num_channels, total_samples = w_copy.shape
            fade_samples = min(int(duration_seconds * sr), total_samples)
            if fade_samples > 0:
                fade_curve = torch.linspace(1.0, 0.0, fade_samples, device=w_copy.device).unsqueeze(0)
                w_copy[:, -fade_samples:] *= fade_curve
            processed_list.append(w_copy)
        return ({"waveform": torch.stack(processed_list), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "AudioFadeOut": AudioFadeOut
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFadeOut": "Audio Fade Out"
}
