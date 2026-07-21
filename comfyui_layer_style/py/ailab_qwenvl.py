# Ported from https://github.com/1038lab/ComfyUI-QwenVL (GPL-3.0)
# Modifications: shared process-wide model cache with reference counting,
# separate processor cache, processed-input cache, parallel frame conversion,
# torch.compile with dynamic=True.

import gc
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

import folder_paths
from comfy.utils import ProgressBar

try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# ── Config paths ───────────────────────────────────────────────────────────────
_NODE_DIR = Path(__file__).parent
_CONFIG_PATH = _NODE_DIR / "ailab_qwenvl_models.json"
_PROMPTS_PATH = _NODE_DIR / "ailab_system_prompts.json"

HF_VL_MODELS: dict = {}
HF_TEXT_MODELS: dict = {}
HF_ALL_MODELS: dict = {}
SYSTEM_PROMPTS: dict = {}
PRESET_PROMPTS: list = ["Describe this image in detail."]

# ── Shared process-wide caches ─────────────────────────────────────────────────
_MODEL_LOCK = threading.Lock()

# key: (model_name, quant_value, attn_impl, device, use_compile)
# value: {'model': ..., 'processor': ..., 'tokenizer': ..., 'refcount': int}
_MODEL_CACHE: dict = {}

# key: model_path — processor/tokenizer are cheap and never evicted
_PROC_CACHE: dict = {}

# key: (cache_sig, chat_hash, image_ptr, video_ptr) — small LRU
_INPUT_CACHE: dict = {}
_INPUT_CACHE_MAX = 8

# ── Tooltips ───────────────────────────────────────────────────────────────────
TOOLTIPS = {
    "model_name": "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL.",
    "quantization": "Precision vs VRAM. FP16 gives the best quality; 8-bit suits 8–16 GB GPUs; 4-bit fits 6 GB or lower.",
    "attention_mode": "auto tries sage → flash-attn v2 → SDPA. Only override when debugging.",
    "preset_prompt": "Built-in instruction describing how Qwen-VL should analyze the media input.",
    "custom_prompt": "Optional override — when filled it completely replaces the preset template.",
    "max_tokens": "Maximum number of new tokens to decode.",
    "keep_model_loaded": "Keeps the model resident in VRAM after the run so the next prompt skips loading.",
    "seed": "Seed controlling sampling and frame picking.",
    "use_torch_compile": "Enable torch.compile (dynamic=True) on CUDA/Torch 2.1+ for extra throughput after first compile.",
    "device": "Choose where to run the model: auto, cpu, mps, or cuda:x.",
    "temperature": "Sampling randomness when num_beams == 1.",
    "top_p": "Nucleus sampling cutoff when num_beams == 1.",
    "num_beams": "Beam-search width. Values >1 disable temperature/top_p.",
    "repetition_penalty": "Values >1 penalize repeated phrases.",
    "frame_count": "Number of frames extracted from video inputs.",
}

# ── Quantization enum ──────────────────────────────────────────────────────────
class Quantization(str, Enum):
    Q4  = "4-bit (VRAM-friendly)"
    Q8  = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")

ATTENTION_MODES = ["auto", "sage", "flash_attention_2", "sdpa"]

# ── Config loading ─────────────────────────────────────────────────────────────
def load_model_configs():
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        HF_VL_MODELS   = data.get("hf_vl_models") or {}
        HF_TEXT_MODELS = data.get("hf_text_models") or {}
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
        HF_VL_MODELS = HF_TEXT_MODELS = {}

    try:
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qp = data.get("qwenvl") or {}
        pp = data.get("_preset_prompts") or []
        if isinstance(qp, dict) and qp:
            SYSTEM_PROMPTS = qp
        if isinstance(pp, list) and pp:
            PRESET_PROMPTS = pp
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    # Optional user overrides
    custom = _NODE_DIR / "ailab_qwenvl_custom_models.json"
    if custom.exists():
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            cv = data.get("hf_vl_models") or {}
            ct = data.get("hf_text_models") or {}
            if cv:
                HF_VL_MODELS.update(cv)
                print(f"[QwenVL] Loaded {len(cv)} custom VL models")
            if ct:
                HF_TEXT_MODELS.update(ct)
                print(f"[QwenVL] Loaded {len(ct)} custom text models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")

    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)

if not HF_ALL_MODELS:
    load_model_configs()

# ── Device helpers ─────────────────────────────────────────────────────────────
def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        gpu = {"available": True, "total_memory": total,
               "free_memory": total - (torch.cuda.memory_allocated(0) / 1024**3)}
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {"total": sys_mem.total / 1024**3, "available": sys_mem.available / 1024**3},
        "device_type": device_type,
        "recommended_device": recommended,
    }

def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"
    if device.isdigit():
        device = f"cuda:{int(device)}"
    if device in ("cuda", ) or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
                if idx >= torch.cuda.device_count():
                    print(f"[QwenVL] CUDA device {idx} not available, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                return "cuda:0"
        return device
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL] MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"
    return device

# ── Attention helpers ──────────────────────────────────────────────────────────
def flash_attn_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    try:
        import flash_attn  # noqa: F401
        import importlib.metadata
        importlib.metadata.version("flash_attn")
        return True
    except Exception:
        return False

def sage_attn_available():
    if not SAGE_ATTENTION_AVAILABLE or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8

def get_sage_attention_config():
    if not sage_attn_available():
        return None, None, None
    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if arch >= 120:
        return sageattn_qk_int8_pv_fp8_cuda, "per_warp", "fp32+fp32"
    if arch >= 90:
        return sageattn_qk_int8_pv_fp8_cuda_sm90, "per_warp", "fp32+fp32"
    if arch == 89:
        return sageattn_qk_int8_pv_fp8_cuda, "per_warp", "fp32+fp32"
    if arch >= 80:
        return sageattn_qk_int8_pv_fp16_cuda, "per_warp", "fp32"
    return None, None, None

def resolve_attention_mode(mode, force_sdpa=False):
    if force_sdpa:
        return "sdpa"
    if mode == "sdpa":
        return "sdpa"
    if mode == "sage":
        if sage_attn_available():
            return "sage"
        print("[QwenVL] SageAttention forced but unavailable, falling back to SDPA")
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    if sage_attn_available():
        print("[QwenVL] Auto mode: Using SageAttention")
        return "sage"
    if flash_attn_available():
        print("[QwenVL] Auto mode: Using Flash Attention 2")
        return "flash_attention_2"
    print("[QwenVL] Auto mode: Using SDPA")
    return "sdpa"

def set_sage_attention(model):
    if not sage_attn_available():
        raise ImportError("SageAttention not installed or GPU incompatible.")
    SAGE_ATTN_FUNC, QK_QUANT_GRAN, PV_ACCUM_DTYPE = get_sage_attention_config()
    if SAGE_ATTN_FUNC is None:
        raise RuntimeError("No compatible SageAttention kernel for this GPU.")

    attention_classes = []
    for cls_name, mod_path, rotary_path in [
        ("Qwen2Attention",        "transformers.models.qwen2.modeling_qwen2",     "transformers.models.qwen2.modeling_qwen2"),
        ("Qwen3Attention",        "transformers.models.qwen3.modeling_qwen3",     "transformers.models.qwen3.modeling_qwen3"),
        ("Qwen3VLTextAttention",  "transformers.models.qwen3_vl.modeling_qwen3_vl", "transformers.models.qwen3_vl.modeling_qwen3_vl"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            rotary_fn = getattr(mod, "apply_rotary_pos_emb")
            attention_classes.append((cls, rotary_fn))
        except (ImportError, AttributeError):
            pass

    if not attention_classes:
        print("[QwenVL] Could not import any attention classes for SageAttention patching")
        return

    def make_sage_forward(AttentionClass, apply_rotary_pos_emb_func):
        def sage_attention_forward(
            self, hidden_states, position_embeddings=None,
            attention_mask=None, past_key_values=None,
            cache_position=None, position_ids=None, **kwargs,
        ):
            original_dtype = hidden_states.dtype
            is_4bit = hasattr(self.q_proj, "quant_state")
            target_dtype = torch.bfloat16 if is_4bit else self.q_proj.weight.dtype
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            else:
                query_states = query_states.view(hidden_shape).transpose(1, 2)
            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            else:
                key_states = key_states.view(hidden_shape).transpose(1, 2)
            value_states = value_states.view(hidden_shape).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_func(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin if position_embeddings else None,
                    "cos": cos if position_embeddings else None,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            q_len = input_shape[1] if len(input_shape) > 1 else hidden_states.size(1)
            is_causal = attention_mask is None and q_len > 1

            attn_output = SAGE_ATTN_FUNC(
                query_states.to(target_dtype), key_states.to(target_dtype), value_states.to(target_dtype),
                tensor_layout="HND", is_causal=is_causal,
                qk_quant_gran=QK_QUANT_GRAN, pv_accum_dtype=PV_ACCUM_DTYPE,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            if attn_output.dtype != original_dtype:
                attn_output = attn_output.to(original_dtype)
            return attn_output, None
        return sage_attention_forward

    patched = 0
    for AttentionClass, rotary_fn in attention_classes:
        fwd = make_sage_forward(AttentionClass, rotary_fn)
        for module in model.modules():
            if isinstance(module, AttentionClass):
                module.forward = fwd.__get__(module, AttentionClass)
                patched += 1

    if patched:
        print(f"[QwenVL] SageAttention: Patched {patched} attention layers")
    else:
        print("[QwenVL] SageAttention: No compatible attention layers found")

# ── Model helpers ──────────────────────────────────────────────────────────────
def ensure_model(model_name):
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    models_dir = Path(llm_paths[0]) / "Qwen-VL" if llm_paths else Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            return str(target)
    snapshot_download(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False, ignore_patterns=["*.md", ".git*"])
    return str(target)

def enforce_memory(model_name, quantization, device_info):
    info = HF_ALL_MODELS.get(model_name, {})
    req = info.get("vram_requirement", {})
    needed = {Quantization.Q4: req.get("4bit", 0), Quantization.Q8: req.get("8bit", 0), Quantization.FP16: req.get("full", 0)}.get(quantization, 0)
    if not needed:
        return quantization
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            print("[QwenVL] Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        raise RuntimeError("Insufficient memory for 4-bit mode")
    return quantization

def is_fp8_model(model_name: str) -> bool:
    return any(s in model_name for s in ("-fp8", "_fp8", "-FP8", "_FP8"))

def quantization_config(model_name, quantization):
    info = HF_ALL_MODELS.get(model_name, {})
    if info.get("quantized") or is_fp8_model(model_name):
        return None, None, True
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        return cfg, None, False
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None, False
    return None, torch.float16 if torch.cuda.is_available() else torch.float32, False

# ── Shared-cache helpers ───────────────────────────────────────────────────────
def _unload_cache_entry(entry):
    """Move model to CPU and free — called with _MODEL_LOCK held."""
    m = entry.pop("model", None)
    entry.pop("processor", None)
    entry.pop("tokenizer", None)
    if m is not None:
        try:
            m.cpu()
        except Exception:
            pass
        del m

def _decrement_refcount(sig):
    """Decrement refcount for sig; unload if it reaches 0. Thread-safe."""
    if sig is None:
        return
    with _MODEL_LOCK:
        entry = _MODEL_CACHE.get(sig)
        if entry is None:
            return
        entry["refcount"] -= 1
        if entry["refcount"] <= 0:
            _unload_cache_entry(entry)
            del _MODEL_CACHE[sig]
            print(f"[QwenVL] '{sig[0]}' unloaded from cache (no more references)")
        else:
            print(f"[QwenVL] '{sig[0]}' refcount → {entry['refcount']}")

# ── Base class ─────────────────────────────────────────────────────────────────
class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL] Node on {self.device_info['device_type']}")

    def __del__(self):
        try:
            sig = self.current_signature
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.current_signature = None
            _decrement_refcount(sig)
        except Exception:
            pass

    def clear(self):
        """Release this node's reference. Model stays alive while others hold it."""
        sig = self.current_signature
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        _decrement_refcount(sig)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self, model_name, quant_value, attention_mode, use_compile, device_choice, keep_model_loaded):
        quant = enforce_memory(model_name, Quantization.from_value(quant_value), self.device_info)
        is_bnb = quant in (Quantization.Q4, Quantization.Q8)
        is_fp8 = is_fp8_model(model_name) or HF_ALL_MODELS.get(model_name, {}).get("quantized", False)
        force_sdpa = is_fp8 or is_bnb
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)

        if force_sdpa and attention_mode in ("auto", "sage", "flash_attention_2"):
            print(f"[QwenVL] {'FP8' if is_fp8 else 'BitsAndBytes'} model — forcing SDPA")
        print(f"[QwenVL] Attention backend: {attn_impl}")

        device_req = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_req)
        signature = (model_name, quant.value, attn_impl, device, use_compile)

        # Already holding the right model
        if self.current_signature == signature:
            return

        # Release current ref before acquiring new
        old_sig = self.current_signature
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        _decrement_refcount(old_sig)

        # Fast path: model already in shared cache
        with _MODEL_LOCK:
            entry = _MODEL_CACHE.get(signature)
            if entry is not None:
                entry["refcount"] += 1
                self.model = entry["model"]
                self.processor = entry["processor"]
                self.tokenizer = entry["tokenizer"]
                self.current_signature = signature
                print(f"[QwenVL] Reusing cached '{model_name}' (refcount={entry['refcount']})")
                return

        # Load fresh — outside lock to avoid blocking
        model_path = ensure_model(model_name)

        # Processor/tokenizer: cached separately, never evicted
        if model_path not in _PROC_CACHE:
            proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tok  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _PROC_CACHE[model_path] = (proc, tok)
        processor, tokenizer = _PROC_CACHE[model_path]

        quant_config, dtype, _ = quantization_config(model_name, quant)
        actual_attn = "sdpa" if attn_impl == "sage" else attn_impl
        load_kwargs = {"attn_implementation": actual_attn, "use_safetensors": True}

        if is_fp8:
            if device == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda:0"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                target_device = device

            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = "auto"
            print(f"[QwenVL] Loading FP8 '{model_name}' to {target_device}...")

            model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
            has_meta = any(p.device.type == "meta" for p in model.parameters())
            if has_meta:
                print("[QwenVL] Materializing meta tensors on CPU...")
                model = model.to_empty(device="cpu")
                try:
                    from transformers.modeling_utils import load_sharded_checkpoint
                    index_file = os.path.join(model_path, "model.safetensors.index.json")
                    if os.path.exists(index_file):
                        print("[QwenVL] Loading sharded checkpoint...")
                        load_sharded_checkpoint(model, model_path, strict=True)
                    else:
                        from transformers.modeling_utils import load_state_dict
                        from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
                        wpath = None
                        for wname in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
                            candidate = os.path.join(model_path, wname)
                            if os.path.exists(candidate):
                                wpath = candidate
                                break
                        if wpath is None:
                            raise RuntimeError(f"No weights found in {model_path}")
                        state_dict = load_state_dict(wpath)
                        try:
                            model.load_state_dict(state_dict, strict=True)
                        except RuntimeError as e:
                            print(f"[QwenVL] Strict load failed: {e}. Trying non-strict...")
                            missing, unexpected = model.load_state_dict(state_dict, strict=False)
                            if missing:
                                print(f"[QwenVL] Missing keys: {missing}")
                            if unexpected:
                                print(f"[QwenVL] Unexpected keys: {unexpected}")
                except Exception as e:
                    print(f"[QwenVL] Weight loading error: {e}")
                    raise

            print(f"[QwenVL] Moving FP8 model to {target_device}")
            model = model.to(target_device).eval()
        else:
            load_kwargs["device_map"] = device if device != "auto" else "auto"
            if dtype:
                load_kwargs["dtype"] = dtype
            if quant_config:
                load_kwargs["quantization_config"] = quant_config

            label = "base=sdpa, will_patch=sage" if attn_impl == "sage" else f"attn={actual_attn}"
            print(f"[QwenVL] Loading '{model_name}' ({quant.value}, {label})")
            model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()

        if attn_impl == "sage":
            try:
                set_sage_attention(model)
                print("[QwenVL] SageAttention enabled")
            except Exception as exc:
                print(f"[QwenVL] SageAttention patching failed: {exc}")

        model.config.use_cache = True
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = True

        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                print("[QwenVL] torch.compile enabled (dynamic=True)")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")

        # Double-checked store — another thread may have loaded while we were busy
        with _MODEL_LOCK:
            existing = _MODEL_CACHE.get(signature)
            if existing is not None:
                existing["refcount"] += 1
                self.model = existing["model"]
                self.processor = existing["processor"]
                self.tokenizer = existing["tokenizer"]
                self.current_signature = signature
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
                print(f"[QwenVL] Concurrent load detected; using cached '{model_name}'")
                return

            _MODEL_CACHE[signature] = {"model": model, "processor": processor, "tokenizer": tokenizer, "refcount": 1}

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.current_signature = signature
        print(f"[QwenVL] '{model_name}' loaded and cached")

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        return Image.fromarray((tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

    @torch.no_grad()
    def generate(self, prompt_text, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty):
        conversation = [{"role": "user", "content": []}]

        pil_image = None
        if image is not None:
            pil_image = self.tensor_to_pil(image)
            conversation[0]["content"].append({"type": "image", "image": pil_image})

        pil_frames = []
        if video is not None:
            frames_raw = list(video)
            if len(frames_raw) > frame_count:
                idx = np.linspace(0, len(frames_raw) - 1, frame_count, dtype=int)
                frames_raw = [frames_raw[i] for i in idx]
            # Parallel PIL conversion
            with ThreadPoolExecutor(max_workers=min(8, len(frames_raw) or 1)) as pool:
                pil_frames = list(pool.map(self.tensor_to_pil, frames_raw))
            if pil_frames:
                conversation[0]["content"].append({"type": "video", "video": pil_frames})

        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Check processed-input cache (keyed by model sig + chat + tensor identity)
        image_ptr = image.data_ptr() if image is not None else 0
        video_ptr = video.data_ptr() if video is not None else 0
        input_key = (self.current_signature, hash(chat), image_ptr, video_ptr)

        if input_key in _INPUT_CACHE:
            model_inputs = _INPUT_CACHE[input_key]
            print("[QwenVL] Reusing cached processed inputs")
        else:
            images      = [item["image"]  for item in conversation[0]["content"] if item["type"] == "image"]
            vid_frames  = [f for item in conversation[0]["content"] if item["type"] == "video" for f in item["video"]]
            videos      = [vid_frames] if vid_frames else None
            processed   = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
            model_dev   = next(self.model.parameters()).device
            model_inputs = {k: v.to(model_dev) if torch.is_tensor(v) else v for k, v in processed.items()}
            if len(_INPUT_CACHE) >= _INPUT_CACHE_MAX:
                _INPUT_CACHE.pop(next(iter(_INPUT_CACHE)))
            _INPUT_CACHE[input_key] = model_inputs

        stop_tokens = [self.tokenizer.eos_token_id]
        if getattr(self.tokenizer, "eot_id", None) is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False

        outputs = self.model.generate(**model_inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        input_len = model_inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

    def run(self, model_name, quantization, preset_prompt, custom_prompt, image, video,
            frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty,
            seed, keep_model_loaded, attention_mode, use_torch_compile, device):
        pbar = ProgressBar(3)
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        pbar.update_absolute(1, 3, None)
        self.load_model(model_name, quantization, attention_mode, use_torch_compile, device, keep_model_loaded)
        pbar.update_absolute(2, 3, None)
        try:
            text = self.generate(prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty)
            pbar.update_absolute(3, 3, None)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()

# ── Node classes ───────────────────────────────────────────────────────────────
class AILab_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred = "🖼️ Detailed Description"
        default_prompt = preferred if preferred in prompts else prompts[0]
        return {
            "required": {
                "model_name":        (models,                     {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization":      (Quantization.get_values(),  {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode":    (ATTENTION_MODES,            {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "preset_prompt":     (prompts,                    {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt":     ("STRING",                   {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens":        ("INT",                      {"default": 512, "min": 64, "max": 2048, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN",                  {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed":              ("INT",                      {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("RESPONSE",)
    FUNCTION      = "process"
    CATEGORY      = "🧪AILab/QwenVL"

    def process(self, model_name, quantization, preset_prompt, custom_prompt,
                attention_mode, max_tokens, keep_model_loaded, seed,
                image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt,
                        image, video, 16, max_tokens, 0.6, 0.9, 1, 1.2,
                        seed, keep_model_loaded, attention_mode, False, "auto")


class AILab_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred = "🖼️ Detailed Description"
        default_prompt = preferred if preferred in prompts else prompts[0]
        num_gpus = torch.cuda.device_count()
        device_options = ["auto", "cpu", "mps"] + [f"cuda:{i}" for i in range(num_gpus)]
        return {
            "required": {
                "model_name":         (models,                    {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization":       (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode":     (ATTENTION_MODES,           {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile":  ("BOOLEAN",                 {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device":             (device_options,            {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "preset_prompt":      (prompts,                   {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt":      ("STRING",                  {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens":         ("INT",                     {"default": 512, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature":        ("FLOAT",                   {"default": 0.6, "min": 0.1, "max": 1.0, "tooltip": TOOLTIPS["temperature"]}),
                "top_p":              ("FLOAT",                   {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": TOOLTIPS["top_p"]}),
                "num_beams":          ("INT",                     {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS["num_beams"]}),
                "repetition_penalty": ("FLOAT",                   {"default": 1.2, "min": 0.5, "max": 2.0, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "frame_count":        ("INT",                     {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS["frame_count"]}),
                "keep_model_loaded":  ("BOOLEAN",                 {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed":               ("INT",                     {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("RESPONSE",)
    FUNCTION      = "process"
    CATEGORY      = "🧪AILab/QwenVL"

    def process(self, model_name, quantization, attention_mode, use_torch_compile, device,
                preset_prompt, custom_prompt, max_tokens, temperature, top_p, num_beams,
                repetition_penalty, frame_count, keep_model_loaded, seed,
                image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt,
                        image, video, frame_count, max_tokens, temperature, top_p,
                        num_beams, repetition_penalty, seed, keep_model_loaded,
                        attention_mode, use_torch_compile, device)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL":          AILab_QwenVL,
    "AILab_QwenVL_Advanced":  AILab_QwenVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL":          "QwenVL",
    "AILab_QwenVL_Advanced":  "QwenVL (Advanced)",
}
