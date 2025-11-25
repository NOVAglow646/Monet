# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import qwen2_vl_attn_forward


def _patch_qwen2_vl_forward() -> bool:
    """Patch Qwen2-VL FA2 attention forward if available."""
    try:
        import importlib
        m = importlib.import_module("transformers.models.qwen2_vl.modeling_qwen2_vl")
    except Exception:
        return False
    # Canonical class name in Qwen2-VL
    if hasattr(m, "Qwen2VLFlashAttention2"):
        m.Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
        return True
    # Fallback via registry if present
    attn_map = getattr(m, "QWEN2_VL_ATTENTION_CLASSES", None)
    if isinstance(attn_map, dict) and "flash_attention_2" in attn_map:
        attn_cls = attn_map["flash_attention_2"]
        attn_cls.forward = qwen2_vl_attn_forward
        return True
    return False


def _patch_qwen2_5_vl_forward() -> bool:
    """Patch Qwen2.5-VL FA2 attention forward across HF versions."""
    try:
        import importlib
        m = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    except Exception:
        return False

    # 1) Prefer the official registry when available (most stable).
    attn_map = getattr(m, "QWEN2_5_VL_ATTENTION_CLASSES", None)
    if isinstance(attn_map, dict) and "flash_attention_2" in attn_map:
        attn_cls = attn_map["flash_attention_2"]
        # Only patch the FA2 class, not the generic attention.
        attn_cls.forward = qwen2_vl_attn_forward
        return True

    # 2) Known class names across versions/forks.
    for name in ("Qwen2_5_VLFlashAttention2",):
        if hasattr(m, name):
            getattr(m, name).forward = qwen2_vl_attn_forward
            return True

    # 3) As a last resort, some versions only expose a unified attention class.
    #    WARNING: Patching a generic class may also affect SDPA/eager paths.
    for name in ("Qwen2_5_VLAttention",):
        if hasattr(m, name):
            getattr(m, name).forward = qwen2_vl_attn_forward
            return True

    return False

'''def apply_ulysses_patch(model_type: str) -> None:
    """
    Make flash-attn compatible with Ulysses sharding by patching attention forward.
    For FA2-enabled LLaMA-like backbones, override the registry entry.
    For Qwen2/2.5-VL, patch the concrete FA2 attention class (version-safe).
    """
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2", "qwen3", "qwen3_moe"):
        # Redirect HF's "flash_attention_2" backend to our FA2 forward.
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
        return

    if model_type == "qwen2_vl":
        if not _patch_qwen2_vl_forward():
            raise RuntimeError("Failed to patch Qwen2-VL FA2 attention forward.")
        return

    if model_type == "qwen2_5_vl":
        if not _patch_qwen2_5_vl_forward():
            raise RuntimeError(
                "Failed to patch Qwen2.5-VL FA2 attention forward. "
                "Your transformers version may have renamed/hidden FA2 classes."
            )
        return

    raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")'''

def apply_ulysses_patch(model_type: str) -> None:
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2", "qwen3", "qwen3_moe"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2

        Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
        Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")
