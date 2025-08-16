import os, importlib, sys

import importlib.util, sys, pathlib, os

os.environ.setdefault("VLLM_USE_V1", "1")  # if you target V1
print("[PATCH] apply_qwen2_5_avt_gpu_model_runner loaded:", __file__)

patch_path = pathlib.Path(__file__).with_name("avt_gpu_model_runner.py")

# Load the patched module once under a private name
spec = importlib.util.spec_from_file_location("avt_gpu_model_runner_patched", patch_path)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

# Write to both V1 and V0 import keys
CANDIDATE_KEYS = [
    "vllm.v1.worker.gpu_model_runner",
    "vllm.worker.gpu_model_runner",
    "vllm.worker.model_runner",
]
for key in CANDIDATE_KEYS:
    if key in sys.modules:
        print(f"[PATCH] apply_qwen2_5_avt_gpu_model_runner: patching {key}")
        del sys.modules[key]
    sys.modules[key] = patched_mod


