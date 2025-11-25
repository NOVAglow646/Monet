# sitecustomize.py
# Runs in every Python process (parent + spawned workers)
import os, sys, importlib
import traceback

import os, sys, importlib, functools, typing as T

def patch():
    os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    os.environ["ABS_VIS_START_ID"] = "151666"
    os.environ["ABS_VIS_END_ID"] = "151667"
    os.environ["AVT_LATENT_HOOK_BIN"] = "1"
    # ---------------------------
    AVT_DEBUG = int(os.environ.get("AVT_DEBUG", "0"))

    def _eprint(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)


    # Optional entry log (stderr only) when debugging is enabled
    if AVT_DEBUG:
        _eprint("Entering sitecustomize.py")
    try:
        sys.modules['vllm.v1.worker.gpu_model_runner'] = importlib.import_module("monet_models.vllm.monet_gpu_model_runner")
        sys.modules['transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'] = importlib.import_module("monet_models.transformers.monet_modeling_qwen2_5_vl")
        _eprint("[Easyr1 AVT vllm init] Replaced original vllm.v1.worker.gpu_model_runner with monet_models.vllm.monet_gpu_model_runner")
    except Exception as e:
        _eprint(f"[AVT vllm init] Override failed: {e}")
        traceback.print_exc()

patch()