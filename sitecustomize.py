import os, sys, importlib
os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["ABS_VIS_START_ID"] = "151666"
os.environ["ABS_VIS_END_ID"] = "151667"
try:
    # Import your patched runner by module name
    patched = importlib.import_module("new.monet_qwen_model.vllm.monet_gpu_model_runner")
    for key in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.worker.gpu_model_runner",
        "vllm.worker.model_runner",
    ):
        sys.modules[key] = patched

    print("[Monet] vLLM runner patched via sitecustomize:", __file__)
except Exception as e:
    print("[Monet] sitecustomize failed:", repr(e))