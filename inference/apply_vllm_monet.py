import os, sys, importlib

def patch():
    os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    os.environ["LATENT_START_ID"] = "151666"
    os.environ["LATENT_END_ID"] = "151667"
    try:
        # apply Monet transformers model patch
        patched = importlib.import_module("inference.vllm.monet_gpu_model_runner")
    
        # apply Monet vLLM GPU model runner patch
        for key in (
            "vllm.v1.worker.gpu_model_runner",
            "vllm.worker.gpu_model_runner",
            "vllm.worker.model_runner",
        ):
            sys.modules[key] = patched

        print("[Monet] vLLM runner patched via sitecustomize:", __file__)
    except Exception as e:
        print("[Monet] sitecustomize failed:", repr(e))

patch()