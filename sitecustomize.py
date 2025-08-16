# sitecustomize.py (top-level)
# Runs in every Python process (parent + spawned workers)

import os, sys, importlib

os.environ.setdefault("VLLM_USE_V1", "1")  # force V1 engine if desired

try:
    # Import your patched runner by module name
    patched = importlib.import_module("new.avt_qwen_model.vllm.avt_gpu_model_runner")

    # Bind to both V1 and V0 keys so either engine path hits this module
    for key in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.worker.gpu_model_runner",
        "vllm.worker.model_runner",
    ):
        sys.modules[key] = patched

    print("[AVT] vLLM runner patched via sitecustomize:", __file__)
except Exception as e:
    print("[AVT] sitecustomize failed:", repr(e))

'''import os
if os.environ.get("VLLM_DEBUGPY_PORT"):
    try:
        import debugpy, socket
        host = os.environ.get("VLLM_DEBUGPY_HOST", "127.0.0.1")
        port = int(os.environ["VLLM_DEBUGPY_PORT"])
        debugpy.listen((host, port))  # bind once per process
        print(f"[debugpy] Listening on {host}:{port} (pid={os.getpid()})")
        if os.environ.get("VLLM_DEBUGPY_WAIT", "0") == "1":
            # Wait only if you want the worker to pause until VS Code attaches
            debugpy.wait_for_client()
            print("[debugpy] Attached.")
    except Exception as e:
        print(f"[debugpy] Failed: {e}")
'''