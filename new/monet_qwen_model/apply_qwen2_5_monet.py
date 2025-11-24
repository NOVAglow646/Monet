# apply_qwen_patch.py  —— 在 import transformers 之前运行
import importlib.util, sys, pathlib, os

patch_path = pathlib.Path(__file__).with_name("modeling_qwen2_5_vl_monet.py")

spec  = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    patch_path,
)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = patched_mod

print("Replaced the original Qwen2.5-VL model with the Monet version.")
