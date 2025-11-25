import torch
from vllm import LLM, SamplingParams, EngineArgs
from dataclasses import asdict
from typing import List
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers import AutoTokenizer,AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import gc
import math
from PIL import Image


# quick setup of the parameters
# model setup
max_num_seqs = 1024 # processing speed
#tp = 4
temperature = 0.1
top_k = 50
top_p = 0.8
repetition_penalty = 1.01
best_of = 1
n_generate_sample = best_of
max_tokens = 4096
swap_space = 7
seed = 0
stop = None

qwen_series = ["Qwen2.5-VL-7B", "Qwen2-VL-7B", "ThinkLite-VL-7B", "Vision-R1", "qwen", "VL-Rethinker-7B"]
max_pixels = 1280*28*28
min_pixels = 256*28*28


def vllm_mllm_init(mllm_dir: str, tp=4, gpu_memory_utilization=0.95, max_model_len=4096):

    engine_args = EngineArgs(
        model=mllm_dir,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tp, 
        trust_remote_code=True,
        seed=seed,
        swap_space=swap_space,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend='ray' if tp > 1 else None,
        dtype="bfloat16",
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
        enable_sleep_mode=True,
        enable_chunked_prefill=True,
    )
    engine_args = asdict(engine_args)
    mllm = LLM(
        **engine_args
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        n=n_generate_sample,
        stop=stop,
        skip_special_tokens=False,
        seed=seed if temperature == 0 else None, # vllm0.6.6.post1
    )
    return mllm, sampling_params



def vllm_mllm_process_batch_from_messages(messages: List[List[dict]], processor):
    assert isinstance(messages, list) and all(isinstance(msg, list) for msg in messages), "messages should be a list of lists"
    vllm_inputs = []

    for msg in tqdm(messages, total=len(messages), desc="Processing vllm inputs"):
        prompt = processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(msg, return_video_kwargs=False)

        if image_inputs and ("<image>" not in prompt and "<im_start>" not in prompt):
            prompt = "<image>\n" + prompt
        vllm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        })

    return vllm_inputs
   
      


def vllm_generate(
    inputs,
    sampling_params: SamplingParams,
    engine: LLM,
):
    if not inputs: return []
    
    outputs = engine.generate(inputs, sampling_params=sampling_params, use_tqdm=True)   
    return outputs

