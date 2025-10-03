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
####################################################################################################################

# quick setup of the parameters
# model setup
max_model_len = 4096
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
# prompt setup
sys_prompt = "You are a helpful assistant. Answer the question based on the image provided. In the last reasoning step, you should output 'The final answer is: \\boxed{}' and put the final answer in \\boxed{}."
sys_prompt_pure_language = "You are a helpful assistant. Answer the question based on the text provided. Put your final answer in \\boxed{}."
qwen_instruct_prompt = "\n\nPlease reason step by step, and put your final answer within \\boxed{}"

####################################################################################################################


def vllm_mllm_init(mllm_dir, tp=4, gpu_memory_utilization=0.95):

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
            #"fps": 1,
            #"do_rescale": False,
            #"use_fast": True
        },
        enable_sleep_mode=True
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


def process_image(image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

def vllm_mllm_process_batch_official(text_prompts, image_paths, mllm_dir):
    if any([x in mllm_dir for x in qwen_series]):
        messages = []
        processor = AutoProcessor.from_pretrained(mllm_dir)
        for question, image_path in zip(text_prompts, image_paths):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 1280,
                        },
                        {"type": "text", "text": question+qwen_instruct_prompt},
                    ],
                }
            ])
        texts = [processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        ) for msg in messages]
        
        image_inputs, video_inputs = process_vision_info(messages, return_video_kwargs=False)
        
        vllm_inputs = [
            {"prompt": text,"multi_modal_data": {"image":image_input}} for text, image_input in zip(texts, image_inputs)
        ]
    else:
        raise NotImplementedError("Only Qwen series is supported for now.")
    return vllm_inputs
  
def vllm_mllm_process_batch_from_messages(messages: List[List[dict]], processor):
    assert isinstance(messages, list) and all(isinstance(msg, list) for msg in messages), "messages should be a list of lists"
    vllm_inputs = []

    for msg in tqdm(messages, total=len(messages), desc="Processing vllm inputs"):                # 逐条对话处理
        prompt = processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 针对这一条对话提取所有图像 (列表)
        image_inputs, _ = process_vision_info(msg, return_video_kwargs=False)

        image_inputs, new_sizes = resize_by_token_budget(image_inputs)

        # 若模板里没占位符而 image_inputs 非空，可手动补：
        if image_inputs and ("<image>" not in prompt and "<im_start>" not in prompt):
            prompt = "<image>\n" + prompt        # 或 "<im_start><image><im_end>\n" 视模型而定

        vllm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},   # 这是“一个对话对应 N 张图”
        })

    return vllm_inputs
   
      
def vllm_mllm_process_batch_data(text_prompts, image_paths, mllm_dir):
    inputs = []
    for question, image_path in zip(text_prompts, image_paths):
        if any([x in mllm_dir for x in qwen_series]):
            prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}{qwen_instruct_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n")
            image = process_image(image_path)
            
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        else:
            raise NotImplementedError("Only Qwen series is supported for now.")
    return inputs

def vllm_mllm_process_single_data(text_prompt, image_path, mllm_dir):
    if any([x in mllm_dir for x in qwen_series]):
        prompt = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  f"{text_prompt}<|im_end|>\n"
                  "<|im_start|>assistant\n")
        image = process_image(image_path)
    else:
        raise NotImplementedError("Only Qwen series is supported for now.")
    return [{"prompt": prompt, "multi_modal_data": {"image": image}}]


def vllm_mllm_process_pure_language_batch_data(text_prompts,  mllm_dir):
    inputs = []
    for question in text_prompts:
        if any([x in mllm_dir for x in qwen_series]):
            prompt = (f"<|im_start|>system\n{sys_prompt_pure_language}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")

            inputs.append({"prompt": prompt})
        else:
            raise NotImplementedError("Only Qwen series is supported for now.")
    return inputs


def count_qwen_vl_tokens(vllm_inputs: Dict, tokenizer, processor) -> int:
    """
    vllm_input 形如:
      {
        "prompt": "...<im_start><image><im_end>...",
        "multi_modal_data": {"image": [PIL.Image, PIL.Image, ...]}
      }
    """
    lens = []
    for input in vllm_inputs: 
        PATCH =  processor.image_processor.patch_size * 2 # 14 is the ViT path size, 28 is the patch size for LLM
        # 1. 文本 token 数（<im_start>/<image>/<im_end> 都会各占 1 个）
        n_text = len(
            tokenizer(input["prompt"], add_special_tokens=False)["input_ids"]
        )

        # 2. 视觉 token 数
        n_img = 0
        if input["multi_modal_data"]["image"] is not None:
            for pil_img in input["multi_modal_data"]["image"]:
                # 用官方 image_processor，保持与模型输入完全一致
                W,H = pil_img.size
                # patch 数 = (H/patch) * (W/patch)，Qwen-VL 每张图还会多 1 个 cls token
                n_img += (H // PATCH) * (W // PATCH) + 1
            
        #num_images = len(input["multi_modal_data"]["image"])
        #print(f"num images: {num_images}, n_img: {n_img}, n_text: {n_text}")
        lens.append(n_text + n_img)
    return lens


PATCH=28
def resize_by_token_budget(images,
                           global_max_pixels=3840*PATCH*PATCH,
                           per_img_max_pixels=1280*PATCH*PATCH,
                           divisor=PATCH):
    """等比缩放，保证一条样本内所有图像像素和 ≤ global_max_pixels"""
    # 1) 统计原总像素
    
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images, None   # 大多数样本会直接返回

    # 2) 统一缩放系数
    ratio = math.sqrt(global_max_pixels / total)

    processed = []
    new_sizes = []
    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        # 保证能被 28 整除（Qwen 的 patch 大小）
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)

        # 3) 仍然超过单张上限时再单独缩放
        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, int(w * r) // divisor * divisor)
            h = max(divisor, int(h * r) // divisor * divisor)

        processed.append(img.resize((w, h), Image.BICUBIC))
        new_sizes.append((w, h))
    return processed, new_sizes

##########################################################################################
# LLM
##########################################################################################


def vllm_llm_init(llm_dir, tp=4, gpu_memory_utilization=0.8, temperature=0.2,
                  max_model_len=8192):
    llm = LLM(
        model=llm_dir, 
        tensor_parallel_size=tp, 
        trust_remote_code=True,
        seed=seed if seed else 0,
        swap_space=swap_space,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        distributed_executor_backend='ray' if tp > 1 else None,
        dtype="bfloat16",
        enable_sleep_mode=True
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        #best_of=best_of,
        max_tokens=max_tokens, 
        n=n_generate_sample,
        stop=stop,
        skip_special_tokens=False,
        seed=seed if temperature == 0 else None # vllm0.6.6.post1 
    )
    return llm, sampling_params

def vllm_llm_process_batch_data(sys_prompt: str, usr_prompts: List[str], tokenizer):
    inputs = []
    for prompt in usr_prompts:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        inputs.append(
            tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
            )
        )
    return inputs


##########################################################################################
# LLM & MLLM
##########################################################################################


def vllm_generate(
    inputs: List[str],
    sampling_params: SamplingParams,
    engine: LLM,
):
    if not inputs: return []
    
    outputs = engine.generate(inputs, sampling_params=sampling_params, use_tqdm=True)   
    return outputs


def vllm_kill_model(model: LLM):
    """
    Kill the vLLM model to free up resources.
    """
    model.sleep()
    del model
    gc.collect()
    torch.cuda.empty_cache()

def vllm_wake_model(model: LLM):
    """
    Wake up a sleeping vLLM model.
    """
    model.wake_up()
    return model