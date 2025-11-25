
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
qwen_series = ["Qwen2.5-VL-7B", "Qwen2-VL-7B", "ThinkLite-VL-7B", "Vision-R1", "qwen", "VL-Rethinker-7B"]
qwen_instruct_prompt = "\n\nPlease reason step by step, and put your final answer within \\boxed{}"
qwen_RL_instruct_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

def hf_mllm_init(mllm_dir):
    if any([x in mllm_dir for x in qwen_series]):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm_dir, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(
            mllm_dir, min_pixels=256*28*28, max_pixels=1280*28*28,
        )
    else:
        raise NotImplementedError("Only Qwen series is supported for now.")
    
    return model, processor

def hf_process_batch_data(text_prompts, image_paths, mllm_dir, processor, device):
    if any([x in mllm_dir for x in qwen_series]):
        messages = []
        for question, image_path in zip(text_prompts, image_paths):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question+qwen_instruct_prompt},
                    ],
                }
            ]
            )
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )    
    else:
        raise NotImplementedError("Only Qwen series is supported for now.")
    inputs = inputs.to(device)
    return inputs

def hf_generate(model, processor, inputs, max_new_tokens=1024):
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)