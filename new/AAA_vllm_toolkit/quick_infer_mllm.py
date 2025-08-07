model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct'
import PIL.Image
from load_and_gen_vllm import *
from load_and_gen_hf import *
import os
import PIL

mode = 'hf'
if mode == 'vllm':
    mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.15)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    #inputs = vllm_mllm_process_single_data("Describe this image in detail.", image_path='/data1/qxwang/codes/Mirage/new/debug_1.jpg', mllm_dir=model_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The object located at "},
                {"type": "image", "image": PIL.Image.open('/home/dids/shiyang/codes/abstract-visual-token/asset/pipeline.png').convert("RGB")},
                {"type": "text", "text": " is a:"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The object located at "},
                {"type": "text", "text": " is a:"}
            ]
        },
    ]
    inputs = vllm_mllm_process_batch_from_messages([conversation], processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    print(output[0].outputs[0].text)
    mllm.sleep()
elif mode == 'hf':
    mllm, processor = hf_mllm_init(model_path)
    inputs = hf_process_batch_data(
        text_prompts=["Describe this image in detail.","Describe this image in detail."],
        image_paths=['/home/dids/shiyang/codes/abstract-visual-token/asset/pipeline.png','/home/dids/shiyang/codes/abstract-visual-token/asset/pipeline.png'],
        mllm_dir=model_path,
        processor=processor,
        device="cuda:0"
    )
    output = hf_generate(mllm, processor, inputs)
    print(output[0])