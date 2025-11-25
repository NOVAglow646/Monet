import inference.apply_vllm_monet # the patch must be applied before importing vllm
import PIL.Image
from inference.load_and_gen_vllm import *
import os
import PIL

model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-avt_stage1-08_17-6-20-30-wt1.0-ce_emphasize3.0'

def main():
    
    mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.15)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    os.environ['LATENT_SIZE'] = '10'

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question:  Which car has the longest rental period? The choices are listed below:\n(A)DB11 COUPE.\n(B) V12 VANTAGES COUPES.\n(C) VANQUISH VOLANTE.\n(D) V12 VOLANTE.\n(E) The image does not feature the time."},
                    {"type": "image", "image": PIL.Image.open('images/example_question.png').convert("RGB")}
                ]
            }
        ]
    ]

    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    print(output[0].outputs[0].text)



if __name__ == '__main__':
    main()

