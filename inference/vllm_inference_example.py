import inference.apply_vllm_monet # the patch must be applied before importing vllm
import PIL.Image
from inference.load_and_gen_vllm import *
import os
import PIL
import re
model_path = 'Path/to/your/model'
def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)


def main():
    
    mllm, sampling_params = vllm_mllm_init(model_path, tp=1, gpu_memory_utilization=0.8)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    os.environ['LATENT_SIZE'] = '10'

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question:  Which car has the longest rental period? The choices are listed below:\n(A)DB11 COUPE.\n(B) V12 VANTAGES COUPES.\n(C) VANQUISH VOLANTE.\n(D) V12 VOLANTE.\n(E) The image does not feature the time. Put your final answer in \\boxed{}."},
                    {"type": "image", "image": PIL.Image.open('images/example_question.png').convert("RGB")}
                ]
            }
        ]
    ]

    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    raw_output_text = output[0].outputs[0].text    
    cleaned_output_text = replace_abs_vis_token_content(raw_output_text)
    print(cleaned_output_text)



if __name__ == '__main__':
    main()

