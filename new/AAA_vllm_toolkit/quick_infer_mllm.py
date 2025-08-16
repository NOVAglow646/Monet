model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-0812-avt_sft-shuffle'
model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct'

model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-avt_sft-shuffle-obs-ce-factor-2.0'
model_path = '/home/dids/shiyang/checkpoints/08_15-avt_stage1-6-30-40-wt1.0-ep2'
import PIL.Image
#import new.avt_qwen_model.vllm.apply_qwen2_5_avt_gpu_model_runner
from new.AAA_vllm_toolkit.load_and_gen_vllm import *
from new.AAA_vllm_toolkit.load_and_gen_hf import *
import os
import PIL



def main():
    mode = 'vllm'
    if mode == 'vllm':
        mllm, sampling_params = vllm_mllm_init(model_path, tp=4, gpu_memory_utilization=0.15)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # inputs = vllm_mllm_process_single_data("Describe this image in detail.", image_path='/data1/qxwang/codes/Mirage/new/debug_1.jpg', mllm_dir=model_path)
        os.environ['ABS_VIS_START_ID'] = str(processor.tokenizer.encode('<|vision_start|>')[0])
        os.environ['ABS_VIS_END_ID'] = str(processor.tokenizer.encode('<|vision_end|>')[0])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: What is the Potassium in Broccoli, frozen (0)?\nAnswer the question using a single word or phrase.\nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."},
                    {"type": "image", "image": PIL.Image.open('/home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoF/images/7_0.jpg').convert("RGB")}
                ]
            }
        ]

        inputs = vllm_mllm_process_batch_from_messages([conversation], processor)
        output = vllm_generate(inputs, sampling_params, mllm)
        print(output[0].outputs[0].text)

    elif mode == 'hf':
        mllm, processor = hf_mllm_init(model_path)
        inputs = hf_process_batch_data(
            text_prompts=["Describe this image in detail.", "Describe this image in detail."],
            image_paths=['/home/dids/shiyang/codes/abstract-visual-token/asset/pipeline.png', '/home/dids/shiyang/codes/abstract-visual-token/asset/pipeline.png'],
            mllm_dir=model_path,
            processor=processor,
            device="cuda:0"
        )
        output = hf_generate(mllm, processor, inputs)
        print(output[0])


if __name__ == '__main__':
    # 必须放在主入口，避免 multiprocessing spawn 子进程导入本模块时再次执行推理代码
    main()

'''
In Stage 2, the model further refines its understanding by incorporating more detailed information. It now takes [Question], [Image], [Thoughts], [Latent], and [Answer] as inputs. The model generates embeddings for thoughts and answers, similar to Stage 1. However, in addition to CE and L2 losses, there is a "generate" step which likely involves generating new content or improving existing predictions. The overall goal is to improve the accuracy of generating answers by leveraging both textual and visual information.
'''