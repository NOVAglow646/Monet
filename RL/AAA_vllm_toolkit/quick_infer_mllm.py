model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-0812-avt_sft-shuffle'
model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct'

model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-avt_sft-shuffle-obs-ce-factor-2.0'
model_path = '/home/dids/shiyang/checkpoints/08_15-avt_stage1-6-30-40-wt1.0-ep2'
model_path = '/home/dids/shiyang/codes/abstract-visual-token/checkpoints/avt_stage1-observation_all-ep2-bsz1-lr1e-05-6-30-40-wt2.0/checkpoint-482'
model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-08_15-avt_stage1-6-30-40-wt1.0-ep5'
model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-08_15-avt_stage1-6-30-40-wt2.0-ep5'
model_path = '/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-08_15-avt_stage1-6-30-40-wt1.0-ep2'
model_path = '/mmu_vcg_ssd/shiyang06-temp/Latent_Think/checkpoint/avt_v5/10.16_v5_s2_rw_pt-sft3884_s1-pt-sft3884_teacher-al2.0-1000_latent8_ce4.0_align2.0/checkpoint-1000'
import PIL.Image
#import new.avt_qwen_model.vllm.apply_qwen2_5_avt_gpu_model_runner
from AAA_vllm_toolkit.load_and_gen_vllm import *
from AAA_vllm_toolkit.load_and_gen_hf import *
import os
import PIL



def main():
    mode = 'vllm'
    if mode == 'vllm':
        mllm, sampling_params = vllm_mllm_init(model_path, tp=1, gpu_memory_utilization=0.55)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # inputs = vllm_mllm_process_single_data("Describe this image in detail.", image_path='/data1/qxwang/codes/Mirage/new/debug_1.jpg', mllm_dir=model_path)
        os.environ['ABS_VIS_START_ID'] = str(processor.tokenizer.encode('<|vision_start|>')[0])
        os.environ['ABS_VIS_END_ID'] = str(processor.tokenizer.encode('<|vision_end|>')[0])
        
        dataset_root = "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/"
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image. \nPut your final answer within \\boxed{}."},
                        {"type": "image", "image": PIL.Image.open('/mmu_vcg_ssd/shiyang06-temp/Latent_Think/Easyr1-temp/debug_thyme_image.png').convert("RGB")}
                    ]
                }
            ]
        ]

        inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
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