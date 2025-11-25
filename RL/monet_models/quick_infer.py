import argparse
import PIL.Image
#import new.avt_qwen_model.vllm.apply_qwen2_5_avt_gpu_model_runner
from avt.AAA_vllm_toolkit.load_and_gen_vllm import *
from avt.AAA_vllm_toolkit.load_and_gen_hf import *
import os
import PIL
import json
from datasets import Dataset
import os, time
from vllm import LLM, SamplingParams
from avt.vllm.latent_recorder import LatentRecorder
from tqdm import tqdm
import re
os.environ["AVT_LATENT_ALWAYS"] = "0"

def avt_single_input_images_preprocess_function_question_only(sample, dataset_root="", processor=None, max_seq_len=4096, cur_max=-1, id=0, rank=-1):
    """
    Preprocess function for AVT with single input images.
    """
    conversations = []

    # Process image loading for all steps first
    for i, step in enumerate(sample['data'][:2]):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        if step["role"] == "user":
            new_step["content"][1]["text"] = new_step["content"][1]["text"].replace("If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge.", "")
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                content["image"] = os.path.join(dataset_root,content.pop("image_file_name")) 
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        return None
            
            new_step["content"][j] = content
        conversations.append(new_step)

    return conversations

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)

def main():
    
    parser = argparse.ArgumentParser(description="Quick inference script for AVT model")
    parser.add_argument('--model_path', default='/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data_path', nargs='+', required=True, help='Path to the dataset files')
    parser.add_argument('--dataset_root', default='/home/dids/shiyang/codes/abstract-visual-token/new')
    parser.add_argument('--num_samples', default=10, type=int)
    args = parser.parse_args()

    # Start recorder BEFORE creating LLM so workers pick up AVT_LATENT_HOOK_UDP
    with LatentRecorder() as rec:
        mllm, sampling_params = vllm_mllm_init(args.model_path, tp=1, gpu_memory_utilization=0.8)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

        preprocess_function = avt_single_input_images_preprocess_function_question_only
        all_train_dataset = []
        for data_path in args.data_path:
            if data_path.endswith('.jsonl'):
                train_dataset = load_jsonl_dataset(data_path)
            elif data_path.endswith('.json'):
                train_dataset = load_json_dataset(data_path)
            all_train_dataset.extend(train_dataset[:])
        all_train_dataset = all_train_dataset[:args.num_samples]
        train_dataset = []
        cur_max = -1
        for i, sample in tqdm(enumerate(all_train_dataset), desc="Collecting training data and length check...", total=len(all_train_dataset)):
            processed = preprocess_function(sample, dataset_root=args.dataset_root, processor=processor, max_seq_len=6000, cur_max=cur_max, id=i, rank=0)
            if processed is not None:
                train_dataset.append(processed)


        inputs = vllm_mllm_process_batch_from_messages(train_dataset, processor)
        # Generate and then inspect accumulated latents
        output = vllm_generate(inputs, sampling_params, mllm)
        grouped = rec.get_grouped()
        # Compact summary: per parent, list sample indices and lengths
        summary = {
            pid: {sid: len(traj) for sid, traj in samples.items()}
            for pid, samples in grouped.items()
        }
        print("[AVT] grouped step-counts per parent/sample:", summary)
        cnt = 0
        '''for o in output:
            if "</abs_vis_token>" in o.outputs[0].text:
                cnt += 1
                print("###############################################")
        print(cnt)'''

        res = []
        for i, o in enumerate(output):
            res_b = []
            for rollouts in o.outputs:
                res_b.append(replace_abs_vis_token_content( rollouts.text))
            res.append(res_b)

        with open('/home/dids/shiyang/codes/Easyr1-temp/debug/output-300step.text', 'w') as f:
            for i in range(10):
                question = inputs[i]['prompt']
                f.write(f"Question: {question}")
                for j in range(4):
                    f.write(res[i][j])
                    f.write("\n\n")
                f.write("\n\n############################\n\n")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: What's the price of the egg sandwhich? \nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."},
                    {"type": "image", "image": PIL.Image.open('/home/dids/shiyang/codes/abstract-visual-token/new/vts_1.jpg').convert("RGB")}
                ]
            }
        ]

        inputs = vllm_mllm_process_batch_from_messages([conversation], processor)
        output = vllm_generate(inputs, sampling_params, mllm)
        print(output[0].outputs[0].text)
if __name__ == '__main__':
    # 必须放在主入口，避免 multiprocessing spawn 子进程导入本模块时再次执行推理代码
    main()

'''
In Stage 2, the model further refines its understanding by incorporating more detailed information. It now takes [Question], [Image], [Thoughts], [Latent], and [Answer] as inputs. The model generates embeddings for thoughts and answers, similar to Stage 1. However, in addition to CE and L2 losses, there is a "generate" step which likely involves generating new content or improving existing predictions. The overall goal is to improve the accuracy of generating answers by leveraging both textual and visual information.
'''