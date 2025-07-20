from load_and_gen_vllm import *
from extract_and_check import *
from data_loader import *
import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import json
import argparse
parser=argparse.ArgumentParser(description="Evaluate VLLM model on MathVerse dataset.")
parser.add_argument("--mllm_dir", type=str, default="/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--use_ds_api", action="store_true")
parser.add_argument("--use_judge_llm", action="store_true")
parser.add_argument("--devices", type=str, default="2,3")
args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
os.environ["NCCL_P2P_DISABLE"] = "1"
ours_inst = r" Remenber: 1. Solve the problem step by step. Use '### Step xxx:' to start a step. 2. You should output 'The final answer is: \boxed{...}' after the reasoning steps, and put the final answer in \boxed{}."

saved_response_path = None
saved_extracted_path = None
mllm_dir = "/data1/qxwang/codes/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo_mc2_prbsz512_oribsz128_tsim0.95_tcrtcls0.25_tacc0.6_easyr1judgeV1_0.1formatRWDV3_0.001lenPenalty_ablPRfalse/global_step_105/actor/huggingface"
mllm_dir = "/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct"
mllm_dir = args.mllm_dir
cache_dir = "/data1/qxwang/cache2"

judge_llm_dir = "/data0/huggingface/Qwen/Qwen2.5-14B-Instruct"
bsz = -1 # all data

dataset_name = "MathVerse"
dataset_mapping = {
    "MathVista": {
        "path": "/data1/qxwang/datasets/multimodal/MathVista",
        "split": "testmini"
    },
    "MathVerse": {
        "path": "/data1/qxwang/datasets/multimodal/MathVerse",
        "split": "testmini"
    }
}

test_dataset_path = dataset_mapping[dataset_name]["path"]
split = dataset_mapping[dataset_name]["split"]


read_from_saved = False
total_num = None # Set to None to use all data
use_ds_api = args.use_ds_api
use_judge_llm = args.use_judge_llm
client = None
ds_api_args = {}
judge_llm = None


if use_ds_api:
    dotenv_file = find_dotenv()
    load_dotenv(dotenv_file)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    ds_api_args = {
        "client": client
    }

if not read_from_saved:
    mllm, sampling_params = vllm_mllm_init(mllm_dir)
if use_judge_llm:
    judge_llm, _ = vllm_llm_init(judge_llm_dir)
eval_dataset = build_multimodal_dataset(test_dataset_path, cache_dir=cache_dir, split=split, total_num=total_num)
data_loader = build_data_loader(eval_dataset, batch_size=bsz, shuffle=False, num_workers=4, drop_last=False)

all_gts = []
all_questions = []
all_gt_choices = []
if not read_from_saved:
    vllm_inputs = []    
    for questions, images, gts, gt_choices in data_loader:
        #if "mc2" in mllm_dir:
        #    questions = [q + ours_inst for q in questions]
        
        vllm_inputs.extend(vllm_mllm_process_batch_official(
            questions,
            images,
            mllm_dir
        ))
        all_gts.extend(gts)
        all_questions.extend(questions)
        all_gt_choices.extend(gt_choices)

    completions = mllm.generate(
        vllm_inputs,
        sampling_params=sampling_params,
        use_tqdm=True
    )
    response_strs = [completion.outputs[0].text.strip() for completion in completions]
else:
    for questions, images, gts, gt_choices in data_loader:
        all_gts.extend(gts)
        all_questions.extend(questions)
        all_gt_choices.extend(gt_choices)
    with open(saved_response_path, 'r') as f:
        response_strs = json.load(f)
    with open(saved_extracted_path, 'r') as f:
        extracted_ans = json.load(f)


if use_ds_api:
    ds_api_args["all_questions"] = all_questions
    ds_api_args["all_gts"] = all_gts
    ds_api_args["all_gt_choices"] = all_gt_choices
preds = batch_extract_answer(response_strs, use_ds_api=use_ds_api, ds_api_args=ds_api_args)

test_results = batch_judge(
    preds=preds,
    gts=all_gts,
    gt_choicess=all_gt_choices,
    questions=all_questions,
    llm=judge_llm,
    use_ds_api=use_ds_api,
    ds_api_args=ds_api_args
)
if use_ds_api:
    saved_results = [{
        "question": question,
        "gt": gt,
        "extracted_answer": pred,
        "raw_response": response,
        "correct": test_results[i]
    } for i, (question, gt, pred, response) in enumerate(zip(all_questions, all_gts, preds, response_strs))]

    with open('saved_model_output/Qwen2.5-VL-7B-Instruct_MathVista_extracted-gt-judge_w_api.json', 'w') as f:
        json.dump(saved_results , f, indent=2, ensure_ascii=False)

print(f"Test Accuracy: {sum(test_results) / len(test_results):.4f}")