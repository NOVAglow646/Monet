from load_and_gen_hf import *
from load_and_gen_vllm import *
from extract_and_check import *
from data_loader import *


if __name__ == "__main__":
    
    mllm_dir = "/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct"
    test_dataset_path = "/data1/qxwang/datasets/multimodal/MathVista"
    cache_dir = "/data1/qxwang/cache2"
    split = "testmini"
    judge_llm_dir = "/data1/qxwang/checkpoints/Qwen2.5-7B-Instruct"
    bsz = 8  # all data
    total_num = None  # Set to None to use all data
    mllm_device = "cuda:8"
    

    mllm, processor = hf_mllm_init(mllm_dir, device=mllm_device)
    judge_llm, _ = vllm_llm_init(judge_llm_dir, device="cuda:9")
    eval_dataset = build_multimodal_dataset(test_dataset_path, cache_dir=cache_dir, split=split, total_num=total_num)
    data_loader = build_data_loader(eval_dataset, batch_size=bsz, shuffle=False, num_workers=4, drop_last=False)

    total_results = []
    for questions, images, gts in data_loader:
        hf_inputs = hf_process_batch_data(
            questions,
            images,
            mllm_dir,
            processor,
            device=mllm_device
        )
        
        response_strs = hf_generate(
            model=mllm,
            processor=processor,
            inputs=hf_inputs,
            max_new_tokens=1024
        )
        
        preds = batch_extract_answer(response_strs)
        test_results = batch_judge(
            preds=preds,
            gts=gts,
            questions=questions,
            llm=judge_llm
        )
        total_results.extend(test_results)
    print(f"Test Accuracy: {sum(total_results) / len(total_results):.4f}")