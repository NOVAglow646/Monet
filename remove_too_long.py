import os
import json
from src.utils import *
from src.utils import get_args
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import copy
def avt_single_input_images_preprocess_function(sample, dataset_root="", processor=None, max_seq_len=4096, cur_max=-1):
    """
    Preprocess function for AVT with single input images.
    """
    sample_copy = copy.deepcopy(sample)
    conversations = sample['data']
    dataset_name = conversations[1]['content'][0]['image_file_name'].split('/')[-3]
    
    # Process image loading for all steps first
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                content["image"] = os.path.join(dataset_root,content.pop("image_file_name")) 
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        return None, cur_max
            
            new_step["content"][j] = content
        conversations[i] = new_step
    
    texts = [processor.apply_chat_template(conversations, tokenize=False)]
    texts = [place_output_image_avt(text) for text in texts]
    image_inputs, _ = process_vision_info([conversations])
    image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    cur_max = max(cur_max, batch["input_ids"].shape[1])
    #if id%1000 == 0 and rank == 0:
    #    logging.info(f"Max_seq_len of all training data={cur_max}")
    if batch["input_ids"].shape[1] > max_seq_len:
        print("Found too long data:", dataset_name, batch["input_ids"].shape[1])
        return None, cur_max
    

    return sample_copy, cur_max

def filter_invalid_samples_in_json(json_path, dataset_root, processor, cur_max=-1, id=-1, max_seq_len=4096):
    save_name = None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered_data = []
    removed_count = 0
    dataset_name = os.path.basename(os.path.dirname(json_path))
    for idx, sample in tqdm(enumerate(data), desc=json_path, total=len(data)):
        result, cur_max = avt_single_input_images_preprocess_function(
            sample, dataset_root=dataset_root, processor=processor, max_seq_len=max_seq_len, cur_max=cur_max
        )
        if result is not None:
            filtered_data.append(result)
        else:
            #print(f"Removed sample from dataset: {dataset_name}")
            removed_count += 1
    if removed_count > 0:
        save_name = json_path.replace('.json',f'_max_seq_len{max_seq_len}.json')
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    return removed_count, cur_max, save_name

def main():
    cur_max=-1
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)
    # 遍历所有数据集文件夹
    total_id = 0
    for json_path in args.data_path:
        if os.path.isfile(json_path):
            removed_count, cur_max, save_name = filter_invalid_samples_in_json(json_path, args.dataset_root, processor, cur_max=cur_max, id=total_id, max_seq_len=args.max_seq_len)
            total_id+=1
            if removed_count > 0:
                print(f"Saved to {save_name}, removed {removed_count} samples due to exceeding max_seq_len {args.max_seq_len}.")


if __name__ == "__main__":
    main()

'''
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python remove_too_long.py \
    --max_seq_len 2048 \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --data_path \
    "/home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json"


source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
python remove_too_long.py \
    --max_seq_len 4096 \
    --load_model_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
    --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
    --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata_9.1.json" \

'''