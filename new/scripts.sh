conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
export NCCL_P2P_DISABLE=1
python -m dataset_utils.filter_data --devices 4,5,6,7 --dataset_name PixelReasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
export NCCL_P2P_DISABLE=1
python -m dataset_utils.filter_data --devices 3,4 --dataset_name CoM


# CoF
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name CoF \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 4,5,6,7 

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/CoF/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoF/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/CoF/stage2_strong_out.jsonl \
  --llm-path "" \
  --devices 0,1,2,3 \
  --out-json ./created_dataset/filtered_data/CoF/filtered_train.json \
  --api_model_name deepseek-chat






# PixelReasoner
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name PixelReasoner \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 6,7,8,9

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/PixelReasoner/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/PixelReasoner/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 2,3,4,5 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/PixelReasoner/stage2_strong_out.jsonl \
  --llm-path "" \
  --devices 6,7,8,9 \
  --out-json ./created_dataset/filtered_data/PixelReasoner/filtered_train.json




# CoM_wo_MathVista
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name CoM_wo_MathVista \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 8,9

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/CoM_wo_MathVista/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoM_wo_MathVista/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/CoM_wo_MathVista/stage2_strong_out.jsonl \
  --llm-path "" \
  --devices 6,7,8,9 \
  --out-json ./created_dataset/filtered_data/CoM_wo_MathVista/filtered_train.json



# CoM_w_MathVista
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name CoM_w_MathVista \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 8,9

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/CoM_w_MathVista/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoM_w_MathVista/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/CoM_w_MathVista/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json 




# ReFocus
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name ReFocus \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 8,9 

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/ReFocus/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/ReFocus/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/ReFocus/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/ReFocus/filtered_train.json 




# Visual_CoT flicker30k
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Visual_CoT_flickr30k \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 6,7,8,9 \
  --judge_llm_dir /data0/huggingface/Qwen/Qwen2.5-32B-Instruct \
  --limit 20 

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Visual_CoT_flickr30k/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Visual_CoT_flickr30k/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Visual_CoT_flickr30k/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Visual_CoT_flickr30k/filtered_train.json 



# Visual_CoT v7w
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Visual_CoT_v7w \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 6,7,8,9 \
  --judge_llm_dir /data0/huggingface/Qwen/Qwen2.5-32B-Instruct 

# stage 2
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Visual_CoT_v7w/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Visual_CoT_v7w/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --devices 6,7,8,9 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Visual_CoT_v7w/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Visual_CoT_v7w/filtered_train.json 




# Zebra_CoT visual search
# stage 1
export NCCL_P2P_DISABLE=1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_visual_search \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 2 \
  --devices 5,6 
  #--limit 200

# stage 2
export NCCL_P2P_DISABLE=1
export RAY_CGRAPH_get_timeout=350
export RAY_CGRAPH_submit_timeout=350
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-32B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /data0/huggingface/Qwen/Qwen2.5-14B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 4096
  #--max-samples 200 

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json 




