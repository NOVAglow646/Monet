source /pfs/wangzihao11/miniconda3/bin/activate

# Zebra_CoT geometry
# stage 1
# export NCCL_P2P_DISABLE=1

conda activate mirage

cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
echo "Starting stage 1 for Zebra_CoT_geometry"
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_geometry \
  --policy-model-path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 2 \
  --devices 0,1,2,3,4,5,6,7 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4
  #--limit 200


# stage 2
# export NCCL_P2P_DISABLE=1
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
echo "Starting stage 2 for Zebra_CoT_geometry"
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_geometry/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_geometry/stage2_strong_out.jsonl \
  --model-path /pfs/Models/Qwen2.5-VL-32B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3,4,5,6,7 \
  --token-limit 8192 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 2048 

# stage 3
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
echo "Starting stage 3 for Zebra_CoT_geometry"
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_geometry/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json \
  --api_model_name gemini-2.5-pro 




# Zebra_CoT physics
# stage 1
# export NCCL_P2P_DISABLE=1
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_physics \
  --policy-model-path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 2 \
  --devices 0,1,2,3,4,5,6,7 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4
  #--limit 200

# stage 2
# export NCCL_P2P_DISABLE=1
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_physics/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_strong_out.jsonl \
  --model-path /pfs/Models/Qwen2.5-VL-32B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3,4,5,6,7 \
  --token-limit 8192 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 4096 
  #--max-samples 200 

# stage 3
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_physics/filtered_train.json \
  --api_model_name gemini-2.5-pro 




# Zebra_CoT maze
# stage 1
# export NCCL_P2P_DISABLE=1
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_maze \
  --policy-model-path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 2 \
  --devices 0,1,2,3,4,5,6,7 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4
  #--limit 200

# stage 2
# export NCCL_P2P_DISABLE=1
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --model-path /pfs/Models/Qwen2.5-VL-32B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3,4,5,6,7 \
  --token-limit 8192 \
  --judge_llm_dir /pfs/Models/Qwen2.5-32B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 4096
  #--max-samples 200 

# stage 3
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json \
  --api_model_name gemini-2.5-pro 