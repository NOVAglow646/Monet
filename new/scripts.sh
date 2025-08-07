conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.filter_data --devices 4,5,6,7 --dataset_name PixelReasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.filter_data --devices 3,4 --dataset_name CoM


# CoF
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name CoF \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/CoF/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoF/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/CoF/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/CoF/filtered_train.json \
  --api_model_name deepseek-chat






# PixelReasoner
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name PixelReasoner \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/PixelReasoner/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/PixelReasoner/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/PixelReasoner/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/PixelReasoner/filtered_train.json \
  --api_model_name deepseek-chat




# CoM_w_MathVista
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name CoM_w_MathVista \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/CoM_w_MathVista/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoM_w_MathVista/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/CoM_w_MathVista/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json \
  --api_model_name deepseek-chat



# ReFocus
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name ReFocus \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/ReFocus/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/ReFocus/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192
  #--max-samples 200

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/ReFocus/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/ReFocus/filtered_train.json \
  --api_model_name deepseek-chat





# Visual_CoT v7w
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Visual_CoT_v7w \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 0,1,2,3 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct 

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Visual_CoT_v7w/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Visual_CoT_v7w/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --devices 0,1,2,3 \
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
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_visual_search \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3
  #--limit 200

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 8192
  #--max-samples 200 

# Step-1: strong MLLM inference
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2_infer \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_inferred.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --max-batch 8192 \
  --max_samples 30

# Step-2: judge answers
python -m dataset_utils.stage2_judge \
  --infer-file ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_inferred.jsonl \
  --stage1     ./created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out.jsonl \
  --out        ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --judge_mode data_spec \
  --batch 8192



# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json \
  --api_model_name deepseek-chat

# stage 3 new
python -m dataset_utils.stage3_new \
  --stage2     ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage2_strong_out_test.jsonl \
  --stage1     ./created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out.jsonl \
  --out-json   ./created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_new.json \
  --api_model_name deepseek-chat


# Zebra_CoT maze
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_maze \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3
  #--limit 200

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 8192
  #--max-samples 200 

# Step-1: strong MLLM inference
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2_infer \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_inferred.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --max-batch 8192 \
  --max_samples 30

# Step-2: judge answers
python -m dataset_utils.stage2_judge \
  --infer-file ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_inferred.jsonl \
  --stage1     ./created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out.jsonl \
  --out        ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --use_llm_to_judge


# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_maze/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json 





# Zebra_CoT geometry
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_geometry \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3
  #--limit 200

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_geometry/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_geometry/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 8192
  #--max-samples 200 

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_geometry/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json \
  --api_model_name deepseek-chat




# Zebra_CoT physics
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_physics \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3
  #--limit 200

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_physics/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 8192
  #--max-samples 200 

# stage 2 new

# Step-1: strong MLLM inference
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2_infer \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_physics/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_inferred.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --max-batch 8192 \
  --max_samples 10

# Step-2: judge answers
python -m dataset_utils.stage2_judge \
  --infer-file ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_inferred.jsonl \
  --stage1     ./created_dataset/filtered_data/Zebra_CoT_physics/stage1_policy_out.jsonl \
  --out        ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_strong_out.jsonl \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --use_llm_to_judge



# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_physics/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_physics/filtered_train.json 



# Zebra_CoT count
# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name Zebra_CoT_count \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 4 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --devices 0,1,2,3
  #--limit 200

# stage 2
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2 \
  --stage1 ./created_dataset/filtered_data/Zebra_CoT_count/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/Zebra_CoT_count/stage2_strong_out.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct \
  --judge_llm_tensor_parallel_size 4 \
  --max-batch 8192
  #--max-samples 200 

# stage 3
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3 \
  --stage2 ./created_dataset/filtered_data/Zebra_CoT_count/stage2_strong_out.jsonl \
  --llm-path "" \
  --out-json ./created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json \
  --api_model_name deepseek-chat








# stage 1
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage1 \
  --dataset-name VTS_2 \
  --policy-model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --policy_mllm_tensor_parallel_size 2 \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct  \
  --judge_llm_tensor_parallel_size 2 \
  --devices 0,1 \
  --judge_mode llm data_spec 
  #--limit 200


# Step-1: strong MLLM inference
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2_infer \
  --stage1 ./created_dataset/filtered_data/VTS_1/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/VTS_1/stage2_inferred.jsonl \
  --model-path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --strong_mllm_tensor_parallel_size 4 \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --max-batch 8192

# Step-2: judge answers
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage2_judge \
  --infer-file ./created_dataset/filtered_data/VTS_1/stage2_inferred.jsonl \
  --stage1     ./created_dataset/filtered_data/VTS_1/stage1_policy_out.jsonl \
  --out        ./created_dataset/filtered_data/VTS_1/stage2_strong_out.jsonl \
  --judge_mode llm data_spec \
  --judge_llm_dir /home/dids/shiyang/checkpoints/Qwen2.5-72B-Instruct\
  --judge_llm_tensor_parallel_size 2 \
  --devices 0,1 \
  --use_llm_to_extract_answer \
  --batch 8192

# stage 3 new
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token/new
python -m dataset_utils.stage3_new \
  --stage2     ./created_dataset/filtered_data/VTS_1/stage2_strong_out.jsonl \
  --stage1     ./created_dataset/filtered_data/VTS_1/stage1_policy_out.jsonl \
  --out-json   ./created_dataset/filtered_data/VTS_1/filtered_train_rest.json \
  --api_model_name deepseek-chat