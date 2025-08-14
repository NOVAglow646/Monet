# new, parallel
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --model "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train.json" \
  --log_file "./log.txt" \
  --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --sft_analysis_enable \
  --sft_analysis_ratio 0.1 \
  --sft_analysis_categories non_observation_poss observation_poss \
  --observation_ce_factor 3.0 \
  --observation_ce_warmup_steps 100
