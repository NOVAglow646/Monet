export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
python -m src.sft_eval_obs_acc_non_dist \
    --bsz 1 \
    --task "mm-reasoning" \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed.json" \
  "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
  "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000_max_seq_len2500.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/after9.28/10.1_avt_sft_wo_maze_ce2.0/10.1_avt_sft_wo_maze_ce2.0


