export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
SAVE_CKPT=10.1_tiny_wo_maze_ce2.0
torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_reps \
    --bsz 1 \
    --task "mm-reasoning" \
    --data_path \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1_max_seq_len2500.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed_max_seq_len3000.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/after9.28/9.27_avt_sft_full_ce2.0/9.27_avt_sft_full_ce2.0 \
    --save_model_path ./new/precomputed_teacher_reps/${SAVE_CKPT} \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --output_hidden_states
