export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# avt v4 training
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=10.1_tiny_avt_v4_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGNMENT_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 5 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v4" \
  --data_path "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1_max_seq_len2500.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed_max_seq_len3000.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000_max_seq_len2750.json" \
  --log_file "./log.txt" \
  --load_model_path /home/dids/shiyang/checkpoints/after9.28/9.27_avt_sft_full_ce2.0/9.27_avt_sft_full_ce2.0 \
  --save_model_path /home/dids/shiyang/checkpoints/avt_v4/${SAVE_CKPT} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --emphasize_latent_weight ${EMPHASIZE_LATENT_WEIGHT} \
  --alignment_layer all_layers
