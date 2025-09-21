export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=5.0
ALIGNMENT_WEIGHT=1.0
EMPHASIZE_LATENT_WEIGHT=1.0
SAVE_CKPT=9.21_debug_avt_v4_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGNMENT_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 5 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v4" \
  --data_path "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  --log_file "./log.txt" \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --save_model_path /home/dids/shiyang/checkpoints/avt_v4 \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --emphasize_latent_weight ${EMPHASIZE_LATENT_WEIGHT} \
  --teacher_reps_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_reps/Qwen2.5-VL-7B-Instruct \
  --alignment_layer all_layers

  