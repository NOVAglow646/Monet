
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=2.0
SAVE_CKPT=9.21_avt_sft_ce${CE_EMPHASIZE_FACTOR}
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --epochs 8 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
  --log_file "./log.txt" \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --save_model_path /home/dids/shiyang/checkpoints/avt_sft/${SAVE_CKPT} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}

  