conda activate monet
cd Monet

# SFT stage1
CE_EMPHASIZE_FACTOR=2.0
SAVE_CKPT=sft_stage1_ce${CE_EMPHASIZE_FACTOR}
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage1" \
  --data_path \
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/CoM_w_MathVista/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json" \
  --log_file "./log.txt" \
  --load_model_path path_to_your_model/Qwen2.5-VL-7B-Instruct \
  --save_model_path path_to_your_model/${SAVE_CKPT} \
  --dataset_root path_to_your_dataset/Monet-SFT-125K \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}

  