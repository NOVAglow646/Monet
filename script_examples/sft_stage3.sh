conda activate monet
cd Monet


# precompute target latent embeddings using the model trained in SFT Stage2
TEACHER_LATENT_SIZE=8
TEACHER_CE_EMPHASIZE_FACTOR=4.0
TEACHER_ALIGN_WEIGHT=2.0
TEACHER_EMPHASIZE_LATENT_WEIGHT=2.0
TEACHER=sft_stage2_latent${TEACHER_LATENT_SIZE}_ce${TEACHER_CE_EMPHASIZE_FACTOR}_al${TEACHER_ALIGN_WEIGHT}_emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}
torchrun --nproc-per-node=8 --master-port=29501 -m src.precompute_teacher_latents \
  --bsz 1 \
  --data_path \
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/CoM_w_MathVista/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json" \
  --log_file "./log.txt" \
  --load_model_path path_to_your_model/${TEACHER} \
  --save_model_path path_to_your_model/monet_precomputed_target_latent/${TEACHER} \
  --dataset_root path_to_your_dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size ${LATENT_SIZE} \
  --output_hidden_states \
  --resume



# SFT stage3
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=sft_stage3_target-latent${TEACHER_LATENT_SIZE}-al${TEACHER_ALIGN_WEIGHT}-emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}_student-latent${LATENT_SIZE}-ce${CE_EMPHASIZE_FACTOR}-al${ALIGNMENT_WEIGHT}-emph${EMPHASIZE_LATENT_WEIGHT}
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 3 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "avt_v5_stage2" \
  --data_path \
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/CoM_w_MathVista/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json" \
  --log_file "./log.txt" \
  --load_model_path path_to_your_model/sft_stage1_ce2.0 \
  --save_model_path path_to_your_model/${SAVE_CKPT} \
  --dataset_root path_to_your_dataset/Monet-SFT-125K \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --teacher_latent_dir path_to_your_model/monet_precomputed_target_latent/${TEACHER} \
  --alignment_layer all_layers