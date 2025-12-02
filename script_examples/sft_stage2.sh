conda activate monet
cd Monet


# STEP 1: precompute teacher representations of the observation tokens using the warm-up SFT model and save them to disk
TEACHER=sft_stage1_ce2.0
torchrun --nproc-per-node=8 --master-port=29505 -m src.precompute_teacher_reps \
  --bsz 1 \
  --data_path \
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/CogCoM/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json" \
  --load_model_path path_to_your_model/Monet_checkpoints/sft_stage1/${TEACHER} \
  --save_model_path path_to_your_model/Monet_checkpoints/monet_precomputed_observation_token_teacher_reps/${TEACHER} \
  --dataset_root path_to_your_dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --output_hidden_states \
  --alignment_layer all_layers





# STEP 2: SFT stage2 training
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=sft_stage2_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_al${ALIGNMENT_WEIGHT}_emph${EMPHASIZE_LATENT_WEIGHT}
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage2" \
  --data_path \
    "path_to_your_dataset/Monet-SFT-125K/Visual_CoT/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/CogCoM/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/ReFocus/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_count/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_visual_search/train.json" \
    "path_to_your_dataset/Monet-SFT-125K/Zebra_CoT_geometry/train.json" \
  --log_file "./log.txt" \
  --load_model_path path_to_your_model/Monet_checkpoints/sft_stage1/${TEACHER} \
  --save_model_path path_to_your_model/Monet_checkpoints/sft_stage2/${SAVE_CKPT} \
  --dataset_root path_to_your_dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --emphasize_latent_weight ${EMPHASIZE_LATENT_WEIGHT} \
  --teacher_reps_dir path_to_your_model/Monet_checkpoints/monet_precomputed_observation_token_teacher_reps/${TEACHER} \
  --alignment_layer all_layers
