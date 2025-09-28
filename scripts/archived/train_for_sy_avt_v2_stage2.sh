export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9


LR=0.00001
LATENT_SIZE=10
CE_EMPHASIZE_FACTOR=1.0
ALIGNMENT_WEIGHT=2.0
LOAD_CKPT=08_26-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor${CE_EMPHASIZE_FACTOR}
SAVE_CKPT=08_29-avt_v2_stage2-lr${LR}-latent${LATENT_SIZE}-ce_factor${CE_EMPHASIZE_FACTOR}-align_wt${ALIGNMENT_WEIGHT}

source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v2_stage2" \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoF/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/PixelReasoner/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/VTS_1/filtered_train_short3000_w_metadata.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v2_stage1/${LOAD_CKPT} \
  --save_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v2_stage2/${SAVE_CKPT} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --teacher_latent_dir /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/precomputed_teacher_latents/${LOAD_CKPT} \
  --alignment_layer all_layers
  
  