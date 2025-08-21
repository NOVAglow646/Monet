export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

LATENT_SIZE=6
CE_EMPHASIZE_FACTOR=1.0
ALIGNMENT_WEIGHT=1.0

source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v2_stage1" \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoF/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/PixelReasoner/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_short3000.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/VTS_1/filtered_train_short3000.json" \
  --log_file "./log.txt" \
  --load_model_path "/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct" \
  --save_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v2_stage1/08_20-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor${CE_EMPHASIZE_FACTOR}-align_wt${ALIGNMENT_WEIGHT} \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name 08_20-avt_v2_stage1-latent${LATENT_SIZE}-ce_factor${CE_EMPHASIZE_FACTOR}-align_wt${ALIGNMENT_WEIGHT}
  