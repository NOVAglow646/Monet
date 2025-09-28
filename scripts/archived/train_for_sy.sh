export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --shuffle_train \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoF/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/PixelReasoner/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_maze/filtered_train_new.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/VTS_1/filtered_train_new.json" \
  --log_file "./log.txt" \
  --load_model_path "/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct" \
  --save_model_path "/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_sft/08_14-shuffle-observation_ce_factor_3" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --sft_analysis_enable \
  --sft_analysis_ratio 0.1 \
  --sft_analysis_categories non_observation_poss observation_poss \
  --observation_ce_factor 3.0 \
  --observation_ce_warmup_steps 100 \
  --wandb_name 08_14-avt_sft
  