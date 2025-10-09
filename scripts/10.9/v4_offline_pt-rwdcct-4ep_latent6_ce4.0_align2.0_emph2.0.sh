export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# precompute teacher representations
TEACHER=10.7_sft_rw-doc-chart_ce2.0


# avt v4 training
LATENT_SIZE=6
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=10.9_v4_offline_pt-rwdcct-4ep_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGNMENT_WEIGHT}_emph-wt${EMPHASIZE_LATENT_WEIGHT}
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --epochs 8 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_v4" \
  --data_path "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/CoM_w_MathVista/filtered_train_w_metadata_9.1_max_seq_len2500.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/ReFocus/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/filtered_train_w_metadata_9.24_further_washed_max_seq_len3000.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_geometry/filtered_train_w_metadata_9.1.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000_max_seq_len2500.json" \
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Visual_CoT/filtered_train_10.7.json" \
  --log_file "./log.txt" \
  --load_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_sft/${TEACHER} \
  --save_model_path /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/${SAVE_CKPT} \
  --dataset_root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name ${SAVE_CKPT} \
  --latent_size ${LATENT_SIZE} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --emphasize_latent_weight ${EMPHASIZE_LATENT_WEIGHT} \
  --teacher_reps_dir /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/precomputed_teacher_reps/${TEACHER} \
  --alignment_layer all_layers

ckpt_name=${SAVE_CKPT}
src=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/${ckpt_name}
output=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/${ckpt_name}.zip
include_subdir=False

if [ -z "$src" ] || [ -z "$output" ] || [ -z "$include_subdir" ]; then
    echo "用法: $0 <source_folder> <output_zip> <include_subdir:True|False>"
    exit 1
fi

if [ "$include_subdir" = "True" ]; then
    # 压缩整个目录（包含子目录），只保留最后一级目录名
    (cd "$(dirname "$src")" && zip -r "$output" "$(basename "$src")")
else
    # 只压缩目录下的文件，不包含子目录，保留最后一级目录名
    (cd "$(dirname "$src")" && zip "$output" "$(basename "$src")"/*)
fi

python /mmu_vcg_ssd/shiyang06/Tool/huggingface.py --item /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/${ckpt_name}.zip