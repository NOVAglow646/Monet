export NCCL_TOPO_FILE=/share/baidu-nccl/topo_a800_hpc_bcc.xml
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
source /pfs/wangzihao11/miniconda3/bin/activate
conda activate mirage
cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_rpm \
    --limit 5 \
    --out-root /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual \
    --api_model_name deepseek-chat
