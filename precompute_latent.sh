export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
CKPT=08_24-avt_v2_stage1-latent10-ce_factor1.0-step500
torchrun --nproc-per-node=4 --master-port=29501 -m src.precompute_teacher_latent \
    --bsz 1 \
    --task "mm-reasoning" \
    --stage "avt_v2_precompute_latent" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT} \
    --save_model_path ./new/precomputed_teacher_latents/${CKPT} \
    --latent_size 10 \
    --ce_emphasize_factor 1.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json