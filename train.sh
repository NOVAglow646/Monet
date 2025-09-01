#####################################################################
# AVT v2 stage1
#####################################################################
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 3 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 3.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json


export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
CKPT=9.1_avt_v2_stage1_latent10_ce3.0_ablation
python -m src.main \
    --epochs 3 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 3.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --wandb_name ${CKPT} \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --observation_tokens_cannot_see_question_image

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
python -m src.main \
    --epochs 3 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 3.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --wandb_name "9.1_avt_v2_stage1_latent10_ce3.0_ablation"

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
python -m src.main \
    --epochs 3 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage1" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --latent_size 10 \
    --ce_emphasize_factor 3.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --wandb_name "9.1_avt_v2_stage1_latent10_ce3.0_ablation"


#####################################################################
# AVT v2 stage2
#####################################################################

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 15 \
    --lr 0.00001 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage2" \
    --data_path \
    "./new/created_dataset/filtered_data/CoF/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train_short3000_w_metadata.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/avt_v2_stage1/08_24-avt_v2_stage1-latent10-ce_factor1.0-step500 \
    --latent_size 10 \
    --ce_emphasize_factor 1.0 \
    --alignment_weight 2.0 \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --teacher_latent_dir /home/dids/shiyang/codes/abstract-visual-token/new/precomputed_teacher_latents/08_24-avt_v2_stage1-latent10-ce_factor1.0-step500
    