#####################################################################
# AVT v2 stage1
#####################################################################
proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
LATENT_SIZE=16
ATTN_LOSS_WEIGHT=100.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce3.0_detach_attn-loss${ATTN_LOSS_WEIGHT}_total8
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor 3.0 \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --observation_tokens_cannot_see_question_image \
    --use_emphasize_latent_attn_loss \
    --emphasize_latent_attn_coef ${ATTN_LOSS_WEIGHT} \
    --attn_loss_layers 1 5 10 15 20 25 26 27


proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=1
LATENT_SIZE=16
ATTN_LOSS_WEIGHT=5.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce3.0_detach_attn-loss${ATTN_LOSS_WEIGHT}_mask-latent
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor 3.0 \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --observation_tokens_cannot_see_question_image \
    --use_emphasize_latent_attn_loss \
    --emphasize_latent_attn_coef ${ATTN_LOSS_WEIGHT} \
    --attn_loss_layers 10 20 26 27 \
    --mask_latent



proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=1
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=1.0
CKPT=9.6_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_mask-q-img_lat-see-pre
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --mask_question_image \
    --latent_can_see_all_previous

proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=2
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=10.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_mask-q-img_align-vis-lat-pool-${ALIGN_VISION_LATENT_LOSS_WEIGHT}
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --mask_question_image \
    --use_align_vision_latent_loss_pooling \
    --align_vision_latent_loss_weight ${ALIGN_VISION_LATENT_LOSS_WEIGHT}


proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=3
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=0.0001
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_mask-q-img_align-vis-lat-proj-${ALIGN_VISION_LATENT_LOSS_WEIGHT}
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --mask_question_image \
    --use_align_vision_latent_loss_projector \
    --align_vision_latent_loss_weight ${ALIGN_VISION_LATENT_LOSS_WEIGHT}

proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_mask-q-img_mask-latent
python -m src.main  \
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
    --load_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/9.1_ablation_avt_v2_stage1_latent24_ce5.0_mask-q-img_mask-latent/checkpoint-200" \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --mask_latent \
    --mask_question_image \
    --resume_from_checkpoint

proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=2
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_mask-q-img_not-mask-image
python -m src.main  \
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
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --wandb_name ${CKPT} \
    --not_mask_image





proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=2
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_pt_obs-see-qtxt-lat_mask-latent
python -m src.main  \
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
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-0812-avt_sft-shuffle" \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --observation_tokens_only_see_question_and_latent \
    --wandb_name ${CKPT} \
    --mask_latent


proxy_on
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=3
LATENT_SIZE=16
CE_EMPHASIZE_FACTOR=3.0
CKPT=9.1_ablation_avt_v2_stage1_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_pt_not-mask-img_attn-analysis
python -m src.main  \
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
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct-avt_sft-shuffle-obs-ce-factor-2.0" \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}\
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage1/${CKPT}" \
    --not_mask_image \
    --wandb_name ${CKPT} \
    --attn_analysis

#####################################################################
# AVT v2 stage2
#####################################################################
proxy_on
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGNMENT_WEIGHT=10.0
ALIGNMENT_LAYER=all_layers
LOAD_CKPT=9.1_ablation_avt_v2_stage1_latent24_ce5.0_mask-q-img_align-vis-lat-proj-0.0001/checkpoint-2300
SAVE_CKPT=9.9_ablation_avt_v2_stage2_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGNMENT_WEIGHT}_${ALIGNMENT_LAYER}
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 1 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage2" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/avt_v2_stage1/${LOAD_CKPT} \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
    --alignment_weight ${ALIGNMENT_WEIGHT} \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --teacher_latent_dir ./new/precomputed_teacher_latents/${LOAD_CKPT} \
    --alignment_layer ${ALIGNMENT_LAYER} \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage2/${SAVE_CKPT}" \
    --wandb_name ${SAVE_CKPT}
    

proxy_on
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGNMENT_WEIGHT=2.0
ALIGNMENT_LAYER=all_layers
LOAD_CKPT=9.1_ablation_avt_v2_stage1_latent24_ce5.0_mask-q-img_align-vis-lat-proj-0.0001/checkpoint-2300
SAVE_CKPT=9.9_ablation_avt_v2_stage2_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGNMENT_WEIGHT}_${ALIGNMENT_LAYER}_mask-latent
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 1 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v2_stage2" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/avt_v2_stage1/${LOAD_CKPT} \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
    --alignment_weight ${ALIGNMENT_WEIGHT} \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --teacher_latent_dir ./new/precomputed_teacher_latents/${LOAD_CKPT} \
    --alignment_layer ${ALIGNMENT_LAYER} \
    --mask_latent \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v2_stage2/${SAVE_CKPT}" \
    --wandb_name ${SAVE_CKPT}
    
#####################################################################
# AVT v3
#####################################################################
proxy_on
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
LATENT_SIZE=24
CE_EMPHASIZE_FACTOR=5.0
ALIGN_VISION_LATENT_LOSS_WEIGHT=0.0001
LOAD_CKPT=Qwen2.5-VL-7B-Instruct
SAVE_CKPT=9.9_ablation_avt_v3_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_align-wt${ALIGN_VISION_LATENT_LOSS_WEIGHT}
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
    --epochs 5 \
    --bsz 1 \
    --grad_accum_steps 16 \
    --task "mm-reasoning" \
    --stage "avt_v3" \
    --data_path \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_9.1.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_9.1.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata_9.1.json" \
    --log_file "./log.txt" \
    --load_model_path /home/dids/shiyang/checkpoints/${LOAD_CKPT} \
    --save_model_path "/home/dids/shiyang/checkpoints/avt_v3/${SAVE_CKPT}" \
    --wandb_name ${SAVE_CKPT} \
    --deepspeed ./deepspeed/ds_zero2_gpu.json \
    --latent_size ${LATENT_SIZE} \
    --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
    --align_vision_latent_loss_weight ${ALIGN_VISION_LATENT_LOSS_WEIGHT} \
    --use_align_vision_latent_loss_pooling
    
    