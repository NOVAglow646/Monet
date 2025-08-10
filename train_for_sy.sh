# new, parallel
conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc-per-node=4 --master-port=29501 -m src.main \
  --model "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --task "mm-reasoning" \
  --stage "avt_sft" \
  --data_path "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train.json" \
    "./new/created_dataset/filtered_data/VTS_1/filtered_train.json" \
  --log_file "./log.txt" \
  --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct"
