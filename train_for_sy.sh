# SFT 

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m src.main \
    --model "xxx/Qwen2.5-VL-7B-Instruct" \
    --epochs "10" \
    --task "mm-reasoning" \
    --stage "avt_sft" \
    --bsz 2 \
    --grad_accum_steps 8 \
    --shuffle_train \
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
    --load_model_path "xxx/Qwen2.5-VL-7B-Instruct" \
    --sft_analysis_enable \
    --sft_analysis_ratio 0.1 \
    --sft_analysis_categories non_observation_poss observation_poss