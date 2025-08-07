conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.main \
    --model "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --epochs "10" \
    --task "mm-reasoning" \
    --min_latent_size 6 \
    --min_latent_compress_factor 20 \
    --max_latent_compress_factor 40 \
    --stage "avt_stage1" \
    --data_path "./new/created_dataset/filtered_data/CoF/filtered_train.json" \
    "./new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train.json" \
    "./new/created_dataset/filtered_data/PixelReasoner/filtered_train.json" \
    "./new/created_dataset/filtered_data/ReFocus/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train.json" \
    "./new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train.json" \
    --log_file "./log.txt" \
    --load_model_path "/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --save_model_path "./checkpoints/model_stage1" \
    --alignment "observation_all"