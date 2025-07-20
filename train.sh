# Training stage 1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage1 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --load_model_path /data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct \
    --save_model_path ./checkpoints/model_stage1

# Training stage 2

python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage2 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/model_stage1
    --save_model_path ./checkpoints/model_stage2