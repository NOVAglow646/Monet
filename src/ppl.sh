conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=CoF
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=0
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
  --no_question_image



conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220


conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=CoM_w_MathVista
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=3
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
  --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220



conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=Zebra_CoT_visual_search
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=3
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
  --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220




conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=PixelReasoner
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=1
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
  --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220





conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=ReFocus
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=2
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
 --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220




conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=Zebra_CoT_geometry
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_w_metadata.json"
CUDA_VISIBLE_DEVICES=2
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
 --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220



conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token

# Input dataset path (single file). Adjust this line if you switch datasets.
DATASET_NAME=Zebra_CoT_maze
DATA_PATH="./new/created_dataset/filtered_data/${DATASET_NAME}/filtered_train_short3000_w_metadata.json"
CUDA_VISIBLE_DEVICES=2
# Save dir reflects dataset name
SAVE_DIR="./logs/ppl_analysis/${DATASET_NAME}"

python -m src.analyze_cot_image_removal \
  --load_model_path /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct \
  --data_path "$DATA_PATH" \
  --stage avt_v2_stage2 --task mm-reasoning \
  --save_model_path "$SAVE_DIR" \
  --num_samples 50 \
 --no_question_image

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m logs.plot_cot_image_removal \
  --jsonl ./logs/ppl_analysis/${DATASET_NAME}/cot_image_ablation.jsonl \
  --out ./logs/ppl_analysis/cot_image_removal_figs/${DATASET_NAME}/ \
  --tokenizer /home/dids/shiyang/checkpoints/Qwen2.5-VL-72B-Instruct/ \
  --cols 40 --cell_w 1.0 --cell_h 1.0 --dpi 220

