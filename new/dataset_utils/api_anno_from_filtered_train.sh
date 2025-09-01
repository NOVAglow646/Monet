conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_short3000_w_metadata_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/ReFocus/filtered_train_w_metadata.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/ReFocus/filtered_train_short3000_w_metadata_9.1.json \
  --api_model_name deepseek-reasoner



conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_1.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_1_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_2.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_2_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_3.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_3_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_4.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_4_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_5.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_5_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_6.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_6_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_7.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_7_9.1.json \
  --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_8.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata_8_9.1.json \
  --api_model_name deepseek-reasoner


conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_filtered_train \
  --input-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata.json \
  --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_short3000_w_metadata_9.1.json \
  --api_model_name deepseek-reasoner \
  --max-record 2