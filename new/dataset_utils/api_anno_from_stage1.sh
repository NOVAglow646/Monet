conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/stage1_policy_out_1.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_w_metadata_from_stage1_1.json \
 --api_model_name deepseek-reasoner \
 --max-records 5

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_geometry/stage1_policy_out.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_geometry/filtered_train_w_metadata_from_stage1.json \
 --api_model_name deepseek-reasoner \
 --max-records 10




conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out_1.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_1.json \
 --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out_2.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_2.json \
 --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out_3.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_3.json \
 --api_model_name deepseek-reasoner

conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out_4.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/filtered_train_w_metadata_from_stage1_4.json \
 --api_model_name deepseek-reasoner
