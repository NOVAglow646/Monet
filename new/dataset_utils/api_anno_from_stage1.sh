conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/stage1_policy_out.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_from_stage1.json \
 --api_model_name deepseek-chat \
 --max-records 3 \
 --batch-size 3


conda activate mirage
cd /home/dids/shiyang/codes/abstract-visual-token
python -m new.dataset_utils.api_anno_from_stage1 \
 --stage1 /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/stage1_policy_out.jsonl \
 --out-json /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/CoM_w_MathVista/filtered_train_w_metadata_from_stage1.json \
 --api_model_name deepseek-chat \
 --max-records 10 \
 --batch-size 3