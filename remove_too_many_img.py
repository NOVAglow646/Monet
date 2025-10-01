import json
import os
num = 3
data_path = "/home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_count/filtered_train_w_metadata_9.25_max_seq_len4096_max_seq_len3000.json"
with open(data_path, 'r') as f:
    data = json.load(f)

new_data = []
for sample in data:
    cot = sample['data']
    assistant_contents = cot[2]
    imt_cnt = 0
    for content in assistant_contents['content']:
        if 'img' in content:
            imt_cnt += 1
    
    if imt_cnt <= num:
        new_data.append(sample)

print(f"Original samples: {len(data)}, after removing too many images: {len(new_data)}")
save_path = data_path.replace(".json", f"_max_img{num}.json")
with open(save_path, 'w') as f:
    json.dump(new_data, f, indent=4)

'''

cd /home/dids/shiyang/codes/abstract-visual-token
conda activate mirage
python remove_too_many_img.py

'''