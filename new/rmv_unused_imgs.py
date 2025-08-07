import json
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Remove unused images from dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
    return parser.parse_args()
args = parse_args()
dataset_name = args.dataset_name
path = f'created_dataset/filtered_data/{dataset_name}/filtered_train.json'
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

imgs_to_keep = []
for lst in data:
    for item in lst:
        for c in item['content']:
            if c['type'] == 'image':
                imgs_to_keep.append(c['image_file_name']) 
                    
# Get all images in the ./images directory
images_dir = f'created_dataset/filtered_data/{dataset_name}/images'
if os.path.exists(images_dir):
    all_images = os.listdir(images_dir)
    
    # Delete images not in imgs_to_keep
    for img_file in all_images[:]:
        #print(img_file)
        if f"created_dataset/filtered_data/{dataset_name}/images/{img_file}" not in imgs_to_keep:
            img_path = os.path.join(images_dir, img_file)
            os.remove(img_path)
            print(f"Deleted: {img_file}")
