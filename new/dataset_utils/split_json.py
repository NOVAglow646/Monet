'''
python split_json.py /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_maze/filtered_train_short3000_w_metadata.json 4

'''

import json
import math
import argparse

def split_json_file(src_file: str, n: int):
    # Load the JSON file
    with open(src_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate chunk size
    total = len(data)
    chunk_size = math.ceil(total / n)

    # Split and save
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size
        part = data[start:end]
        out_file = f"{src_file.rsplit('.', 1)[0]}_{i+1}.json"
        with open(out_file, "w", encoding="utf-8") as f_out:
            json.dump(part, f_out, ensure_ascii=False, indent=2)
        print(f"Saved {out_file}, items: {len(part)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSON list into n parts")
    parser.add_argument("json_path", type=str, help="Path to the source JSON file (outermost must be a list)")
    parser.add_argument("n", type=int, help="Number of parts to split into")
    args = parser.parse_args()

    split_json_file(args.json_path, args.n)
