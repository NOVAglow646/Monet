import json
import argparse
from typing import List, Any, Dict

def merge_json_files(file_list: List[str], output_file: str, start_id: int = 0) -> None:
    """
    Merge multiple JSON files (each is a top-level list) in the given order.
    While merging, rewrite each element's metadata.sample_id to be sequential:
    start_id, start_id+1, ...
    """
    merged_data: List[Any] = []
    next_id = start_id

    for fpath in file_list:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{fpath} is not a JSON list!")

        print(f"Loaded {fpath}, items: {len(data)}")

        for item in data:
            # Ensure item is a dict to attach metadata; skip otherwise.
            if not isinstance(item, dict):
                raise ValueError(f"An element in {fpath} is not a JSON object (dict), cannot set metadata.sample_id.")

            # Ensure metadata exists and is a dict.
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                item["metadata"] = metadata

            # Overwrite sample_id with the new sequential id.
            metadata["sample_id"] = next_id
            next_id += 1

            merged_data.append(item)

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

    print(
        f"Merged {len(file_list)} files into {output_file}, "
        f"total items: {len(merged_data)}, "
        f"sample_id range: [{start_id}, {next_id - 1}]"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON list files in given order and rewrite metadata.sample_id sequentially."
    )
    parser.add_argument("files", nargs="+", help="JSON files to merge, in order")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    parser.add_argument("--start-id", type=int, default=0, help="Starting sample_id (default: 0)")
    args = parser.parse_args()

    merge_json_files(args.files, args.output, start_id=args.start_id)
