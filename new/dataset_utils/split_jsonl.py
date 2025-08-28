#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

'''
cd /home/dids/shiyang/codes/abstract-visual-token/new/dataset_utils
python split_jsonl.py /home/dids/shiyang/codes/abstract-visual-token/new/created_dataset/filtered_data/Zebra_CoT_visual_search/stage1_policy_out.jsonl -n 4

cd /mmu_vcg_ssd/shiyang06/Project/Latent_Think/abstract-visual-token/new/dataset_utils
python split_jsonl.py /ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_visual_search/stage1_policy_out.jsonl -n 4

'''

def split_jsonl(input_path: str, num_parts: int = 4) -> list[str]:
    in_path = Path(input_path)
    assert in_path.exists(), f"Input file not found: {in_path}"
    assert in_path.is_file(), f"Not a file: {in_path}"

    # Read total line count first for contiguous even split
    total = 0
    with in_path.open('r', encoding='utf-8') as f:
        for _ in f:
            total += 1

    if total == 0:
        # Still create empty shards
        base = 0
        rem = 0
    else:
        base = total // num_parts
        rem = total % num_parts

    # Prepare output files
    dir_ = in_path.parent
    stem = in_path.stem  # for 'x.jsonl' this is 'x'
    ext = ''.join(in_path.suffixes) or '.jsonl'
    out_paths: list[Path] = []
    for i in range(num_parts):
        out_paths.append(dir_ / f"{stem}_{i+1}{ext}")

    # Re-read and write contiguous chunks
    counts = [base + (1 if i < rem else 0) for i in range(num_parts)]
    with in_path.open('r', encoding='utf-8') as fin:
        for i, out_p in enumerate(out_paths):
            need = counts[i]
            with out_p.open('w', encoding='utf-8') as fout:
                for _ in range(need):
                    line = fin.readline()
                    if not line:
                        break
                    fout.write(line)

    return [str(p) for p in out_paths]


def main():
    parser = argparse.ArgumentParser(description='Split a JSONL file into N contiguous parts.')
    parser.add_argument('input', help='Path to the JSONL file')
    parser.add_argument('-n', '--num-parts', type=int, default=4, help='Number of parts (default: 4)')
    args = parser.parse_args()

    out_files = split_jsonl(args.input, args.num_parts)
    print('Wrote shards:')
    for p in out_files:
        print(p)


if __name__ == '__main__':
    main()
