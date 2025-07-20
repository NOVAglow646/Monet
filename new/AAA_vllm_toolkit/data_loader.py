import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, BatchSampler, SequentialSampler
from datasets import load_dataset


class MathVistaDataset(Dataset):
    """MathVista 数据集（testmini split）。"""
    def __init__(
        self,
        root: str,
        split: str = "testmini",
        cache_dir: Optional[str] = None,
        total_num: Optional[int] = None,  # ← 新增
    ):
        self.ds = load_dataset(root, cache_dir=cache_dir)[split]

        # 只取前 total_num 条
        if total_num is not None:
            self.ds = self.ds.select(range(min(total_num, len(self.ds))))

        self.root = root

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample["image"] = os.path.join(self.root, sample["image"])
        return sample


class MathVerseDataset(Dataset):
    """MathVerse 数据集（testmini.parquet）。"""
    def __init__(
        self,
        root: str,
        split: str = "testmini",
        cache_dir: Optional[str] = None,
        total_num: Optional[int] = None,  # ← 新增
    ):
        data_files = {split: os.path.join(root, f"{split}.parquet")}
        self.ds = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)[split]

        if total_num is not None:
            self.ds = self.ds.select(range(min(total_num, len(self.ds))))

        self.root = root

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img_path = sample["image"]["path"]
        sample["image"] = os.path.join(self.root, Path("images") / img_path)
        sample["query"] = sample["query_cot"] + ". Put the final answer in \\boxed{}."
        sample["choices"] = self.__getchoices__(sample["query"])
        return sample
    
    def __getchoices__(self, question: str):
        s = question.find("\nChoices:")
        e = question.find(". Put the final answer in")
        if s != -1 and e != -1:
            choices_str = question[s+len("\nChoices:"):e]
            return [c[2:] for c in choices_str.strip().split("\n") ]
        return None


class GSM8KDataset(Dataset):
    """GSM8K 多模态扩展版（testmini.parquet）。"""
    def __init__(
        self,
        root: str,
        split: str = "testmini",
        cache_dir: Optional[str] = None,
        total_num: Optional[int] = None,  # ← 新增
    ):
        data_files = {split: os.path.join(root, f"{split}.parquet")}
        self.ds = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)[split]

        if total_num is not None:
            self.ds = self.ds.select(range(min(total_num, len(self.ds))))

        self.root = root

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample["image"] = os.path.join(self.root, sample["image"])
        return sample


def build_multimodal_dataset(
    dataset_path: str,
    cache_dir: Optional[str] = None,
    split: str = "testmini",
    total_num: Optional[int] = None,  # ← 新增
) -> Dataset:
    lower = dataset_path.lower()
    if "mathvista" in lower:
        return MathVistaDataset(dataset_path, split, cache_dir, total_num=total_num)
    if "mathverse" in lower:
        return MathVerseDataset(dataset_path, split, cache_dir, total_num=total_num)
    if "gsm8k" in lower:
        return GSM8KDataset(dataset_path, split, cache_dir, total_num=total_num)
    raise ValueError(f"Unrecognized dataset type from path: {dataset_path}")


def collate_fn(batch):
    questions = [item["query"] for item in batch]
    images = [item["image"] for item in batch]
    gts = [item["answer"] for item in batch]
    gt_choices = [item.get("choices", None) for item in batch]
    return questions, images, gts, gt_choices

def build_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=len(dataset) if batch_size==-1 else batch_size, drop_last=drop_last)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler
    )