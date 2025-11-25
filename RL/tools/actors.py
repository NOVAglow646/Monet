# tools/actors.py
import torch
from transformers import AutoTokenizer, AutoModel
from tools.hash_dict import StepHashDict, SampleHashDict
import ray
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import psutil, os, gc
from sentence_transformers import SentenceTransformer
from vllm import LLM
@ray.remote(num_cpus=4, num_gpus=0)
class StepHashServer:
    def __init__(self, config):
        self.step_hash_dict = StepHashDict(similarity_threshold=config.rollout.mc.step_hash_threshold, correct_cluster_threshold=config.rollout.mc.correct_cluster_threshold, rep_mode="all")


    def update_sample_step_hash_dict(self, sample_id, steps, embeds, lead_to_correct_list):
        return self.step_hash_dict.update_sample_step_hash_dict(
            sample_id=sample_id,
            embeddings=embeds,
            texts=steps,
            lead_correct_list=lead_to_correct_list
        )
    
    def look_up_step_correctness(self,
        sample_id: int,
        texts: Union[str, List[str]]  # shape (N, D) 已 L2 归一化
    ) -> List[bool]:
        return self.step_hash_dict.look_up_step_correctness(
            sample_id=sample_id,
            texts=texts
        )
        
    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        return self.step_hash_dict.update_min_mean_correct_resp_len(
            sample_id=sample_id,
            resp_len=resp_len
        )
    
    def look_up_min_mean_correct_resp_len(self, sample_id: int):
        return self.step_hash_dict.look_up_min_mean_correct_resp_len(sample_id=sample_id)
    
    def get_step_dict_info(self, verbose_info: bool = False, print_info: bool = False):
        return self.step_hash_dict.get_step_dict_info(verbose_info, print_info)
    
    def get_rss(self):
        gc.collect()
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 2**30
        return rss_gb
    
    def save_info(self, filepath: str, overwrite: bool = True):
        return self.step_hash_dict.save_info(filepath, overwrite)
    
    def load_info(self, filepath: str):
        return self.step_hash_dict.load_info(filepath)     
    
    def ping(self):
        return

#@ray.remote(num_gpus=1, resources={"embed_gpu": 1}, runtime_env={"env": {"CUDA_VISIBLE_DEVICES": "9"}})
@ray.remote(num_gpus=1, resources={"embed_gpu": 1})
class EmbedServer:
    def __init__(self, model_path: str):
        #breakpoint()
        torch.cuda.set_device(0)            # 在本 Actor 里 0 就是runtime_env所指定的第一块gpu
        self.model = LLM(model=model_path, task="embed")  # or SentenceTransformer(...)
        #self.model.eval()

    @torch.no_grad()
    def encode(self, sentences, use_tqdm):
        return self.model.embed(sentences, use_tqdm=use_tqdm)
    
    def ping(self):
        return "ok"


@ray.remote(num_cpus=2, num_gpus=0)
class SampleHashServer:
    """A thin Ray actor wrapper around SampleHashDict."""
    def __init__(self):
        self.sample_dict = SampleHashDict()

    def set_correct_answered(self, sample_id: int, value: bool):
        return self.sample_dict.set_correct_answered(sample_id, value)

    def get_info(self, sample_id: int):
        return self.sample_dict.get_info(sample_id)

    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        return self.sample_dict.update_min_mean_correct_resp_len(sample_id, resp_len)

    def look_up_min_mean_correct_resp_len(self, sample_id: int):
        return self.sample_dict.look_up_min_mean_correct_resp_len(sample_id)

    def ping(self):
        return
    
    def save_info(self, filepath: str, overwrite: bool = True):
        return self.sample_dict.save_info(filepath, overwrite)
    
    def load_info(self, filepath: str):
        return self.sample_dict.load_info(filepath)