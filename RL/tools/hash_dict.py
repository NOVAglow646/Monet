import numpy as np
from collections import defaultdict
from typing import List, Dict, Union
import pickle
import os

class StepHashDict:
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        correct_cluster_threshold: float = 0.5, # whether a cluster is correct or not
        rep_mode: str = "all",          # "first" | "centroid" | "medoid" | "all"
    ):
        self.dicts: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self.resp_len_stats: Dict[int, dict] = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0})
        self.similarity_threshold = similarity_threshold
        self.correct_cluster_threshold = correct_cluster_threshold
        self.rep_mode = rep_mode.lower()
        assert self.rep_mode in {"first", "centroid", "medoid", "all"}
        # first    : 代表向量固定为首个成员
        # centroid : 代表向量为均值
        # medoid   : 代表向量为离均值最近成员
        # all      : *平均* 相似度过阈才并入簇，代表向量仍取首成员

    # ------------ 私有辅助函数 ------------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _build_rep_matrix(self, clusters: Dict[int, dict]) -> np.ndarray:
        """把当前所有 rep_embedding 拼成 (K, D) 矩阵；K=0 返回 None"""
        if not clusters:
            return None
        reps = [info["rep_embedding"] for info in clusters.values()]
        reps = np.vstack(reps).copy()
        reps.setflags(write=True)
        return reps

    # ------------ 对外接口 ------------
    def update_sample_step_hash_dict(
        self,
        sample_id: int,
        embeddings: np.ndarray,   # (N, D) 已 L2 归一化
        texts: List[str],
        lead_correct_list: List[bool] | None = None
    ):
        #breakpoint()
        assert len(embeddings) == len(texts), "embeddings 和 texts 数量不一致"

        clusters = self.dicts[sample_id]          # 取引用
        rep_matrix = self._build_rep_matrix(clusters)
        correctness = []
        
        for idx, (emb, txt) in enumerate(zip(embeddings, texts)):
            lead_to_correct = lead_correct_list[idx] if lead_correct_list else None

            # ---------- 第一个样本 ----------
            if rep_matrix is None:
                clusters[0] = dict(
                    rep_embedding=emb,            # all/first 均保持不变
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=[emb],
                    correct_cnt=1 if lead_to_correct else 0
                )
                rep_matrix = emb[None, :].copy()
                rep_matrix.setflags(write=True)
                correctness.append(True if lead_to_correct else False)
                continue

            # ---------- 找最合适的簇 ----------
            insert_cid = None
            if self.rep_mode == "all":
                    # ① 先用代表向量做一次粗筛
                sims_rep = rep_matrix @ emb           # (K,)
                cand_cids = np.where(sims_rep > self.similarity_threshold)[0]
                
                if cand_cids.size:                    # 有潜在候选才进一步检查
                    best_avg, insert_cid = -1.0, None
                    for cid in cand_cids:
                        cinfo = clusters[cid]
                        member_embs = cinfo["member_embeddings"]  # 直接就是 (M,D) ndarray
                        sims = member_embs @ emb                  # (M,)
                        
                        if np.all(sims > self.similarity_threshold):
                            avg_sim = sims.mean()
                            if avg_sim > best_avg:
                                insert_cid, best_avg = cid, avg_sim
            else:
                sims = np.dot(rep_matrix, emb)      # (K,)
                best_row = int(np.argmax(sims))
                if float(sims[best_row]) > self.similarity_threshold:
                    insert_cid = best_row

            # ---------- 插入或新建 ----------
            if insert_cid is not None:              # 插入现有簇
                cinfo = clusters[insert_cid]
                cinfo["members_texts"].append(txt)
                cinfo["members_idx"].append(idx)
                cinfo["correct_cnt"] += 1 if lead_to_correct else 0
                correctness.append(True if cinfo["correct_cnt"]/len(cinfo["members_texts"])> self.correct_cluster_threshold else False)
                cinfo["member_embeddings"] = np.concatenate(
                    (cinfo["member_embeddings"], emb[None, :]), axis=0
                )

                # 仅 centroid/medoid 需要更新代表向量与 rep_matrix
                if self.rep_mode == "centroid":
                    new_rep = self._normalize(np.mean(cinfo["member_embeddings"], 0))
                    cinfo["rep_embedding"] = new_rep
                    rep_matrix[insert_cid] = new_rep
                elif self.rep_mode == "medoid":
                    centroid = np.mean(cinfo["member_embeddings"], 0)
                    sims_centroid = np.dot(cinfo["member_embeddings"], centroid)
                    best_idx = int(np.argmax(sims_centroid))
                    new_rep = cinfo["member_embeddings"][best_idx]
                    cinfo["rep_embedding"] = new_rep
                    cinfo["rep_text"] = cinfo["members_texts"][best_idx]
                    rep_matrix[insert_cid] = new_rep
                # rep_mode == "first" 或 "all"：代表向量保持不变

            else:                                   # 新建簇
                new_cid = len(clusters)
                clusters[new_cid] = dict(
                    rep_embedding=emb,
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=emb[None, :].copy(),   # (1,D) ndarray
                    correct_cnt=1 if lead_to_correct else 0
                )
                rep_matrix = np.vstack([rep_matrix, emb[None, :]]).copy()
                rep_matrix.setflags(write=True)
                correctness.append(True if lead_to_correct else False)
        return correctness


    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        self.resp_len_stats[sample_id]["min_len"] = min(
            self.resp_len_stats[sample_id]["min_len"], resp_len
        )
        self.resp_len_stats[sample_id]["mean_len"] = (
            self.resp_len_stats[sample_id]["mean_len"] * self.resp_len_stats[sample_id]["cnt"] + resp_len
        ) / (self.resp_len_stats[sample_id]["cnt"] + 1)
        self.resp_len_stats[sample_id]["cnt"] += 1

    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        return self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["min_len"], \
               self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["mean_len"]

    def look_up_step_correctness(
        self,
        sample_id: int,
        texts: Union[str, List[str]]
    ) -> List[bool]:
        """
        按 *字符串* 精确匹配 members_texts：
        - 输入可以是单个 str，也可以是 str 列表。
        - 对于每个待查字符串，遍历该 sample 的所有簇，
          若在某簇 cinfo["members_texts"] 中找到完全一致的项，
          返回该簇的 lead_to_correct。
        - 若找不到，则抛 ValueError。
        """
        # 统一成列表
        if isinstance(texts, str):
            texts = [texts]

        clusters = self.dicts.get(sample_id, {})
        if not clusters:
            raise KeyError(f"No clusters found for sample_id {sample_id}")

        results: List[bool] = []

        for query in texts:
            found = False
            for cinfo in clusters.values():
                if query in cinfo["members_texts"]:
                    results.append(True if cinfo['correct_cnt'] / len(cinfo["members_texts"]) > self.correct_cluster_threshold else False)
                    found = True
                    break

            if not found:
                raise ValueError(
                    f'Text "{query}" not found in any cluster for sample_id {sample_id}'
                )

        return results
    
    
    def get_step_dict_info(self, verbose_info: bool = False, print_info: bool = False):
        """
        打印当前字典的统计信息。
        """
        info_dict = defaultdict(dict)
        if print_info:
            print(f"Total samples: {len(self.dicts)}")
        for sample_id, clusters in self.dicts.items():
            avg_member_len = np.mean([len(cinfo["members_texts"]) for cinfo in clusters.values()])
            if print_info:
                print(f"Sample ID: {sample_id}, Clusters: {len(clusters)}, Avg Members: {avg_member_len:.2f}")
            info_dict[sample_id]["overall_info"] = {
                "clusters_cnt": len(clusters),
                "avg_member_len": avg_member_len
            }
            if verbose_info:
                info_dict[sample_id]["verbose_info"] = []
                for cid, cinfo in clusters.items():
                    if print_info:
                        print(f"  Cluster ID: {cid}, Rep text: {cinfo['rep_text'][:80]}, Members: {len(cinfo['members_texts'])}, Acc: {cinfo['correct_cnt'] / len(cinfo['members_texts']) if cinfo['members_texts'] else 0}")
                    info_dict[sample_id]["verbose_info"].append(
                        {
                            "cluster_id": cid,
                            "rep_text": cinfo["rep_text"][:80],
                            "members_count": len(cinfo["members_texts"]),
                            "sampled_member_texts": cinfo["members_texts"],
                            "lead_to_correct": cinfo["correct_cnt"],
                            "accuracy": cinfo["correct_cnt"] / len(cinfo["members_texts"]) if cinfo["members_texts"] else 0,
                        }
                    )
                
        return info_dict


    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """
        将当前 self.dicts 序列化保存到本地文件。

        Parameters
        ----------
        filepath : str
            目标文件路径，建议以 .pkl 结尾。
        overwrite : bool
            当文件已存在时是否覆盖，默认 True。
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"{filepath} already exists. "
                                  "Set overwrite=True to overwrite.")
        # defaultdict -> 普通 dict 更通用
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "wb") as f:
            pickle.dump(dicts_to_dump, f)
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "wb") as f:
            pickle.dump(resp_len_stats_to_dump, f)

        print(f"[StepHashDict] and [RespLenStats] saved to folder {filepath}")

    def load_info(self, filepath: str) -> None:
        """
        从本地文件加载 dicts 并覆盖当前 self.dicts。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "rb") as f:
            dicts_loaded = pickle.load(f)
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "rb") as f:
            resp_len_stats_loaded = pickle.load(f)
        # 再包回 defaultdict 以保持行为一致
        self.dicts = defaultdict(dict, dicts_loaded)
        self.resp_len_stats = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}, resp_len_stats_loaded)
        print(f"[StepHashDict] and [RespLenStats] loaded dicts from folder {filepath}")


'''d = StepHashDict(similarity_threshold=0.85, rep_mode="medoid")
d.update_sample_step_hash_dict(
    sample_id=1,
    embeddings=np.array([[0.1, 0.2], [0.2, 0.3], [0.9, 0.8]]),
    texts=["", "a", "b"],
    lead_correct_list=[True, False, True]
)

d.look_up_step_correctness(
    sample_id=1,
    texts=["", "b", "c"]
)'''


class SampleHashDict:
    """A lightweight per-sample info store.

    - self.dicts[sample_id] -> info_dict with keys:
        - 'corret_answered': bool  是否已经答对（一次为真永久为真）
        - 'min_len': float         该 sample 观测到的最短响应长度
    - 同时保留最小/均值响应长度统计（与 StepHashDict 的接口一致）：
        update_min_mean_correct_resp_len / look_up_min_mean_correct_resp_len
    """

    def __init__(self):
        self.dicts: Dict[int, dict] = defaultdict(lambda: {"corret_answered": False, "min_len": float("inf")})
        self.resp_len_stats: Dict[int, dict] = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0})

    # ---- Public APIs ----
    def set_correct_answered(self, sample_id: int, value: bool) -> None:
        info = self.dicts[sample_id]
        # 一旦为 True 就保持 True（幂等累计）
        info["corret_answered"] = bool(info.get("corret_answered", False) or value)

    def get_info(self, sample_id: int) -> dict:
        # 返回一个浅拷贝以避免外部修改内部状态
        info = self.dicts[sample_id]
        return dict(info)

    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        # 更新最小/均值响应长度统计
        stats = self.resp_len_stats[sample_id]
        stats["min_len"] = min(stats["min_len"], resp_len)
        stats["mean_len"] = (stats["mean_len"] * stats["cnt"] + resp_len) / (stats["cnt"] + 1)
        stats["cnt"] += 1
        # 同步更新 info_dict 的 min_len
        info = self.dicts[sample_id]
        info["min_len"] = min(info.get("min_len", float("inf")), resp_len)
        return None

    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        stats = self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})
        return stats["min_len"], stats["mean_len"]

    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """将当前字典序列化保存到本地目录 filepath 下。

        会输出两个文件：
        - sample_hash_dict.pkl
        - sample_resp_len_stats.pkl
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"{filepath} already exists. Set overwrite=True to overwrite.")
        # 使用普通 dict 以提升兼容性
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'wb') as f:
            pickle.dump(dicts_to_dump, f)
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'wb') as f:
            pickle.dump(resp_len_stats_to_dump, f)
        print(f"[SampleHashDict] and [SampleRespLenStats] saved to folder {filepath}")

    def load_info(self, filepath: str) -> None:
        """从本地目录加载并覆盖当前字典。"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'rb') as f:
            dicts_loaded = pickle.load(f)
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'rb') as f:
            resp_len_stats_loaded = pickle.load(f)
        # 包回 defaultdict 以保持默认值行为
        self.dicts = defaultdict(lambda: {"corret_answered": False, "min_len": float("inf")}, dicts_loaded)
        self.resp_len_stats = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}, resp_len_stats_loaded)
        print(f"[SampleHashDict] and [SampleRespLenStats] loaded dicts from folder {filepath}")