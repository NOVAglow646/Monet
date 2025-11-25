"""
LatentRecorder: accumulate per-step latents during generation and access them
after completion. Works across processes via the UDP hook.

Usage 1 (recommended: create before LLM so it sets AVT_LATENT_HOOK_UDP):

    from avt.vllm.latent_recorder import LatentRecorder
    from vllm import LLM, SamplingParams

    rec = LatentRecorder()               # starts UDP listener and sets env
    llm = LLM(model=...)                 # create AFTER recorder
    outs = llm.generate(["hi"], SamplingParams(max_tokens=16))
    rec.stop()
    trajs = rec.get_all()                # {request_id: List[List[float]]}

Usage 2 (LLM already created):
    - Ensure AVT_LATENT_HOOK_UDP was already set to an address you can bind.
    - Create recorder with set_env=False; it will read and bind to that addr.

    os.environ["AVT_LATENT_HOOK_UDP"] = "127.0.0.1:56789"  # set before LLM
    llm = LLM(...)
    rec = LatentRecorder(set_env=False)   # binds to existing env address
    outs = llm.generate([...], ...)
    rec.stop()
    print(rec.get_all())
"""
from __future__ import annotations

import os
import pdb
from typing import Dict, List, Optional, Tuple
from threading import Lock
import re

from .latent_hook import start_udp_listener, start_tcp_listener, start_tcp_listener_bin


class LatentRecorder:
    """Collects per-step latents keyed by request_id.

    If set_env=True (default), the recorder will start a local TCP (binary) listener
    by default, and set AVT_LATENT_HOOK_TCP/AVT_LATENT_HOOK_BIN so worker processes
    send events to this recorder. Create it BEFORE you create LLM.

    If set_env=False, the recorder will attempt to bind to the address given in
    AVT_LATENT_HOOK_TCP or AVT_LATENT_HOOK_UDP (must be set beforehand). This is useful when the LLM
    was already created with a preconfigured hook destination.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, set_env: bool = True, prefer_tcp: bool = True, filter_rank: Optional[int] = None):
        self._lock = Lock()
        self._traj: Dict[str, List[List[float]]] = {}
        # Grouping cache: parent_id -> sample_index -> trajectory
        self._group: Dict[str, Dict[int, List[List[float]]]] = {}
        self._stop = None
        self._addr = None
        # Optional rank filter (drop events not matching)
        if filter_rank is None:
            env_rank = os.environ.get("AVT_FILTER_RANK")
            try:
                filter_rank = int(env_rank) if env_rank is not None else None
            except Exception:
                filter_rank = None
        self._filter_rank: Optional[int] = filter_rank
        if set_env:
            # Start listener (TCP preferred for reliability) and export address for workers
            if prefer_tcp:
                # Prefer binary TCP frames for efficiency
                self._addr, self._stop = start_tcp_listener_bin(host, port, on_event=self._on_event)
                os.environ["AVT_LATENT_HOOK_TCP"] = f"{self._addr[0]}:{self._addr[1]}"
                os.environ["AVT_LATENT_HOOK_BIN"] = "1"
            else:
                self._addr, self._stop = start_udp_listener(host, port, on_event=self._on_event)
                os.environ["AVT_LATENT_HOOK_UDP"] = f"{self._addr[0]}:{self._addr[1]}"
        else:
            # Bind to the existing address from env
            tcp_env = os.environ.get("AVT_LATENT_HOOK_TCP")
            udp_env = os.environ.get("AVT_LATENT_HOOK_UDP")
            env = tcp_env or udp_env
            if not env:
                raise RuntimeError("Neither AVT_LATENT_HOOK_TCP nor AVT_LATENT_HOOK_UDP is set and set_env=False; cannot receive latents.")
            try:
                host_env, port_env = env.rsplit(":", 1)
                port_int = int(port_env)
            except Exception as e:
                raise RuntimeError(f"Invalid hook address '{env}'") from e
            if tcp_env:
                self._addr, self._stop = start_tcp_listener(host_env, port_int, on_event=self._on_event)
            else:
                self._addr, self._stop = start_udp_listener(host_env, port_int, on_event=self._on_event)

    # Callback for events from latent_hook
    def _on_event(self, evt):
        # Optional rank filtering
        #pdb.set_trace()
        if self._filter_rank is not None:
            evt_rank = evt.get("rank")
            # If event has no rank (older emitters), drop to avoid cross-rank mix
            if evt_rank is None or int(evt_rank) != int(self._filter_rank):
                #if os.environ.get("AVT_DEBUG") == "1":
                print(f"Dropping event from rank {evt_rank} (listener: {self._filter_rank})")
                return
        req_ids = evt.get("req_ids") or []
        latents = evt.get("latents") or []
        with self._lock:
            for rid, lat in zip(req_ids, latents):
                if not lat:
                    continue
                #breakpoint()
                #pdb.set_trace()
                self._traj.setdefault(rid, []).append(lat)
                parent, sample = self._parse_req_id(rid)
                self._group.setdefault(parent, {}).setdefault(sample, []).append(lat)

    @staticmethod
    def _parse_req_id(rid: str) -> Tuple[str, int]:
        """Heuristically parse a request_id into (parent_id, sample_index).

        Expected patterns (best effort, non-fatal):
        - "<sample>_<parent>", e.g. "0_1" -> ("1", 0)
        - "<digits>", e.g. "3" -> ("3", 0)
        Fallback: (rid, 0)
        """
        # 1) strict "d+[_-]d+" pattern: treat last as parent, first as sample
        m = re.match(r"^(\d+)[_-](\d+)$", rid)
        if m:
            try:
                return m.group(2), int(m.group(1))
            except Exception:
                return rid, 0
        # 2) pure digits -> parent=rid, sample=0
        if rid.isdigit():
            return rid, 0
        # 3) best-effort: split by '_' and take last numeric as parent, previous numeric as sample
        if "_" in rid:
            parts = rid.split("_")
            if parts[-1].isdigit():
                parent = parts[-1]
                # sample defaults to 0; if penultimate numeric exists, use it
                sample = 0
                if len(parts) >= 2 and parts[-2].isdigit():
                    sample = int(parts[-2])
                return parent, sample
        return rid, 0

    def stop(self) -> None:
        if self._stop:
            self._stop()
            self._stop = None

    def get(self, request_id: str) -> List[List[float]]:
        with self._lock:
            return list(self._traj.get(request_id, []))

    def get_all(self) -> Dict[str, List[List[float]]]:
        with self._lock:
            # deep-copy outer lists only; inner lists are primitives
            return {k: list(v) for k, v in self._traj.items()}

    def get_grouped(self) -> Dict[str, Dict[int, List[List[float]]]]:
        """Get nested grouping: {parent_id: {sample_index: [latents...]}}"""
        with self._lock:
            # shallow copies at each level; inner lists are primitives
            return {p: {s: list(traj) for s, traj in d.items()} for p, d in self._group.items()}

    def get_by_parent(self, parent_id: str) -> Dict[int, List[List[float]]]:
        with self._lock:
            return {s: list(traj) for s, traj in self._group.get(parent_id, {}).items()}

    def get_by_index(self, parent_index: int) -> Dict[int, List[List[float]]]:
        """Convenience for integer-indexed parents (e.g., "0", "1", ...)."""
        return self.get_by_parent(str(parent_index))

    # Tensor-returning helpers
    def get_all_tensors(self):
        """Like get_all(), but each step vector is a torch.tensor(float16)."""
        import torch  # local import to avoid hard dep when unused
        with self._lock:
            return {rid: [torch.tensor(x, dtype=torch.float16) for x in steps]
                    for rid, steps in self._traj.items()}

    def get_grouped_tensors(self):
        """Like get_grouped(), but step vectors are torch.tensor(float16)."""
        import torch  # local import to avoid hard dep when unused
        with self._lock:
            return {
                parent: {sidx: [torch.tensor(x, dtype=torch.float16) for x in traj]
                         for sidx, traj in samples.items()}
                for parent, samples in self._group.items()
            }
        
    def get_grouped_arrays(self, bsz, rollout_n):
        """Like get_grouped(), but step vectors are numpy.ndarray(float16)."""
        import numpy as np  # local import to avoid hard dep when unused
        with self._lock:
            
            return {
                parent: {sidx: [np.array(x) for x in traj]
                         for sidx, traj in samples.items()}
                for parent, samples in self._group.items()
            }

    def to_object_array(self, bsz: int, rollout_n: int):
        """Arrange self._group into a 2D numpy array of dtype=object.

        Shape: (rollout_n, bsz)
        - Columns are parent ids in numeric order 0..bsz-1.
        - Rows are sample ids in numeric order 0..rollout_n-1 (top to bottom).
        - Missing parent -> a full column of None.
        - Existing parent but missing sample -> that cell is None.

        Each cell holds the trajectory list (List[List[float]]) for that
        parent/sample pair, or None if missing.
        """
        import numpy as np  # local import to avoid hard dep when unused
        arr = np.empty((rollout_n, bsz), dtype=object)
        with self._lock:
            for p in range(bsz):
                parent_id = str(p)
                samples = self._group.get(parent_id)
                if not samples:
                    for s in range(rollout_n):
                        arr[s, p] = None
                    continue
                for s in range(rollout_n):
                    traj = samples.get(s)
                    arr[s, p] = list(traj) if traj is not None else None
        return arr

    def to_object_array_auto(self, bsz: int, rollout_n: int, min_req_id: int):
        """Return a 1D numpy object array of length bsz*rollout_n.

        Ordering: parent-major then sample-minor.
        - Parent ids observed are numeric and in range [2*bsz, 3*bsz-1];
          we remap by parent_mapped = int(parent) - 2*bsz, so columns are 0..bsz-1.
        - For each mapped parent p (0..bsz-1) and sample s (0..rollout_n-1),
          the linear index is idx = p*rollout_n + s.
        - Missing parent/sample entries are filled with None.
        - Each element is a numpy.ndarray(float16) of shape (steps, H), or None.
        """
        import numpy as np
        total = bsz * rollout_n
        arr = np.empty((total,), dtype=object)
        # init to None
        #breakpoint()
        #pdb.set_trace() # self._group['4'].keys()
        for i in range(total):
            arr[i] = None
        with self._lock:
            for parent_str, samples in self._group.items():
                #pdb.set_trace()
                if not (isinstance(parent_str, str) and parent_str.isdigit()):
                    continue
                parent_orig = int(parent_str)
                parent_mapped = parent_orig - min_req_id
                if parent_mapped < 0 or parent_mapped >= bsz:
                    #breakpoint()
                    raise RuntimeError(f"Parent id {parent_orig} out of expected range [{min_req_id}, {min_req_id + bsz - 1}]")
                    #continue
                for s in range(rollout_n):
                    traj = samples.get(s)
                    idx = parent_mapped * rollout_n + s
                    arr[idx] = (np.asarray(traj, dtype=np.float16)
                                if traj is not None else None)
        #breakpoint()
        return arr

    def to_array_by_req_ids(self, req_ids: List[str]):
        """Return a 1D numpy object array aligned to the provided req_ids order.

        Each element is a numpy.ndarray(float16) of shape (steps, H), or None
        if no latents recorded for that request id. This avoids heuristic
        parent/sample remapping and is robust to arbitrary request id formats.
        """
        import numpy as np
        arr = np.empty((len(req_ids),), dtype=object)
        with self._lock:
            for i, rid in enumerate(req_ids):
                traj = self._traj.get(rid)
                arr[i] = (np.asarray(traj, dtype=np.float16) if traj else None)
        return arr

    # Context manager helpers
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
