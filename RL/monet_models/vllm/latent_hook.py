"""
Lightweight latent-steps hook for vLLM workers.

In-process subscribers: subscribe(fn) to receive events in the same process.
Optional UDP publisher: set env AVT_LATENT_HOOK_UDP="host:port" and each
worker will emit a JSON line per decoding step with latent vectors.

Payload shape (per step):
{
    "ts": float,            # epoch seconds
    "pid": int,             # worker pid
    "rank": Optional[int],  # torch.distributed rank if available
    "req_ids": [str,...],   # request ids aligned with latents
    "latents": [            # per-request vectors or nulls, aligned to req_ids
        [float, ...] | null,
        ...
    ]
}

Notes:
- Full latent vectors may be large. UDP packets have size limits; oversized
    payloads can be dropped by the network stack. For large vectors, prefer the
    in-process subscribe() API or ensure local loopback with modest batch sizes.
"""
from __future__ import annotations

import os
import json
import time
import socket
import threading
import struct
from typing import Callable, Optional, List, Dict, Any
import numpy as np

_SUBS: list[Callable[[Dict[str, Any]], None]] = []
_UDP_SOCK: Optional[socket.socket] = None
_UDP_ADDR: Optional[tuple[str, int]] = None
_TCP_SOCK: Optional[socket.socket] = None
_TCP_ADDR: Optional[tuple[str, int]] = None
_TCP_LOCK = threading.Lock()


def _init_udp() -> None:
    global _UDP_SOCK, _UDP_ADDR
    if _UDP_SOCK is not None:
        return
    addr = os.environ.get("AVT_LATENT_HOOK_UDP")
    if not addr:
        return
    try:
        host, port_s = addr.rsplit(":", 1)
        port = int(port_s)
    except Exception:
        # Ignore invalid config
        return
    try:
        _UDP_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _UDP_ADDR = (host, port)
    except OSError:
        _UDP_SOCK = None
        _UDP_ADDR = None


def _init_tcp() -> None:
    """Initialize a TCP client socket if AVT_LATENT_HOOK_TCP is set."""
    global _TCP_SOCK, _TCP_ADDR
    if _TCP_SOCK is not None:
        return
    addr = os.environ.get("AVT_LATENT_HOOK_TCP")
    if not addr:
        return
    try:
        host, port_s = addr.rsplit(":", 1)
        port = int(port_s)
    except Exception:
        return
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(1.0)
        s.connect((host, port))
        s.settimeout(None)
        _TCP_SOCK = s
        _TCP_ADDR = (host, port)
    except OSError:
        _TCP_SOCK = None
        _TCP_ADDR = None


def subscribe(callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    """Register an in-process subscriber. Returns an unsubscribe function."""
    _SUBS.append(callback)

    def _unsub() -> None:
        try:
            _SUBS.remove(callback)
        except ValueError:
            pass

    return _unsub


def emit_latents_step(
    req_ids: List[str],
    latents: List[Optional[Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a per-step event to local subscribers and optional UDP sink.

    Args:
        req_ids: Current batch request ids in-order.
        latents: Per-request small vectors (or None) aligned to req_ids.
        extra:   Optional extra metadata to include.
    """
    # Build payload
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "pid": os.getpid(),
        "req_ids": list(req_ids),
        "latents": latents,
    }
    if extra:
        payload.update(extra)

    # Attach distributed rank if available (best-effort)
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            payload.setdefault("rank", dist.get_rank())
    except Exception:
        pass

    # Notify in-proc subscribers
    for cb in list(_SUBS):
        try:
            cb(payload)
        except Exception:
            # Don't let user callbacks break the loop
            continue

    global _TCP_SOCK
    data = None
    try:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    except Exception:
        # Likely because latents contain tensors; that's fine when using binary mode.
        data = None

    # Optional TCP publish in binary frame mode (more efficient than JSON)
    bin_mode = os.environ.get("AVT_LATENT_HOOK_BIN", "0") == "1"
    if bin_mode:
        _init_tcp()
        if _TCP_SOCK is not None:
            try:
                _send_tcp_bin(req_ids, latents, payload.get("step"))
                return  # done (prefer binary over JSON)
            except Exception:
                try:
                    _TCP_SOCK.close()
                except Exception:
                    pass
                _TCP_SOCK = None

    # Optional TCP publish (preferred when configured)
    if data is not None:
        _init_tcp()
        if _TCP_SOCK is not None:
            try:
                with _TCP_LOCK:
                    _TCP_SOCK.sendall(data + b"\n")
            except Exception:
                # Reset and skip this event; will retry next time
                try:
                    _TCP_SOCK.close()
                except Exception:
                    pass
                _TCP_SOCK = None

    # Optional UDP publish (best-effort)
    if data is not None:
        _init_udp()
        if _UDP_SOCK is not None and _UDP_ADDR is not None:
            try:
                _UDP_SOCK.sendto(data, _UDP_ADDR)
            except Exception:
                pass


def _send_tcp_bin(req_ids: List[str], latents: List[Optional[Any]], step: Optional[int]) -> None:
    """Pack a binary frame and send over TCP. Preserves order and nulls.

    Frame format (big-endian):
    - magic:      4s  'AVTB'
    - version:    B   1 or 2 (2 adds rank)
    - flags:      B   bit0: 1=float16, 0=float32; others reserved
    - reserved:   H   0
    - frame_len:  I   total bytes including header
    - n_reqs:     H
    - dim:        I   vector length; 0 if all None
    - step:       i   -1 if unknown
    - rank:       i   (only when version==2; -1 if unknown)
    - repeated per req (n_reqs times):
        - rid_len:  H
        - rid:      bytes
        - has_vec:  B  (1 or 0)
        - vec:      dim * elem_size bytes if has_vec==1
    """
    global _TCP_SOCK
    # Choose dtype via env (default fp16)
    use_fp16 = os.environ.get("AVT_LATENT_DTYPE", "fp16").lower() in ("fp16", "float16", "half")
    flags = 1 if use_fp16 else 0
    n = len(req_ids)
    # infer dim from the first non-None vector
    dim = 0
    for v in latents:
        if v is not None:
            # Determine length without materializing when possible
            try:
                dim = len(v)  # works for list/np/torch
            except Exception:
                try:
                    # torch.Tensor has .numel() and is 1-D here
                    dim = int(getattr(v, "numel", lambda: 0)())
                except Exception:
                    dim = 0
            break
    step_i = int(step) if isinstance(step, (int, np.integer)) else -1
    # Try obtain rank (best-effort)
    rank_i = -1
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            rank_i = int(dist.get_rank())
    except Exception:
        rank_i = -1
    # Use header version 2 (with rank)
    header_fmt = ">4sBBHIHIii"
    header_len = struct.calcsize(header_fmt)
    parts: list[bytes] = []
    elem_size = 2 if use_fp16 else 4
    for rid, v in zip(req_ids, latents):
        rid_b = rid.encode("utf-8")
        parts.append(struct.pack(">H", len(rid_b)))
        parts.append(rid_b)
        if v is None or dim == 0:
            parts.append(struct.pack(">B", 0))
        else:
            parts.append(struct.pack(">B", 1))
            # Normalize v to a 1-D NumPy array (handles torch.Tensor gracefully)
            arr_np = None
            # Torch path (duck-typed): detach->cpu->contiguous->numpy
            if hasattr(v, "detach"):
                v = v.detach()
            if hasattr(v, "cpu"):
                v = v.cpu()
            if hasattr(v, "contiguous"):
                v = v.contiguous()
            if hasattr(v, "numpy"):
                arr_np = v.numpy()

            if arr_np is None:
                arr_np = np.asarray(v)
            arr = arr_np.astype(np.float16 if use_fp16 else np.float32, copy=False)
            if dim != len(arr):
                # pad/truncate defensively if mismatch
                if len(arr) < dim:
                    pad = np.zeros(dim - len(arr), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad])
                else:
                    arr = arr[:dim]
            parts.append(arr.tobytes(order="C"))
    body = b"".join(parts)
    frame_len = header_len + len(body)
    header = struct.pack(
        header_fmt,
        b"AVTB",  # magic
        2,         # version (v2: includes rank)
        flags,
        0,         # reserved
        frame_len,
        n,
        dim,
        step_i,
        rank_i,
    )
    frame = header + body
    with _TCP_LOCK:
        _TCP_SOCK.sendall(frame)


def start_udp_listener(host: str = "127.0.0.1", port: int = 0,
                       on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
    """Utility for driver-side testing: start a background UDP listener.

    Returns (addr, stop_fn) where addr=(host,port) bound, and stop_fn() stops it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Best-effort: allow port reuse to avoid EADDRINUSE on re-runs
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Some platforms expose SO_REUSEPORT for UDP fanout
            reuse_port = getattr(socket, "SO_REUSEPORT", None)
            if reuse_port is not None:
                sock.setsockopt(socket.SOL_SOCKET, reuse_port, 1)
        except Exception:
            pass
    except Exception:
        pass
    sock.bind((host, port))
    real_addr = sock.getsockname()
    stop_flag = threading.Event()

    def _loop():
        sock.settimeout(0.5)
        while not stop_flag.is_set():
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                evt = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if on_event:
                try:
                    on_event(evt)
                except Exception:
                    pass
    th = threading.Thread(target=_loop, daemon=True)
    th.start()

    def _stop():
        stop_flag.set()
        try:
            sock.close()
        except Exception:
            pass

    return real_addr, _stop


def start_tcp_listener(host: str = "127.0.0.1", port: int = 0,
                       on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
    """Start a background TCP JSONL server. Returns (addr, stop_fn)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    srv.bind((host, port))
    srv.listen(8)
    real_addr = srv.getsockname()
    stop_flag = threading.Event()

    def _handle_client(conn: socket.socket):
        buf = b""
        try:
            conn.settimeout(0.5)
            while not stop_flag.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl], buf[nl + 1:]
                    try:
                        evt = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if on_event:
                        try:
                            on_event(evt)
                        except Exception:
                            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _accept_loop():
        while not stop_flag.is_set():
            try:
                srv.settimeout(0.5)
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            th = threading.Thread(target=_handle_client, args=(conn,), daemon=True)
            th.start()

    th = threading.Thread(target=_accept_loop, daemon=True)
    th.start()

    def _stop():
        stop_flag.set()
        try:
            srv.close()
        except Exception:
            pass

    return real_addr, _stop


def start_tcp_listener_bin(host: str = "127.0.0.1", port: int = 0,
                           on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
    """Start a background TCP server for AVTB binary frames. Returns (addr, stop_fn)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    srv.bind((host, port))
    srv.listen(8)
    real_addr = srv.getsockname()
    stop_flag = threading.Event()

    # Support v1 (no rank) and v2 (with rank) headers
    preamble_fmt = ">4sBBH"  # magic, ver, flags, rsv
    preamble_len = struct.calcsize(preamble_fmt)
    header_fmt_v1 = ">4sBBHIHIi"      # + frame_len, n_reqs, dim, step
    header_len_v1 = struct.calcsize(header_fmt_v1)
    header_fmt_v2 = ">4sBBHIHIii"     # + rank
    header_len_v2 = struct.calcsize(header_fmt_v2)

    def _handle_client(conn: socket.socket):
        buf = b""
        try:
            conn.settimeout(0.5)
            while not stop_flag.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                # parse frames
                while True:
                    # Need at least preamble to know version
                    if len(buf) < preamble_len:
                        break
                    magic, ver, flags, rsv = struct.unpack(preamble_fmt, buf[:preamble_len])
                    if magic != b"AVTB" or ver not in (1, 2):
                        # invalid; drop buffer
                        buf = b""
                        break
                    # pick header based on version
                    if ver == 1:
                        header_len = header_len_v1
                        if len(buf) < header_len:
                            break
                        magic, ver, flags, rsv, frame_len, n_reqs, dim, step = struct.unpack(header_fmt_v1, buf[:header_len])
                        rank = None
                    else:
                        header_len = header_len_v2
                        if len(buf) < header_len:
                            break
                        magic, ver, flags, rsv, frame_len, n_reqs, dim, step, rank = struct.unpack(header_fmt_v2, buf[:header_len])
                    if frame_len < header_len or len(buf) < frame_len:
                        # incomplete frame
                        break
                    body = buf[header_len:frame_len]
                    buf = buf[frame_len:]
                    # decode body
                    req_ids: list[str] = []
                    latents: list[Optional[list[float]]] = []
                    pos = 0
                    elem_size = 2 if (flags & 1) == 1 else 4
                    dtype = np.float16 if (flags & 1) == 1 else np.float32
                    for _ in range(n_reqs):
                        if pos + 2 > len(body):
                            req_ids.append("")
                            latents.append(None)
                            break
                        (rid_len,) = struct.unpack_from(">H", body, pos)
                        pos += 2
                        rid = body[pos:pos + rid_len].decode("utf-8", errors="ignore")
                        pos += rid_len
                        if pos + 1 > len(body):
                            req_ids.append(rid)
                            latents.append(None)
                            break
                        (has_vec,) = struct.unpack_from(">B", body, pos)
                        pos += 1
                        if has_vec and dim > 0:
                            nbytes = dim * elem_size
                            vec_b = body[pos:pos + nbytes]
                            pos += nbytes
                            try:
                                arr = np.frombuffer(vec_b, dtype=dtype, count=dim)
                                lat = arr.astype(np.float32).tolist()
                            except Exception:
                                lat = None
                            latents.append(lat)
                        else:
                            latents.append(None)
                        req_ids.append(rid)
                    evt = {
                        "req_ids": req_ids,
                        "latents": latents,
                        "step": step,
                    }
                    if rank is not None:
                        evt["rank"] = int(rank)
                    if on_event:
                        try:
                            on_event(evt)
                        except Exception:
                            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _accept_loop():
        while not stop_flag.is_set():
            try:
                srv.settimeout(0.5)
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            th = threading.Thread(target=_handle_client, args=(conn,), daemon=True)
            th.start()

    th = threading.Thread(target=_accept_loop, daemon=True)
    th.start()

    def _stop():
        stop_flag.set()
        try:
            srv.close()
        except Exception:
            pass

    return real_addr, _stop
