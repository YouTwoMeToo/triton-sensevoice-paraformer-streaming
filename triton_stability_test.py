#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 对Triton部署的sensevoice模型进行压测

import argparse
import os
import random
import sys
import threading
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import gcd
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import requests

# =========================
# 配置与音频预处理
# =========================

TARGET_SR = 16000


def load_wav_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x.astype(np.float32, copy=False)
    g = gcd(sr, TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)


def dc_remove_and_peak_norm(x: np.ndarray, peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    x = x - np.mean(x)
    mx = np.max(np.abs(x)) + eps
    return (x / mx * peak).astype(np.float32, copy=False)


def preprocess_wav(path: str) -> Tuple[np.ndarray, float]:
    """返回 (float32 波形, 秒级时长)，时长在 16k 采样后计算。"""
    wav, sr = load_wav_mono_float32(path)
    wav = resample_to_16k(wav, sr)
    wav = dc_remove_and_peak_norm(wav, peak=0.95)
    duration_s = len(wav) / float(TARGET_SR)
    return wav, duration_s


# =========================
# 纯 HTTP/REST 推理（不依赖 tritonclient）
# =========================

_tls = threading.local()  # 每个线程独立的 Session


def _get_thread_session() -> requests.Session:
    if not hasattr(_tls, "sess"):
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json"})
        _tls.sess = s
    return _tls.sess


def _norm_base_url(url: str) -> str:
    return url if url.startswith("http://") or url.startswith("https://") else ("http://" + url)


def triton_infer_once_http_json(
        url: str,
        model_name: str,
        wav_f32: np.ndarray,
        lang_id: int,
        text_norm: int,
        request_timeout_s: Optional[float] = None,
) -> Optional[str]:
    """
    直接按 Triton HTTP/REST v2 协议用 JSON 发送推理请求。
    注意：这里为简便使用 JSON 数据字段（不是二进制拼接）。
    """
    base = _norm_base_url(url).rstrip("/")
    endpoint = f"{base}/v2/models/{model_name}/infer"

    # 组装输入张量（JSON）
    T = int(wav_f32.shape[0])
    wav_batch = wav_f32[np.newaxis, :]  # (1, T)

    # JSON 模式要求 Python 原生类型
    inp_wav = {
        "name": "WAV",
        "datatype": "FP32",
        "shape": [1, T],
        "data": wav_batch.flatten().astype(np.float32).tolist(),
    }
    inp_len = {
        "name": "WAV_LENS",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [T],
    }
    inp_lang = {
        "name": "LANGUAGE",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [int(lang_id)],
    }
    inp_norm = {
        "name": "TEXT_NORM",
        "datatype": "INT32",
        "shape": [1, 1],
        "data": [int(text_norm)],
    }

    # 请求输出，显式要求非二进制，便于直接从 JSON 解析
    out_trans = {"name": "TRANSCRIPTS", "parameters": {"binary_data": False}}

    payload = {
        "inputs": [inp_wav, inp_len, inp_lang, inp_norm],
        "outputs": [out_trans],
    }

    sess = _get_thread_session()
    resp = sess.post(endpoint, json=payload, timeout=request_timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    out = resp.json()
    # 解析输出
    for o in out.get("outputs", []):
        if o.get("name") == "TRANSCRIPTS":
            data = o.get("data", None)
            if not data:
                return None
            v = data[0]
            # Triton 可能返回 bytes-like 的 base64（通常不会，因为我们禁了 binary），兼容处理
            return v if isinstance(v, str) else str(v)
    return None


# =========================
# 统计与度量
# =========================

@dataclass
class StatWindow:
    maxlen: int
    latencies: deque = field(default_factory=deque)  # seconds
    durations: deque = field(default_factory=deque)  # seconds
    rtf_list: deque = field(default_factory=deque)  # latency/duration
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, latency: float, duration: float):
        rtf = latency / max(duration, 1e-9)
        with self.lock:
            if len(self.latencies) >= self.maxlen:
                self.latencies.popleft()
                self.durations.popleft()
                self.rtf_list.popleft()
            self.latencies.append(latency)
            self.durations.append(duration)
            self.rtf_list.append(rtf)

    def snapshot(self):
        with self.lock:
            la = list(self.latencies)
            du = list(self.durations)
            rtfs = list(self.rtf_list)
        return la, du, rtfs


@dataclass
class GlobalStats:
    total_reqs: int = 0
    total_ok: int = 0
    total_err: int = 0
    total_latency: float = 0.0
    total_duration: float = 0.0
    bucket_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, ok: bool, latency: float = 0.0, duration: float = 0.0):
        with self.lock:
            self.total_reqs += 1
            if ok:
                self.total_ok += 1
                self.total_latency += latency
                self.total_duration += duration
                if duration <= 3:
                    self.bucket_counts["<=3s"] += 1
                elif duration <= 5:
                    self.bucket_counts["3-5s"] += 1
                else:
                    self.bucket_counts[">5s"] += 1
            else:
                self.total_err += 1

    def snapshot(self):
        with self.lock:
            return (self.total_reqs, self.total_ok, self.total_err,
                    self.total_latency, self.total_duration, dict(self.bucket_counts))


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


# =========================
# 工作线程与主循环
# =========================

def worker_task(
        file_path: str,
        url: str,
        model_name: str,
        lang_id: int,
        text_norm: int,
        request_timeout_s: Optional[float],
        retry: int = 1,
) -> Tuple[bool, float, float, Optional[str], Optional[Exception], str]:
    """
    返回: (ok, latency_s, duration_s, transcript, exception, file_path)
    """
    try:
        wav, duration_s = preprocess_wav(file_path)
    except Exception as e:
        return False, 0.0, 0.0, None, e, file_path

    attempt = 0
    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            text = triton_infer_once_http_json(
                url=url,
                model_name=model_name,
                wav_f32=wav,
                lang_id=lang_id,
                text_norm=text_norm,
                request_timeout_s=request_timeout_s,
            )
            latency = time.perf_counter() - t0
            return True, latency, duration_s, text, None, file_path
        except Exception as e:
            if attempt <= retry:
                time.sleep(min(0.05 * attempt, 0.5))
                continue
            return False, time.perf_counter() - t0, duration_s, None, e, file_path


def list_wavs(folder: str) -> List[str]:
    exts = {".wav", ".wave"}
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in exts:
                files.append(os.path.join(root, n))
    return files


def format_ms(x: float) -> str:
    return f"{x * 1000:.1f} ms"


def print_stats(prefix: str, latencies: List[float], durations: List[float], rtfs: List[float]):
    if not latencies:
        print(f"{prefix} 无样本")
        return

    avg_lat = sum(latencies) / len(latencies)
    avg_dur = sum(durations) / len(durations)
    p50 = pct(latencies, 50)
    p95 = pct(latencies, 95)
    p99 = pct(latencies, 99)
    avg_rtf = sum(rtfs) / len(rtfs)

    est_3s = avg_rtf * 3.0
    est_5s = avg_rtf * 5.0

    print(
        f"{prefix} size={len(latencies)} | "
        f"avg={format_ms(avg_lat)}, p50={format_ms(p50)}, p95={format_ms(p95)}, p99={format_ms(p99)} | "
        f"avg_dur={avg_dur:.2f}s | avg_RTF={avg_rtf:.3f} | "
        f"~time@3s={format_ms(est_3s)}, ~time@5s={format_ms(est_5s)}"
    )


def write_csv_header(fp):
    fp.write("ts,ok,file,latency_s,duration_s,rtf,error\n")
    fp.flush()


def write_csv_row(fp, ts, ok, file, latency, duration, rtf, err):
    err_str = "" if err is None else repr(err).replace(",", ";")
    fp.write(f"{ts},{int(ok)},{file},{latency:.6f},{duration:.6f},{rtf:.6f},{err_str}\n")


def wait_any(futures_set):
    it = as_completed(futures_set, timeout=None)
    first = next(it)
    done = {first}
    for fut in it:
        if fut.done():
            done.add(fut)
        else:
            break
    pending = futures_set - done
    return done, pending


def main():
    parser = argparse.ArgumentParser(description="Triton 并发稳定性压力测试（HTTP/REST v2 纯 requests 版）")
    parser.add_argument("--folder", required=True, help="包含若干 wav 的文件夹路径")
    parser.add_argument("--url", required=True, help='Triton HTTP 地址，如 "http://127.0.0.1:8000" 或 "10.0.0.1"')
    parser.add_argument("--model", required=True, help='Triton 模型名，如 "sensevoice"')
    parser.add_argument("-n", "--concurrency", type=int, default=8, help="并发线程数（默认 8）")
    parser.add_argument("--lang_id", type=int, default=0, help="LANGUAGE 输入（默认 0）")
    parser.add_argument("--text_norm", type=int, default=14, help="TEXT_NORM 输入（默认 14）")
    parser.add_argument("--timeout", type=float, default=30.0, help="单请求超时秒（默认 30s）")
    parser.add_argument("--max-requests", type=int, default=0, help="最多请求次数；0 表示无限持续")
    parser.add_argument("--warmup", type=int, default=2, help="预热请求数（默认 2）")
    parser.add_argument("--stats-interval", type=int, default=50, help="每处理多少条打印一次统计（默认 50）")
    parser.add_argument("--window", type=int, default=500, help="滑动窗口大小（默认 500）")
    parser.add_argument("--csv", type=str, default="", help="将明细写入 CSV 文件路径（可选）")
    args = parser.parse_args()

    wavs = list_wavs(args.folder)
    if not wavs:
        print(f"[FATAL] 在 {args.folder} 未找到 wav 文件", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 发现 {len(wavs)} 个 wav；并发={args.concurrency}，窗口={args.window}。按 Ctrl+C 停止。")

    # 预热
    if args.warmup > 0:
        print(f"[INFO] 预热 {args.warmup} 次...")
        for i in range(args.warmup):
            fp = random.choice(wavs)
            try:
                ok, lat, dur, _, err, fpath = worker_task(
                    file_path=fp,
                    url=args.url,
                    model_name=args.model,
                    lang_id=args.lang_id,
                    text_norm=args.text_norm,
                    request_timeout_s=args.timeout,
                    retry=0,
                )
                if not ok:
                    print(f"[WARN] 预热失败: {fpath} err={err}")
                else:
                    print(f"[OK] 预热{i + 1}: {format_ms(lat)} dur={dur:.2f}s RTF={lat / dur:.3f}")
            except Exception as e:
                print(f"[WARN] 预热异常: {repr(e)}")

    global_stats = GlobalStats()
    window_stats = StatWindow(maxlen=args.window)

    csv_fp = None
    if args.csv:
        csv_fp = open(args.csv, "w", encoding="utf-8")
        write_csv_header(csv_fp)

    stop = False
    submitted = 0

    def submit_one(executor):
        nonlocal submitted
        fpath = random.choice(wavs)
        fut = executor.submit(
            worker_task,
            fpath,
            args.url,
            args.model,
            args.lang_id,
            args.text_norm,
            args.timeout,
            1,  # 重试 1 次
        )
        submitted += 1
        return fut

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = set()
            for _ in range(args.concurrency):
                futures.add(submit_one(ex))

            completed = 0
            last_print = 0

            while True:
                done, _ = wait_any(futures)
                for fut in done:
                    futures.remove(fut)
                    ok, lat, dur, _, err, fpath = fut.result()

                    global_stats.add(ok, lat, dur)
                    if ok:
                        window_stats.push(lat, dur)

                    if csv_fp:
                        ts = int(time.time())
                        rtf = (lat / dur) if (ok and dur > 0) else 0.0
                        write_csv_row(csv_fp, ts, ok, fpath, lat, dur, rtf, err)
                        if (completed + 1) % 20 == 0:
                            csv_fp.flush()

                    completed += 1

                    if completed - last_print >= args.stats_interval:
                        last_print = completed
                        la, du, rtfs = window_stats.snapshot()
                        print_stats("[WINDOW]", la, du, rtfs)

                        tr, okc, errc, tlat, tdur, buckets = global_stats.snapshot()
                        avg_lat = (tlat / okc) if okc > 0 else 0.0
                        avg_dur = (tdur / okc) if okc > 0 else 0.0
                        avg_rtf = (avg_lat / avg_dur) if avg_dur > 0 else 0.0
                        print(
                            f"[GLOBAL] total={tr}, ok={okc}, err={errc} | "
                            f"avg_lat={format_ms(avg_lat)}, avg_dur={avg_dur:.2f}s, avg_RTF={avg_rtf:.3f} | "
                            f"buckets={buckets}"
                        )

                    if args.max_requests > 0 and completed >= args.max_requests:
                        stop = True

                    if not stop:
                        futures.add(submit_one(ex))

                if stop and not futures:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在等待已提交任务完成...")
    finally:
        if csv_fp:
            csv_fp.flush()
            csv_fp.close()

    la, du, rtfs = window_stats.snapshot()
    print_stats("[FINAL WINDOW]", la, du, rtfs)
    tr, okc, errc, tlat, tdur, buckets = global_stats.snapshot()
    avg_lat = (tlat / okc) if okc > 0 else 0.0
    avg_dur = (tdur / okc) if okc > 0 else 0.0
    avg_rtf = (avg_lat / avg_dur) if avg_dur > 0 else 0.0
    print(
        f"[FINAL GLOBAL] total={tr}, ok={okc}, err={errc} | "
        f"avg_lat={format_ms(avg_lat)}, avg_dur={avg_dur:.2f}s, avg_RTF={avg_rtf:.3f} | "
        f"buckets={buckets}"
    )


if __name__ == "__main__":
    """使用示例： python triton_stability_test.py \ 
                            --folder /path/to/wavs \ 
                            --url http://127.0.0.1:8001 \ 
                            --model sensevoice \ 
                            -n 32 \ 
                            --timeout 20 \ 
                            --stats-interval 50 \ 
                            --window 1000 \ 
                            --csv result_detail.csv  """
    main()
