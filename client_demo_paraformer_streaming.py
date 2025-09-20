#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8001"  # 改为你自己的地址
MODEL_NAME = "streaming_paraformer"
WAV_PATH = '/Users/youtwometoo/Downloads/8_minutes_ac_1.wav'

TARGET_SR = 16000
CHUNK_MS = 480  # 每 480 ms 发送一次
REALTIME = False  # 是否等待 CHUNK_MS 再发下一块


def load_wav_mono_float32(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_to_16k(x, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return x
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)


def dc_remove_and_peak_norm(x, peak=0.95, eps=1e-8):
    x = x - np.mean(x)
    mx = np.max(np.abs(x)) + eps
    return (x / mx * peak).astype(np.float32, copy=False)


def longest_common_prefix(a: str, b: str) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def print_incremental(new_text: str, last_text: str) -> str:
    if new_text is None:
        return last_text
    if not last_text:
        print(new_text, end="", flush=True)
        return new_text
    i = longest_common_prefix(last_text, new_text)
    if i < len(new_text):
        print(new_text[i:], end="", flush=True)
    return new_text


def stream_infer(client, wav_f32, chunk_ms=CHUNK_MS):
    sr = TARGET_SR
    chunk_size = int(sr * (chunk_ms / 1000.0))
    n = len(wav_f32)
    seq_id = (time.time_ns() & 0x7FFFFFFFFFFFFFFF) or 1

    last_text = ""  # 已打印的累计文本
    t0 = time.time()

    offset = 0
    chunk_index = 0
    while offset < n:
        end = min(offset + chunk_size, n)
        chunk = wav_f32[offset:end]
        offset = end
        chunk_index += 1

        wav = np.ascontiguousarray(chunk[np.newaxis, :], dtype=np.float32)  # (1, T_chunk)
        wav_lens = np.asarray([[wav.shape[1]]], dtype=np.int32)  # (1, 1)

        inp_wav = httpclient.InferInput("WAV", list(wav.shape), "FP32")
        inp_wav.set_data_from_numpy(wav, binary_data=True)
        inp_len = httpclient.InferInput("WAV_LENS", list(wav_lens.shape), "INT32")
        inp_len.set_data_from_numpy(wav_lens, binary_data=True)
        out_trans = httpclient.InferRequestedOutput("TRANSCRIPTS", binary_data=False)

        is_first = (offset - wav.shape[1] == 0)
        is_last = (offset >= n)

        try:
            result = client.infer(
                model_name=MODEL_NAME,
                inputs=[inp_wav, inp_len],
                outputs=[out_trans],
                sequence_id=seq_id,
                sequence_start=is_first,
                sequence_end=is_last,
                # timeout=30000,  # 可按需打开
            )
        except Exception as e:
            print(f"\n[ERROR] chunk#{chunk_index} infer failed: {e}")
            break

        resp = result.get_response()
        new_text = None
        for out in resp.get("outputs", []):
            if out.get("name") == "TRANSCRIPTS":
                data = out.get("data")
                if data:
                    new_text = data[0] if isinstance(data[0], str) else str(data[0])
                break

        last_text = print_incremental(new_text, last_text)

        if REALTIME and not is_last:
            target_next = t0 + chunk_index * (chunk_ms / 1000.0)
            now = time.time()
            if target_next > now:
                time.sleep(target_next - now)

    print("")


def main():
    wav, sr = load_wav_mono_float32(WAV_PATH)
    wav = resample_to_16k(wav, sr, TARGET_SR)
    wav = dc_remove_and_peak_norm(wav, peak=0.95)

    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    stream_infer(client, wav, chunk_ms=CHUNK_MS)


if __name__ == "__main__":
    while 1:
        main()
