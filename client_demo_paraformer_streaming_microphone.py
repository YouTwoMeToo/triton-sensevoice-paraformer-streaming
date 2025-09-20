#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import queue
import signal
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8001"  # 更换为你自己的地址
MODEL_NAME = "streaming_paraformer"

TARGET_SR = 16000
CHUNK_MS = 480  # 每 480 ms 发送一次
INPUT_CHANNELS = 1  # 采集单声道
INPUT_DEVICE = None  # 指定声卡设备ID（None 则用默认设备）
VERBOSE_CLIENT = False  # Triton httpclient 是否 verbose


def resample_to_16k(x, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return x.astype(np.float32, copy=False)
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


class MicStreamer:
    def __init__(self, samplerate=TARGET_SR, channels=INPUT_CHANNELS, block_ms=CHUNK_MS, device=INPUT_DEVICE):
        self.channels = channels
        self.device = device
        self.block_ms = block_ms
        self.target_sr = TARGET_SR
        self.capture_sr = samplerate
        self.block_size = int(self.capture_sr * (self.block_ms / 1000.0))
        self.q = queue.Queue(maxsize=32)
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        if self.channels > 1:
            mono = np.mean(indata, axis=1, dtype=np.float32)
        else:
            mono = indata[:, 0].astype(np.float32, copy=False)
        self.q.put_nowait(mono)

    def __enter__(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.capture_sr,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            blocksize=self.block_size,  # 设备以该帧数回调
            device=self.device,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def chunks(self):
        buf = np.zeros(0, dtype=np.float32)
        while self._running:
            try:
                piece = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if self.capture_sr != self.target_sr:
                piece = resample_to_16k(piece, self.capture_sr, self.target_sr)
            if buf.size == 0:
                buf = piece
            else:
                buf = np.concatenate([buf, piece], axis=0)
            chunk_len = int(self.target_sr * (self.block_ms / 1000.0))
            while buf.size >= chunk_len:
                chunk = buf[:chunk_len]
                buf = buf[chunk_len:]
                yield chunk


def infer_send_chunk(client, seq_id, chunk, is_first=False, is_last=False):
    chunk = dc_remove_and_peak_norm(chunk, peak=0.95)

    wav = np.ascontiguousarray(chunk[np.newaxis, :], dtype=np.float32)
    wav_lens = np.asarray([[wav.shape[1]]], dtype=np.int32)

    inp_wav = httpclient.InferInput("WAV", list(wav.shape), "FP32")
    inp_wav.set_data_from_numpy(wav, binary_data=True)

    inp_len = httpclient.InferInput("WAV_LENS", list(wav_lens.shape), "INT32")
    inp_len.set_data_from_numpy(wav_lens, binary_data=True)

    out_trans = httpclient.InferRequestedOutput("TRANSCRIPTS", binary_data=False)

    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[inp_wav, inp_len],
        outputs=[out_trans],
        sequence_id=seq_id,
        sequence_start=is_first,
        sequence_end=is_last,
    )

    new_text = None
    try:
        arr = result.as_numpy("TRANSCRIPTS")
        if arr is not None and arr.size > 0:
            v = arr[0]
            new_text = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
    except Exception:
        resp = result.get_response()
        for out in resp.get("outputs", []):
            if out.get("name") == "TRANSCRIPTS":
                data = out.get("data")
                if data:
                    new_text = data[0] if isinstance(data[0], str) else str(data[0])
                break
    return new_text


def send_end_packet(client, seq_id):
    tiny = np.zeros(1, dtype=np.float32)
    try:
        _ = infer_send_chunk(client, seq_id, tiny, is_first=False, is_last=True)
    except Exception:
        pass


def run_realtime():
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=VERBOSE_CLIENT)
    seq_id = (time.time_ns() & 0x7FFFFFFFFFFFFFFF) or 1
    stop_flag = {"stop": False}

    def _sigint(_sig, _frm):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint)

    last_text = ""
    first = True

    with MicStreamer(samplerate=TARGET_SR, channels=INPUT_CHANNELS, block_ms=CHUNK_MS, device=INPUT_DEVICE) as mic:
        print("🎙️ 开始采集麦克风音频，按 Ctrl+C 结束。\n")
        try:
            for chunk in mic.chunks():
                if stop_flag["stop"]:
                    break
                try:
                    new_text = infer_send_chunk(
                        client=client,
                        seq_id=seq_id,
                        chunk=chunk,
                        is_first=first,
                        is_last=False,
                    )
                    first = False
                    last_text = print_incremental(new_text, last_text)
                except Exception as e:
                    print(f"\n[ERROR] infer failed: {e}\n")
                    time.sleep(0.2)
        finally:
            send_end_packet(client, seq_id)
            print("\n\n✅ 已结束。")


if __name__ == "__main__":
    run_realtime()
