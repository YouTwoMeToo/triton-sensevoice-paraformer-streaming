#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8001"     # 改为你的 Triton HTTP 地址与端口
MODEL_NAME = "infer_pipeline"      # 和 config.pbtxt 里的 name 一致
WAV_PATH = "/your/path/test.wav"
TARGET_SR = 16000


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


def infer_once(client: httpclient.InferenceServerClient, wav_f32: np.ndarray):
    T = len(wav_f32)
    wav = np.ascontiguousarray(wav_f32[np.newaxis, :], dtype=np.float32)  # (1, T)
    wav_lens = np.asarray([[T]], dtype=np.int32)                           # (1, 1)

    inp_wav = httpclient.InferInput("WAV", list(wav.shape), "FP32")
    inp_wav.set_data_from_numpy(wav, binary_data=True)

    inp_len = httpclient.InferInput("WAV_LENS", list(wav_lens.shape), "INT32")
    inp_len.set_data_from_numpy(wav_lens, binary_data=True)

    out_trans = httpclient.InferRequestedOutput("TRANSCRIPTS", binary_data=False)

    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[inp_wav, inp_len],
        outputs=[out_trans]
    )

    arr = result.as_numpy("TRANSCRIPTS")
    if arr is None:
        return None

    try:
        text = arr.flatten()[0]
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", errors="ignore")
        return str(text)
    except Exception:
        return None


def main():
    wav, sr = load_wav_mono_float32(WAV_PATH)
    wav = resample_to_16k(wav, sr)
    wav = dc_remove_and_peak_norm(wav, peak=0.95)
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    text = infer_once(client, wav)
    print(text)


if __name__ == "__main__":
    main()
