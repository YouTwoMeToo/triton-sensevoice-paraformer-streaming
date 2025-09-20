#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8001"  # 改为你自己的地址
MODEL_NAME = "sensevoice"
WAV_PATH = "/your/path/test.wav"
TARGET_SR = 16000


def load_wav_mono_float32(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def resample_to_16k(x, sr):
    if sr == TARGET_SR:
        return x
    from math import gcd
    g = gcd(sr, TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)


def dc_remove_and_peak_norm(x, peak=0.95, eps=1e-8):
    x = x - np.mean(x)
    mx = np.max(np.abs(x)) + eps
    return (x / mx * peak).astype(np.float32, copy=False)


def infer_once(client, wav_f32, lang_id: int, text_norm: int = 14):
    T = len(wav_f32)
    wav = np.ascontiguousarray(wav_f32[np.newaxis, :], dtype=np.float32)  # (1, T)
    wav_lens = np.asarray([[T]], dtype=np.int32)  # (1, 1)
    language = np.asarray([[lang_id]], dtype=np.int32)  # (1, 1)
    text_norm_arr = np.asarray([[text_norm]], dtype=np.int32)  # (1, 1)

    inp_wav = httpclient.InferInput("WAV", list(wav.shape), "FP32")
    inp_wav.set_data_from_numpy(wav, binary_data=True)

    inp_len = httpclient.InferInput("WAV_LENS", list(wav_lens.shape), "INT32")
    inp_len.set_data_from_numpy(wav_lens, binary_data=True)

    inp_lang = httpclient.InferInput("LANGUAGE", list(language.shape), "INT32")
    inp_lang.set_data_from_numpy(language, binary_data=True)

    inp_norm = httpclient.InferInput("TEXT_NORM", list(text_norm_arr.shape), "INT32")
    inp_norm.set_data_from_numpy(text_norm_arr, binary_data=True)

    out_trans = httpclient.InferRequestedOutput("TRANSCRIPTS", binary_data=False)

    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[inp_wav, inp_len, inp_lang, inp_norm],
        outputs=[out_trans]
    )
    resp = result.get_response()
    print(resp)
    text = None
    for out in resp.get("outputs", []):
        if out.get("name") == "TRANSCRIPTS":
            data = out.get("data")
            if data:
                text = data[0] if isinstance(data[0], str) else str(data[0])
            break
    return text


def main():
    wav, sr = load_wav_mono_float32(WAV_PATH)
    wav = resample_to_16k(wav, sr)
    wav = dc_remove_and_peak_norm(wav, peak=0.95)

    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    text = infer_once(client, wav, lang_id=0, text_norm=14)
    print(text)


if __name__ == "__main__":
    main()
