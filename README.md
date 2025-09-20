# Triton-SenseVoice-Paraformer-Streaming



## 使用Triton部署SenseVoice和Paraformer（包含流式）模型



## 简介

Triton Inference Server（简称 Triton Server）是 NVIDIA 开源的一款高性能推理服务框架，专门用来在生产环境中部署和管理机器学习/深度学习模型。它可以统一管理多种框架训练出的模型，并通过标准化的 API 提供推理服务。
本项目介绍如何使用Triton部署**SenseVoice**、**Paraformer**和**Paraformer-Streaming**模型，并提供使用**HTTP**接口请求模型的demo，对于流式模型，提供了从麦克风读取音频流的demo。



## 1 SenseVoice-Triton部署

### 1.1 文件结构

```shell
model_repo_sense_voice_small
|-- encoder
|   |-- 1
|       └── model.onnx -> sensevoice模型，backend为onnxruntime
|   |-- config.pbtxt -> 这里定义了模型的输入输出，动态批处理和实例组配置
|-- feature_extractor
|   |-- 1
|       └── model.py
|   |-- am.mvn
|   |-- config.pbtxt -> 这里有am.mvn和config.yaml的路径，请修改为你自己的路径，同样也有动态批处理和实例组配置
|   |-- config.yaml
|-- scoring
|   |-- 1
|       └── model.py
|   |-- chn_jpn_yue_eng_ko_spectok.bpe.model
|   |-- config.pbtxt -> 这里有chn_jpn_yue_eng_ko_spectok.bpe.model的路径
|-- sensevoice
    |-- 1 -> 这是一个空文件夹
    |-- config.pbtxt -> 这里是整个模型处理流程的组合，有客户端调用的modelname、input和output等
```

### 1.2 启动命令

```shell
tritonserver --model-repository /your/path/model_repo_sensevoice_small \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000
```

### 1.3 说明

1.上述`am.mvn`、`config.yaml`、`model.onnx 或者 model-quant.onnx`可在[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall-onnx)中下载。

2.`model.onnx 或者 model-quant.onnx`还可以自己训练后，通过[FunASR](https://github.com/modelscope/FunASR)导出为onnx格式。

3.`chn_jpn_yue_eng_ko_spectok.bpe.model`可在[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall)中下载。

4.关于triton的启动命令，可在[这里](https://www.cnblogs.com/zzk0/p/15932542.html)查看。

### 1.4 客户端请求

```shell
python client_demo_sensevoice.py
```

其中的`text_norm`和`language`取值参数可以在[SenseVoice:model.py](https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py)中找到，常用如下：

```shell
text_norm = {"withitn": 14, "woitn": 15}
language = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
```



## 2 Paraformer-Triton部署

### 2.1 文件结构

```shell
model_repo_paraformer_large_offline
|-- encoder
|   |-- 1
|   |   └── model.onnx
|   |-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   └── model.py
|   |-- config.pbtxt
|   |-- am.mvn
|   |-- config.yaml
|-- infer_pipeline
|   |-- 1
|   |-- config.pbtxt
|-- scoring
    |-- 1
    |   |-- model.py
    |   |-- tokens.json
    |-- config.pbtxt
```

### 2.2 启动命令

```shell
tritonserver --model-repository /your/path/model_repo_paraformer_large_offline \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000
```

### 2.3 说明

1.`model.onnx`、`am.mvn`、`config.yaml`、`tokens.json`可以在[ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx)获得。

2.`model.onnx`也可自己训练后，通过[FunASR](https://github.com/modelscope/FunASR)导出为onnx格式。

### 2.4 客户端请求

```shell
python client_demo_paraformer.py
```

具体参数可以在`infer_pipeline`下的`config.pbtxt`中获取。



## 3 Paraformer-Streaming-Triton部署

### 3.1 文件结构

```shell
model_repo_paraformer_large_online
    ├── cif_search
    │   ├── 1
    │   │   └── model.py
    │   └── config.pbtxt -> 这里有feature_extractor中config.yaml的路径
    ├── decoder
    │   ├── 1
    │   │   └── decoder.onnx
    │   └── config.pbtxt
    ├── encoder
    │   ├── 1
    │   │   └── model.onnx
    │   └── config.pbtxt
    ├── feature_extractor
    │   ├── 1
    │   │   └── model.py
    │   ├── config.pbtxt -> 这里有feature_extractor中config.yaml的路径
    │   └── config.yaml
    ├── lfr_cmvn_pe
    │   ├── 1
    │   │   └── lfr_cmvn_pe.onnx
    │   ├── am.mvn
    │   ├── config.pbtxt
    │   └── export_lfr_cmvn_pe_onnx.py
    └── streaming_paraformer
        ├── 1
        └── config.pbtxt
```

### 3.2 启动命令

```shell
tritonserver --model-repository /your/path/model_repo_paraformer_large_online \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000
```

### 3.3 说明

1.`decoder.onnx`、`model.onnx`、`config.yaml`、`am.mvn`可以在[ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx)获得。

2.`lfr_cmvn_pe.onnx`需要通过`export_lfr_cmvn_pe_onnx.py`获得，你只需要在`lfr_cmvn_pe`新建一个名为`1`的文件夹，然后

```shell
python export_lfr_cmvn_pe_onnx.py
```

`lfr_cmvn_pe.onnx`会转换完成。

3.同样的，`model.onnx`可以通过自己训练后，由[FunASR](https://github.com/modelscope/FunASR)导出。

### 3.4 客户端请求

```shell
单个音频请求
python client_demo_paraformer_streaming.py
由麦克风输入
python client_demo_paraformer_streaming_microphone.py
```

需要说明的是，在流式模型的请求中**Triton Inference Server 的“sequence batching”/“stateful model”机制** 提供了客户端控制字段，以帮助Triton**跨请求维持状态**。他们分别是：

- `sequence_id`：用来标识同一个会话/音频流的所有请求，保证它们路由到同一个模型实例。
- `sequence_start`：标记这个序列的第一帧请求（Triton 会重置内部状态）。
- `sequence_end`：标记这个序列的最后一帧请求（Triton 可以清理内部缓存状态）。

换句话说，这些参数告诉 Triton：

```shell
“我现在要开始一段新的流式推理（start），这是这段推理的第 N 个 chunk，最后一个 chunk 到了要结束（end）。
```

这写参数并不出现在`config.pbtxt` 中，`config.pbtxt` 描述的是 **模型输入/输出的 tensor 接口**，更多信息请移步[Sequence Extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_sequence.html)，你会看到这些参数的说明。
