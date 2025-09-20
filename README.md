# Triton-SenseVoice-Paraformer-Streaming

## 使用Triton部署SenseVoice和Paraformer（包含流式）模型

Triton Inference Server（简称 Triton Server）是 NVIDIA 开源的一款高性能推理服务框架，专门用来在生产环境中部署和管理机器学习/深度学习模型。它可以统一管理多种框架训练出的模型，并通过标准化的 API 提供推理服务。
本项目介绍如何使用triton部署SenseVoice、paraformer和Paraformer-streaming模型，并提供使用http接口请求模型的demo，对于流式模型，提供了从麦克风读取音频流的demo。

## 1 SenseVoice-Triton部署

### 1.1 文件结构

```sh
model_repo_sense_voice_small
|-- encoder
|   |-- 1
|       |-- model.onnx -> sensevoice模型，backend为onnxruntime
|   |-- config.pbtxt -> 这里定义了模型的输入输出，动态批处理和实例组配置
|-- feature_extractor
|   |-- 1
|       |-- model.py
|   |-- am.mvn
|   |-- config.pbtxt -> 这里有am.mvn和config.yaml的路径，请修改为你自己的路径，同样也有动态批处理和实例组配置
|   |-- config.yaml
|-- scoring
|   |-- 1
|       |-- model.py
|   |-- chn_jpn_yue_eng_ko_spectok.bpe.model
|   |-- config.pbtxt -> 这里有chn_jpn_yue_eng_ko_spectok.bpe.model的路径
|-- sensevoice
    |-- 1
    |-- config.pbtxt -> 这里是整个模型处理流程的组合，有客户端调用的modelname、input和output等
```

### 1.2 启动命令

```
tritonserver --model-repository /your/path/model_repo_sensevoice_small \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000
```

### 1.3 说明

1.上述`am.mvn`、`feature_extractor中的config.yaml`、`model.onnx 或者 model-quant.onnx`可在[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall-onnx)中下载。

2.`model.onnx 或者 model-quant.onnx`还可以自己训练后，通过[FunASR](https://github.com/modelscope/FunASR)导出为onnx格式。

3.`chn_jpn_yue_eng_ko_spectok.bpe.model`可在[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall)中下载。

4.关于triton的启动命令，可在[这里](https://www.cnblogs.com/zzk0/p/15932542.html)查看

### 1.4 客户端请求

```
python client_demo_sensevoice.py
```

其中的`text_norm`和`language`参数可以在[SenseVoice-model.py](https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py)中找到
