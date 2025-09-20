# triton-sensevoice-paraformer-streaming
## 使用triton部署sensevoice和paraformer（包含流式）模型
Triton Inference Server（简称 Triton Server）是 NVIDIA 开源的一款高性能推理服务框架，专门用来在生产环境中部署和管理机器学习/深度学习模型。它可以统一管理多种框架训练出的模型，并通过标准化的 API 提供推理服务。
本项目介绍如何使用triton部署SenseVoice、paraformer和Paraformer-streaming模型，并提供使用http接口请求模型的demo，对于流式模型，提供了从麦克风读取音频流，并送入模型的demo
## 1.SenseVoice-triton部署
### 1.1文件结构
model_repo_sense_voice_small
|-- encoder
|   |-- 1
|   |   `-- model.onnx -> /your/path/model.onnx
|   `-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   `-- model.py
|   |-- am.mvn
|   |-- config.pbtxt
|   `-- config.yaml
|-- scoring
|   |-- 1
|   |   `-- model.py
|   |-- chn_jpn_yue_eng_ko_spectok.bpe.model -> /your/path/chn_jpn_yue_eng_ko_spectok.bpe.model
|   `-- config.pbtxt
`-- sensevoice
    |-- 1
    `-- config.pbtxt
