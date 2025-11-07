# 模型转换

## 导出模型（ONNX）
导出onnx可以参考代码：
```
import torch
from pathlib import Path
from deoldify.generators import gen_inference_wide , gen_inference_deep
learn = gen_inference_wide(
    root_folder=Path('.'),
    weights_name='ColorizeStable_gen'
)

model = learn.model.eval()   # 设置模型为评估模式
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model,
    dummy_input,
    "./colorize_stable.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
```

这里固定onnx输入尺寸为：1x3x512x512

## 动态onnx转静态
```
onnxsim colorize_stable.onnx  colorize_stable_sim.onnx --overwrite-input-shape=1,3,512,512
```

## 转换模型（ONNX -> Axera）
使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 量化数据集
准备量化npy文件，输入若干张图片经过预处理保存为npy文件

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：

```
pulsar2 build --input colorize_stable.onnx --config ./build_config.json --output_dir ./output --output_name rcolorize_stable.axmodel  --target_hardware AX650 --compiler.check 0

也可将参数写进json中，直接执行：
pulsar2 build --config ./build_config.json
```
