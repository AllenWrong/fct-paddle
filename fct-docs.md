# Forward Compatible Training for Large-Scale Embedding Retrieval Systems
论文出处：[https://arxiv.org/abs/2112.02805](https://arxiv.org/abs/2112.02805)

## 目录
* [1. 原理介绍](#1-原理介绍)
* [2. 精度指标](#2-精度指标)
* [3. 数据准备](#3-数据准备)
* [4. 模型训练](#4-模型训练)
* [5. 模型评估与推理部署](#5-模型评估与推理部署)
* [5.1 模型评估](#51-模型评估)
* [5.2 模型推理](#52-模型推理)
* * [5.2.1 推理模型准备](#521-推理模型准备)
* * [5.2.2 基于Python预测引擎推理](#522-基于-python-预测引擎推理)
* * [5.2.3 基于C++预测引擎推理](#523-基于c预测引擎推理)
* [5.4 服务化部署](#54-服务化部署)
* [5.5 端侧部署](#55-端侧部署)
* [5.6 Paddle2ONNX模型转换与预测](#56-paddle2onnx-模型转换与预测)
* [6. 参考文献](#6-参考资料)

## 1. 原理介绍
作者提出了一种新的学习范式，称为大规模嵌入检索系统的前向兼容训练(FCT)。FCT旨在解决在更新嵌入模型时为每个数据项重新计算特征的昂贵过程。为了避免重新计算特征的昂贵过程，作者提出了使用一个附加信息模型，来使得旧特征对新特征的兼容。附加信息模型使用自监督的方式进行训练以提取任务无关的特征，此外还训练一个mixer模型，用来将附加信息和旧特征进行混合以兼容新特征。

## 2. 精度指标
以下表格总结了复现的FCT在ImageNet数据集上的精度指标。

| PyTorch Case | CMC Top-1(%) | CMC Top-5(%) | mAP(%) |
| :----------: | :----------: | :----------: | :----: |
|   old/old    |     46.4     |     65.1     |  28.3  |
|   new/new    |     68.4     |     84.7     |  45.6  |
|   new/old    |     65.1     |     82.7     |  44.0  |

| Paddle Case | CMC Top-1(%) | CMC Top-5(%) | mAP(%) |
| :---------: | :----------: | :----------: | :----: |
|   old/old   |     46.5     |     64.8     |  28.4  |
|   new/new   |     67.7     |     83.9     |  45.2  |
|   new/old   |     64.6     |     82.1     |  44.0  |

ImageNet数据集上，Paddle版本的配置文件及训练好的模型如下表所示

|数据集|配置文件地址|模型下载链接|
|:----:|:----:|:----:|
|ImageNet old/old| [配置文件](../../../../ppcls/configs/ImageEmbSearch/fct_old.yaml) ||
|ImageNet new/new|[配置文件](../../../../ppcls/configs/ImageEmbSearch/fct_new.yaml)||
|ImageNet new/old|[配置文件](../../../../ppcls/configs/ImageEmbSearch/fct_transform.yaml)||

上表中的配置是基于8卡GPU训练的。
**接下来** 以 `fct_old.yaml` 配置和训练好的模型文件为例，展示在ImangeNet数据集上进行训练，测试，推理的过程。

## 3. 数据准备
Image数据在路径`./dataset/ILSVR2012/`下。

## 4. 模型训练
1. 执行以下命令开始训练

单卡训练执行以下命令

**old/old**

```
python tools/train.py -c ppcls/configs/ImageEmbSearch/fct_old_1_gpu.yaml
```

**new/new**

```
python tools/train.py -c ppcls/configs/ImageEmbSearch/fct_new_1_gpu.yaml
```

**new/old**



8卡训练执行以下操作

**old/old**

```
python -m paddle.distributed.launch --gpus='0,1,2,3' tools/train.py -c ppcls/configs/ImageEmbSearch/fct_old.yaml
```

**new/new**

```
python -m paddle.distributed.launch --gpus='0,1,2,3' tools/train.py -c ppcls/configs/ImageEmbSearch/fct_new.yaml
```

**new/old**

```
python -m paddle.distributed.launch --gpus='0,1,2,3' tools/train.py -c ppcls/configs/ImageEmbSearch/fct_transform.yaml
```

1. **查看训练日志和保存的模型参数文件** 训练过程中屏幕会实时打印loss等指标信息，同时会保存日志文件 `train.log` ，模型参数文件 `*.pdparams`，优化器参数文件 `*.pdopt` 等内容到`Global.output_dir`指定的文件夹下，默认在 `PaddleClas/output/RecModel/`文件夹下。

## 5. 模型评估与推理部署
### 5.1 模型评估
准备用于评估的 `*.pdparams` 模型参数文件，可以使用训练好的模型，也可以使用 *4. 模型训练* 中保存的模型。
* 以训练过程中保存的 `best_model_ema.ema.pdparams`为例，执行如下命令即可进行评估。
```
python3.7 tools/eval.py -c ppcls/configs/ssl/CCSSL/FixMatchCCSSL_cifar10_4000_4gpu.yaml -o Global.pretrained_model="./output/RecModel/best_model_ema.ema"
```

* 以训练好的模型为例，下载提供的已经训练好的模型，到 `PaddleClas/pretrained_models` 文件夹中，执行如下命令即可进行评估。
```
# 下载模型
cd PaddleClas
mkdir pretrained_models
cd pretrained_models
wget 
cd ..
# 评估
python tools/eval.py -c ppcls/configs/ssl/CCSSL/FixMatchCCSSL_cifar10_4000_4gpu.yaml -o Global.pretrained_model="./output/RecModel/best_model_ema.ema"
```
**注：** `pretrained_model` 后填入的地址不需要加 `.pdparams`后缀，在程序运行时会自动补上。

* 查看输出结果
```
[2023/01/02 03:07:48] ppcls INFO: [Eval][Epoch 0][Iter: 0/157]CELoss: 0.01224, loss: 0.01224, top1: 1.00000, top5: 1.00000, batch_cost: 4.57323s, reader_cost: 0.76991, ips: 13.99447 images/sec
[2023/01/02 03:07:48] ppcls INFO: [Eval][Epoch 0][Iter: 20/157]CELoss: 0.05035, loss: 0.05035, top1: 0.95759, top5: 0.99851, batch_cost: 0.02510s, reader_cost: 0.00009, ips: 2549.51698 images/sec
[2023/01/02 03:07:49] ppcls INFO: [Eval][Epoch 0][Iter: 40/157]CELoss: 0.02832, loss: 0.02832, top1: 0.95541, top5: 0.99848, batch_cost: 0.02364s, reader_cost: 0.00008, ips: 2707.22687 images/sec
[2023/01/02 03:07:49] ppcls INFO: [Eval][Epoch 0][Iter: 60/157]CELoss: 0.05375, loss: 0.05375, top1: 0.95569, top5: 0.99898, batch_cost: 0.02209s, reader_cost: 0.00009, ips: 2897.88691 images/sec
[2023/01/02 03:07:50] ppcls INFO: [Eval][Epoch 0][Iter: 80/157]CELoss: 0.02459, loss: 0.02459, top1: 0.95872, top5: 0.99904, batch_cost: 0.02318s, reader_cost: 0.00009, ips: 2761.57735 images/sec
[2023/01/02 03:07:50] ppcls INFO: [Eval][Epoch 0][Iter: 100/157]CELoss: 0.06381, loss: 0.06381, top1: 0.95777, top5: 0.99876, batch_cost: 0.02258s, reader_cost: 0.00009, ips: 2834.16342 images/sec
[2023/01/02 03:07:51] ppcls INFO: [Eval][Epoch 0][Iter: 120/157]CELoss: 0.01684, loss: 0.01684, top1: 0.95713, top5: 0.99884, batch_cost: 0.02253s, reader_cost: 0.00009, ips: 2841.09327 images/sec
[2023/01/02 03:07:51] ppcls INFO: [Eval][Epoch 0][Iter: 140/157]CELoss: 0.05013, loss: 0.05013, top1: 0.95667, top5: 0.99889, batch_cost: 0.02238s, reader_cost: 0.00009, ips: 2860.07617 images/sec
[2023/01/02 03:07:51] ppcls INFO: [Eval][Epoch 0][Avg]CELoss: 0.15216, loss: 0.15216, top1: 0.95730, top5: 0.99890
```
默认评估日志保存在 `PaddleClas/output/RecModel/eval.log`中，可以看到我们提供的模型在cifar10数据集上的评估指标为top1: 95.57, top5: 99.95

### 5.2 模型推理
#### 5.2.1 推理模型准备
将训练过程中保存的模型文件转成inference模型，同样以 `best_model_ema.ema_pdparams`为例，执行以下命令进行转换
```
python3.7 tools/export_model.py \
-c ppcls/configs/ssl/CCSSL/FixMatchCCSSL_cifar10_4000_4gpu.yaml \
-o Global.pretrained_model="output/RecModel/best_model_ema.ema" \
-o Global.save_inference_fir="./deploy/inference"
```

#### 5.2.2 基于 Python 预测引擎推理
1. 修改 `PaddleClas/deploy/configs/inference_cls.yaml`
* * 将`infer_imgs:` 后的路径段改为 query 文件夹下的任意一张图片路径（下方配置使用的是`demo.jpg`图片的路径）
* * 将`inference_model_dir:` 后的字段改为解压出来的 inference 模型文件夹路径
* * 将`transform_ops:` 字段下的预处理配置改为 `FixMatch_CCSSL_cifar10_40000_4gpu.yaml` 中 `Eval.dataset`下的预处理配置
```
Global:
  infer_imgs: "./images/ImageNet/ILSVRC2012_val_00000010.jpeg"
  inference_model_dir: "../inference"
  batch_size: 1
  use_gpu: True
  enable_mkldnn: True
  cpu_num_threads: 10
  enable_benchmark: True
  use_fp16: False
  ir_optim: True
  use_tensorrt: False
  gpu_mem: 8000
  enable_profile: False

PreProcess:
  transform_ops:
    - ResizeImage:
        size: [32, 32]
        backend: pil
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.4914, 0.4822, 0.4465]
        std: [0.2471, 0.2435, 0.2616]
        order: hwc
    - ToCHWImage:

PostProcess:
  main_indicator: Topk
  Topk:
    topk: 5
```

2. 执行推理命令
```
cd ./deploy/
python3.7 python/predict_cls.py -c ./configs/inference_cls.yaml
```

3. 查看输出结果，实际结果为一个长度为5的向量，表示图像分类的结果，如
```
ILSVRC2012_val_00000010.jpeg:   class id(s): [3, 5, 2, 6, 0], score(s): [6.16, 3.26, 0.02, -0.26, -0.76], label_name(s): []
```

#### 5.2.3 基于C++预测引擎推理
PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考基于 Visual Studio 2019 Community CMake 编译指南完成相应的预测库编译和模型预测工作。

### 5.4 服务化部署
Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考Paddle Serving 代码仓库。

## 5.5 端侧部署
Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考Paddle Lite 代码仓库。

PaddleClas 提供了基于 Paddle Lite 来完成模型[端侧部署](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/paddle_lite.md)的示例，您可以参考端侧部署来完成相应的部署工作。

## 5.6 Paddle2ONNX 模型转换与预测
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考Paddle2ONNX 代码仓库。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/paddle2onnx.md) 模型转换与预测来完成相应的部署工作。

## 6. 参考资料
1. [CCSSL](https://arxiv.org/abs/2203.02261)