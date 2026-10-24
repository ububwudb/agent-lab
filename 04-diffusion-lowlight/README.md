# 04 · Diffusion 低光图像增强（补充项目）

> 这是**复现 + 调优**类项目，不是从零提出新方法。放在这里是为了说明：
> 我能读懂并跑通别人的研究代码，也能把推理过程工程化成一个可调用的接口。

## 来源

基于论文 **Low-light Image Enhancement with Wavelet-based Diffusion Models**（SIGGRAPH Asia 2023）
官方实现复现：

- 论文：https://arxiv.org/pdf/2306.00306.pdf
- 原始代码：WaveDiff（见论文作者仓库）

## 我做了什么

| 环节 | 内容 |
|---|---|
| 环境 | 在 Linux / Windows 双环境下搭 PyTorch 环境，跑通训练与推理脚本（含 CUDA 版本与依赖冲突排查） |
| 数据 | 按官方要求准备成对数据集（LOLv1 / LOLv2 等），调整 `datasets/dataset.py` 的路径与预处理 |
| 训练 | 跑通 `train.py`，调整 batch size / 学习率 / 训练轮次，观察 PSNR / SSIM 变化趋势 |
| 推理 | 用 `infer_single.py` / `infer_single_image.py` 对单图做增强，比对原图与增强结果 |
| 工程化 | 把推理过程包装成 HTTP 接口 + 简单网页，上传图片返回增强结果，做接口联调与交互调整 |

## 关键概念（能讲的部分）

- **为什么用 Diffusion 做低光增强**：低光图像的退化（噪声、色偏、细节丢失）很难用单一映射复原；
  扩散模型学习的是"从噪声恢复清晰图像"的分布，能生成更自然的纹理。
- **为什么加小波变换**：在频域把图像拆成低频（光照、颜色）和高频（细节、噪声）分别处理，
  避免"提亮的同时把噪声放大"。
- **和 GAN 方案的区别**：GAN 训练不稳定、容易产生伪影；Diffusion 训练稳定但推理慢（多步去噪），
  工程上要在质量和速度之间做权衡（减少采样步数、蒸馏）。

## 复现步骤（概要）

```bash
pip install -r requirements.txt
# 1) 下载预训练权重与数据集，按官方说明放到对应目录
# 2) 单图推理
python infer_single_image.py --input input/night.jpg --output output/night_enhanced.png
# 3) 训练（需要改 datasets/dataset.py 里的数据路径）
python train.py
```

## 说明

- 本目录**不包含**第三方论文的完整源码和模型权重（体积大且版权归原作者），
  只保留我在复现过程中写的推理/接口封装与说明；
- 结果图见 `assets/`（如有）；
- 这个项目的定位是"能读懂、能跑通、能包成服务"，不作为算法创新点来讲。

## 工程价值

生成式模型的实际使用经验（不是只会聊天），以及把模型能力封装成 HTTP 接口 + 网页的工程经验。
