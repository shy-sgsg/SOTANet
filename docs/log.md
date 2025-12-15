# 修改记录

## 2025.12.13

1. 增加 SN，防止 D 训练过快
2. 添加一个新的 loss 模块并更新 pix2pix_model.py：实现 Charbonnier 和 Huber 损失，并在命令行选项中加入三选一配置：
    - rec_loss（可选：L1|Charbonnier|Huber，默认 L1）
    - charb_eps（Charbonnier 的 eps，默认 1e-3）
    - huber_delta（Huber 的 delta，默认 1.0）

## 2025.12.15

> **引入边缘损失（梯度损失），并添加到总损失中。**

### 变更详情

- 新增文件: `models/losses.py`
  - `CharbonnierLoss`（已实现）
  - `HuberLoss`（已实现）
  - `get_rec_loss(...)`：工厂函数，返回 L1/Charbonnier/Huber
  - `GradientLoss`：使用 Sobel 卷积（按通道 depthwise conv）计算水平/垂直梯度，并在梯度图上应用可配置的重建损失（L1/Charbonnier/Huber）。

- 修改文件: `models/pix2pix_model.py`
  - 在 `modify_commandline_options` 中新增参数：
    - `--rec_loss` (`L1|Charbonnier|Huber`, 默认 `L1`)：像素重建损失类型。
    - `--charb_eps` (`float`, 默认 `1e-3`)：Charbonnier epsilon。
    - `--huber_delta` (`float`, 默认 `1.0`)：Huber delta。
    - `--use_grad_loss` (`flag`)：启用梯度（边缘）损失。
    - `--grad_loss` (`L1|Charbonnier|Huber`, 默认 `Charbonnier`)：梯度图上的损失类型。
    - `--lambda_Grad` (`float`, 默认 `10.0`)：梯度损失权重。
  - 在模型初始化中根据 `opt` 实例化 `self.criterionL1`（像素重建）和可选的 `self.criterionGrad`（梯度损失），并在 `backward_G` 中把 `loss_G_Grad` 加入总损失。

- 修改文件: `models/networks.py`
  - 判别器中的 `Conv2d` 已用 `spectral_norm(...)` 包裹，稳定对抗训练（Spectral Normalization）。

### 使用方法

- 查看已注册参数：

    ```bash
    python train.py --help
    ```

- 常见命令示例：
  - 默认（L1 重建，无梯度损失）:

      ```bash
      python train.py --dataroot /path/to/dataset --name exp_default --model pix2pix
      ```

  - 使用 Charbonnier 作为像素重建损失：

      ```bash
      python train.py --dataroot /path/to/dataset --name exp_charb --model pix2pix --rec_loss Charbonnier --charb_eps 1e-3 --lambda_L1 100
      ```

  - 启用梯度损失（在梯度图上使用 Charbonnier），并给较大权重：

      ```bash
      python train.py --dataroot /path/to/dataset --name exp_grad --model pix2pix --use_grad_loss --grad_loss Charbonnier --charb_eps 1e-3 --lambda_L1 100 --lambda_Grad 50
      ```

### 建议与说明

- `--lambda_L1` 仍作为重建损失总权重名使用（为了兼容旧参数）。如果你希望更语义化，可以改为 `--lambda_rec`，但当前实现保持兼容。
- `--lambda_Grad` 的推荐初值为 `1` 到 `100`，取决于 `--lambda_L1` 的大小（若 `lambda_L1=100`，可以从 `lambda_Grad=10` 或 `50` 开始实验）。
- 启用梯度损失后，训练日志中会出现 `G_Grad`（若 `use_grad_loss` 开启），可用以监控梯度项收敛情况。
- Sobel 卷积按通道独立计算（groups=C），这能保持彩色/多通道图像每通道的边缘响应。


