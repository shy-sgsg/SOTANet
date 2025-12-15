# 修改记录

## 2025.12.13

1. 增加 SN，防止 D 训练过快
2. 添加一个新的 loss 模块并更新 pix2pix_model.py：实现 Charbonnier 和 Huber 损失，并在命令行选项中加入三选一配置：
    - rec_loss（可选：L1|Charbonnier|Huber，默认 L1）
    - charb_eps（Charbonnier 的 eps，默认 1e-3）
    - huber_delta（Huber 的 delta，默认 1.0）
我将查看 train.py 和选项加载相关文件来准确说明参数的解析流程。之后我会描述从命令行到最终使用 self.criterionL1 的完整链路。

Read train.py

Read train_options.py

Read __init__.py

下面把命令行参数从输入到被使用的完整链路一步步说明（包含你刚新增的 --rec_loss、--charb_eps、--huber_delta），并指出相关代码位置和要点。

要点概览

命令行解析入口: train.py → TrainOptions().parse()
模型追加参数: 模型类的静态方法 modify_commandline_options（我们在 pix2pix_model.py 中修改）负责添加 --rec_loss、--charb_eps、--huber_delta。
创建损失实例: 在 Pix2PixModel.__init__ 里用 get_rec_loss(...)（位于 losses.py）构造具体的损失模块并赋给 self.criterionL1。
使用损失: 训练时在 Pix2PixModel.backward_G() 中以 self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 计算重建损失。
详细流程（逐步、带文件/函数位置）

train.py 启动并触发解析
文件/位置: train.py，主流程顶部：
执行 opt = TrainOptions().parse()，进入选项解析链。
TrainOptions / BaseOptions 负责组合全局、训练、模型和数据集选项
文件/位置:
train_options.py -> TrainOptions.initialize()（添加训练相关参数）
base_options.py -> BaseOptions.gather_options() / BaseOptions.parse()（主解析逻辑）
关键逻辑（简述）:
先用 BaseOptions.initialize() 创建基础 parser 并添加基础参数。
使用 parser.parse_known_args() 取得初始 opt（包含 --model）。
根据 opt.model 调用 models.get_option_setter(model_name)，得到该模型的 modify_commandline_options（即 pix2pix_model.Pix2PixModel.modify_commandline_options）。
调用该方法把模型特定的额外参数追加到 parser 上（这一步就是我们新增参数会被注册的地方）。
再次 parse 以获得完整的 opt，并继续处理数据集-specific 的参数。
我在哪儿添加了新的参数
文件/位置: pix2pix_model.py 的 modify_commandline_options（我们已修改）
新增参数（已添加到 parser）:
--rec_loss : choices L1|Charbonnier|Huber（默认 "L1"）
--charb_eps : float，Charbonnier 的 epsilon（默认 1e-3）
--huber_delta : float，Huber 的 delta（默认 1.0）
因此，当你运行
argparse 会把这些参数解析进 opt.rec_loss, opt.charb_eps, opt.huber_delta。
何时把参数变为损失实例（构造具体的损失）
文件/位置: pix2pix_model.py 的 __init__
我做的改动: 在训练分支（if self.isTrain:）增加：
self.criterionL1 = get_rec_loss(opt.rec_loss, eps=opt.charb_eps, delta=opt.huber_delta)
get_rec_loss 的实现位置：losses.py，它会返回：
nn.L1Loss() （当 L1）
CharbonnierLoss(eps) （当 Charbonnier）
HuberLoss(delta) （当 Huber）
因此解析好的 opt.* 参数在模型初始化时立即被用来构造并绑定到 self.criterionL1。
损失在哪里被使用
文件/位置: pix2pix_model.py 的 backward_G()：
计算： self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
注意：变量名仍为 criterionL1 / lambda_L1，但现在 criterionL1 可以是 L1、Charbonnier 或 Huber（保持兼容原代码的调用点不变）。
其他说明、调试与验证
查看已注册的参数：运行 python [train.py](http://_vscodecontentref_/38) --help 会显示 --rec_loss, --charb_eps, --huber_delta（因为它们在 model 的 modify_commandline_options 被注册到 parser 上）。
参数生效时机：opt 在 TrainOptions().parse() 返回，models.create_model(opt) 调用模型构造函数，Pix2PixModel.__init__ 使用 opt 创建实际损失实例。也就是说，参数在训练开始前就固定了（如果想中途改变需要重新启动训练或实现可热更逻辑）。
文件改动回顾（你可以打开查看）:
losses.py（新增）：包含 CharbonnierLoss, HuberLoss, get_rec_loss
pix2pix_model.py（修改）：在 modify_commandline_options 中注册参数；在 __init__ 中用 get_rec_loss 创建 self.criterionL1
