import torch
import os
from .base_model import BaseModel
from . import networks
from .losses import get_rec_loss, GradientLoss, SSIMLoss, VGGFeatureLoss


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for reconstruction loss (L1 or alternatives)")
            parser.add_argument("--rec_loss", type=str, choices=["L1", "Charbonnier", "Huber"], default="L1", help="reconstruction loss to use: L1 | Charbonnier | Huber")
            parser.add_argument("--charb_eps", type=float, default=1e-3, help="epsilon for Charbonnier loss")
            parser.add_argument("--huber_delta", type=float, default=1.0, help="delta for Huber loss")
            parser.add_argument("--use_grad_loss", action="store_true", help="enable gradient (edge) loss using Sobel kernels")
            parser.add_argument("--grad_loss", type=str, choices=["L1", "Charbonnier", "Huber"], default="Charbonnier", help="loss type to use on gradient maps")
            parser.add_argument("--lambda_Grad", type=float, default=50.0, help="weight for gradient loss")
            parser.add_argument("--grad_downsample", type=int, default=1, help="downsample factor for gradient loss (1=no downsample, 2=2x, 4=4x)")
            parser.add_argument("--grad_mag_thresh", type=float, default=0.05, help="gradient magnitude threshold to form edge mask (on downsampled gradients)")
            parser.add_argument("--grad_density_window", type=int, default=0, help="local window size for gradient density suppression (0=disabled, e.g. 7)")
            parser.add_argument("--grad_density_thresh", type=float, default=0.5, help="density threshold (fraction) above which regions are suppressed for gradient loss")
            parser.add_argument("--use_canny_grad", action="store_true", help="enable Canny-based structural edge loss (medianBlur + Canny + DT)")
            parser.add_argument("--canny_median_ksize", type=int, default=3, help="median blur kernel size for Canny preprocessing (odd int)")
            parser.add_argument("--canny_scales", type=str, default="4,2,1", help="comma-separated downsample scales for multi-scale Canny DT (coarse->fine)")
            parser.add_argument("--canny_weights", type=str, default="0.6,0.3,0.1", help="comma-separated weights for scales (same order as --canny_scales)")
            parser.add_argument("--canny_thresholds", type=str, default="50:150", help="semicolon-separated Canny threshold pairs (t1:t2;...), applied OR-combined")
            parser.add_argument("--canny_min_area", type=int, default=30, help="minimum connected component area (pixels) to keep in Canny edge map")
            parser.add_argument("--lambda_Canny", type=float, default=10.0, help="weight for Canny edge DT loss")
            parser.add_argument("--save_edge_vis", action="store_true", help="save edge visualization maps (real edges, dt, soft edges, grad mag) to disk")
            parser.add_argument("--edge_vis_dir", type=str, default="results/edge_vis", help="directory to save edge visualizations when --save_edge_vis is set")
            parser.add_argument("--use_ssim", action="store_true", help="enable SSIM loss (1 - SSIM)")
            parser.add_argument("--lambda_SSIM", type=float, default=1.0, help="weight for SSIM loss")
            parser.add_argument("--use_perc", action="store_true", help="enable perceptual (VGG shallow) loss")
            parser.add_argument("--lambda_Perc", type=float, default=1.0, help="weight for perceptual loss")
            parser.add_argument("--perc_layers", type=str, default="relu1_2,relu2_2", help="comma-separated VGG layers to use for perceptual loss (small set recommended)")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            # reconstruction loss (configurable): keep attribute name 'criterionL1' for backward compatibility
            self.criterionL1 = get_rec_loss(opt.rec_loss, eps=opt.charb_eps, delta=opt.huber_delta)
            # optional gradient (edge) loss
            if getattr(opt, "use_grad_loss", False):
                self.criterionGrad = GradientLoss(
                    loss_type=opt.grad_loss,
                    eps=opt.charb_eps,
                    delta=opt.huber_delta,
                    downsample=getattr(opt, "grad_downsample", 1),
                    density_window=getattr(opt, "grad_density_window", 0),
                    density_thresh=getattr(opt, "grad_density_thresh", 0.5),
                    mag_thresh=getattr(opt, "grad_mag_thresh", 0.05),
                )
                # add logging name
                if "G_Grad" not in self.loss_names:
                    self.loss_names.insert(1, "G_Grad")
            # optional Canny-based structural loss
            if getattr(opt, "use_canny_grad", False):
                # parse scales and weights and thresholds
                scales = [int(s) for s in opt.canny_scales.split(",") if s.strip()]
                weights = [float(w) for w in opt.canny_weights.split(",") if w.strip()]
                # parse threshold pairs like "50:150;30:90"
                th_pairs = []
                for seg in opt.canny_thresholds.split(";"):
                    seg = seg.strip()
                    if not seg:
                        continue
                    if ":" in seg:
                        a, b = seg.split(":")
                        th_pairs.append((int(a), int(b)))
                    else:
                        # single value => use as high threshold with low=high/3
                        v = int(seg)
                        th_pairs.append((v // 3, v))

                self.criterionCanny = None
                try:
                    self.criterionCanny = VGGFeatureLoss  # dummy to ensure name exists
                except Exception:
                    self.criterionCanny = None
                # import CannyEdgeLoss lazily from losses
                from .losses import CannyEdgeLoss

                self.criterionCanny = CannyEdgeLoss(
                    scales=scales,
                    weights=weights,
                    median_ksize=opt.canny_median_ksize,
                    canny_thresholds=th_pairs,
                    min_area=opt.canny_min_area,
                    device=self.device,
                )
                # enable visualization if requested
                if getattr(opt, "save_edge_vis", False):
                    vis_dir = getattr(opt, "edge_vis_dir", None)
                    if vis_dir is None:
                        vis_dir = os.path.join("results", "edge_vis")
                    # ensure absolute path under opt.results_dir if available
                    try:
                        import os as _os
                        base_dir = getattr(opt, "results_dir", None) or getattr(opt, "checkpoints_dir", None) or "."
                        if base_dir:
                            vis_dir = _os.path.join(base_dir, vis_dir)
                    except Exception:
                        pass
                    self.criterionCanny.enable_visualization(vis_dir)
                if "G_Canny" not in self.loss_names:
                    self.loss_names.insert(1, "G_Canny")
            # optional SSIM loss
            if getattr(opt, "use_ssim", False):
                self.criterionSSIM = SSIMLoss(window_size=11, sigma=1.5, data_range=1.0)
                if "G_SSIM" not in self.loss_names:
                    self.loss_names.insert(1, "G_SSIM")
            # optional perceptual loss (VGG shallow)
            if getattr(opt, "use_perc", False):
                perc_layers = [s.strip() for s in opt.perc_layers.split(",") if s.strip()]
                self.criterionPerc = VGGFeatureLoss(layers=perc_layers, device=self.device)
                if "G_Perc" not in self.loss_names:
                    self.loss_names.insert(1, "G_Perc")
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # optional gradient loss (edge loss)
        if hasattr(self, "criterionGrad"):
            self.loss_G_Grad = self.criterionGrad(self.fake_B, self.real_B) * self.opt.lambda_Grad
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Grad
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # optional SSIM loss
        if hasattr(self, "criterionSSIM"):
            self.loss_G_SSIM = self.criterionSSIM(self.fake_B, self.real_B) * self.opt.lambda_SSIM
            self.loss_G = self.loss_G + self.loss_G_SSIM
        # optional perceptual loss
        if hasattr(self, "criterionPerc"):
            self.loss_G_Perc = self.criterionPerc(self.fake_B, self.real_B) * self.opt.lambda_Perc
            self.loss_G = self.loss_G + self.loss_G_Perc
        # optional Canny-based loss
        if hasattr(self, "criterionCanny"):
            self.loss_G_Canny = self.criterionCanny(self.fake_B, self.real_B) * getattr(self.opt, "lambda_Canny", 10.0)
            self.loss_G = self.loss_G + self.loss_G_Canny
        # combine loss and calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
