import torch
from collections import OrderedDict

from basicsr.utils.img_process_util import USMSharp

from ..utils import get_root_logger
from ..utils.registry import MODEL_REGISTRY

from .base_sisr_model import BaseSISRModel
from .networks import build_archs
from .losses import build_loss, get_refined_artifact_map


@MODEL_REGISTRY.register()
class SISRGANModel(BaseSISRModel):
    """SRGAN model for single image super-resolution."""

    def __init__(self, opt):
        super(SISRGANModel, self).__init__(opt)
        self.usm_sharpener = USMSharp()
        self.lq, self.gt, self.ref = None, None, None
        self.log_dict = None
        self.ema_decay = opt.get('ema_decay', 0.999)
        # set up models
        self._setup_model_gan()
        # set up loss, optimizers and schedulers
        self._setup_optimizers_gan()
        self._setup_loss()
        self.net_d.train()

    """Different Generator and Discriminator input."""
    def _generator_(self, ema=False):
        net = self.net_g_ema if ema else self.net_g
        if self.opt['model_reference_generator'] and self.ref is not None:
            output = net(self.lq, self.ref)
        else:
            output = net(self.lq)
        return output

    def _discriminator_(self, sr):
        if self.opt['model_semantic_discriminator']:
            fake_pred = self.net_d(sr, self.gt)
        else:
            fake_pred = self.net_d(sr)
        return fake_pred

    """Public Functions optimize_parameters Overwrite."""
    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self._generator_()
        self.output_ema = self._generator_(ema=True) if self.cri_ldl else None

        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:  # added in basicsr project, but not in origin realesrgan project
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self._discriminator_(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            l_g_total.backward()
            self.optimizer_g.step()
            # end if

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self._discriminator_(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self._discriminator_(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    """Public Overwrite Functions."""
    def feed_data(self, data):
        self.lq = data['lq']
        self.gt = data['gt']
        self.gt = self.usm_sharpener(self.gt) if self.opt.get('usm_sharp', False) else self.gt
        if 'ref' in data:
            self.ref = data['ref']

    def model_ema(self, decay=0.999):
        net = self.get_bare_model(self.net_g)
        net_params = dict(net.named_parameters())
        net_ema_params = dict(self.net_g_ema.named_parameters())
        for k in net_ema_params.keys():
            net_ema_params[k].data.mul_(decay).add_(net_params[k].data, alpha=1 - decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            self.output = self._generator_(ema=hasattr(self, 'net_g_ema'))

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    """Private Setup Functions"""
    def _setup_model_gan(self):
        self.net_g_ema = build_archs(self.opt['network_g'])
        self.net_g = build_archs(self.opt['network_g'])
        self.print_network(self.net_g)
        # load pretrained schedule
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params_ema')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        self.model_ema(0)  # copy net_g weight
        self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_archs(self.opt['network_d'])
        # self.print_network(self.net_d)

        # load pretrained schedule
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_d_iters = self.opt.get('net_d_iters', 1)
        self.net_d_init_iters = self.opt.get('net_d_init_iters', 0)

    def _setup_loss(self):
        self.cri_pix = build_loss(self.opt['loss_pixel']) if self.opt.get('loss_pixel') else None
        self.cri_perceptual = build_loss(self.opt['loss_perceptual']) if self.opt.get('loss_perceptual') else None
        self.cri_ldl = build_loss(self.opt['loss_ldl']) if self.opt.get('loss_ldl') else None
        self.cri_gan = build_loss(self.opt['loss_gan']) if self.opt.get('loss_gan') else None
        if self.cri_ldl is None and self.cri_gan is None and self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def _setup_optimizers_gan(self):
        # optimizer g
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = self.opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **self.opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_params = []
        for k, v in self.net_d.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = self.opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **self.opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
