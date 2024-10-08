import torch
from collections import OrderedDict

from ..utils import get_root_logger
from ..utils.registry import MODEL_REGISTRY

from .base_sisr_model import BaseSISRModel
from .networks import build_archs
from .losses import build_loss, get_refined_artifact_map


@MODEL_REGISTRY.register()
class SISRNETModel(BaseSISRModel):
    """Base model for single image super-resolution."""

    def __init__(self, opt):
        super(SISRNETModel, self).__init__(opt)
        self.lq, self.gt = None, None
        self.log_dict = None
        self.ema_decay = opt.get('ema_decay', 0.999)
        # set up models
        self._setup_model()
        # set optimizers and schedulers
        self._setup_optimizers()
        self._setup_loss()
        self.net.train()

    """Optimize Functions."""
    def optimize_parameters(self, current_iter):
        # optimize net
        for p in self.net.parameters():
            p.requires_grad = True
        self.optimizer.zero_grad()
        self.output = self.net(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_ema(self.lq)
        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_g_pix'] = l_pix
        if self.cri_ldl:  # added in basicsr project, but not in origin realesrgan project
            pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
            l_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
            l_total += l_ldl
            loss_dict['l_g_ldl'] = l_ldl
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_g_percep'] = l_percep
            if l_g_style is not None:
                l_total += l_g_style
                loss_dict['l_g_style'] = l_g_style
        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    """Public Overwrite Functions."""
    def feed_data(self, data):
        self.lq = data['lq']
        if 'gt' in data:
            self.gt = data['gt']

    def model_ema(self, decay=0.999):
        net = self.get_bare_model(self.net)
        net_params = dict(net.named_parameters())
        net_ema_params = dict(self.net_ema.named_parameters())
        for k in net_ema_params.keys():
            net_ema_params[k].data.mul_(decay).add_(net_params[k].data, alpha=1 - decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_ema'):
            self.save_network([self.net, self.net_ema], 'net', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net, 'net', current_iter)
        self.save_training_state(epoch, current_iter)

    def test(self):
        net = self.net_ema if hasattr(self, 'net_ema') else self.net
        net.eval()
        with torch.no_grad():
            self.output = net(self.lq)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    """Private Setup Functions"""
    def _setup_loss(self):
        self.cri_pix = build_loss(self.opt['loss_pixel']) if self.opt.get('loss_pixel') else None
        self.cri_perceptual = build_loss(self.opt['loss_perceptual']) if self.opt.get('loss_perceptual') else None
        self.cri_ldl = build_loss(self.opt['loss_ldl']) if self.opt.get('loss_ldl') else None
        if self.cri_ldl is None and self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def _setup_model(self):
        # define network
        self.net_ema = build_archs(self.opt['network_g'])
        self.net = build_archs(self.opt['network_g'])
        self.print_network(self.net)
        # load pretrained schedule
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.model_ema(0)  # copy net_g weight
        self.net_ema.eval()

    def _setup_optimizers(self):
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = self.opt['optim_g'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **self.opt['optim_g'])
        self.optimizers.append(self.optimizer)


