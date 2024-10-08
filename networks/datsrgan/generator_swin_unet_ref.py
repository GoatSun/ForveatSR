# Modified from https://github.com/caojiezhang/DATSR
# Reference-based Image Super-Resolution with Deformable Attention Transformer, https://arxiv.org/abs/2207.11938


import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.models.vgg as vgg

from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import basicsr.models.networks.arch_util as arch_util
from basicsr.utils.registry import ARCH_REGISTRY

from networks.swinirgan.arch_swinir import RSTB, PatchEmbed, PatchUnEmbed


from .dcn_v2 import DCNSepPreMultiOffsetFlowSimilarity as DynAgg
from .similarity_flow_correspondence import FlowSimCorrespondenceGenerationArch


class ContentExtractor(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, num_feat=nf)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)
        return feat


class ContrastExtractorLayer(nn.Module):
    def __init__(self):
        super(ContrastExtractorLayer, self).__init__()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg, 'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v

        self.model = nn.Sequential(modified_net)
        # the mean is for image with range [0, 1]
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


class ContrasExtractorSep(nn.Module):
    def __init__(self):
        super(ContrasExtractorSep, self).__init__()
        self.feature_extraction_image1 = ContrastExtractorLayer()
        self.feature_extraction_image2 = ContrastExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extraction_image1(image1)
        dense_features2 = self.feature_extraction_image2(image2)
        return {'dense_features1': dense_features1, 'dense_features2': dense_features2}


class SwinBlock(nn.Module):
    def __init__(self,
                 img_size=40,
                 patch_size=1,
                 embed_dim=180,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 **kwargs):
        super(SwinBlock, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        # x: [1, 60, 264, 184]
        x_size = torch.tensor(x.shape[2:4])  # [264, 184]
        x = self.patch_embed(x)  # [1, 48576, 60]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # [1, 48576, 60]
        for layer in self.layers:
            x = layer(x, x_size)  # [1, 48576, 60]
        x = self.norm(x)  # b seq_len c       # [1, 48576, 60]
        x = self.patch_unembed(x, x_size)  # [1, 60, 264, 184]
        return x


@ARCH_REGISTRY.register()
class SwinUnetv3RestorationNet(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, groups=8, depths=(8, 8), num_heads=(8, 8), window_size=8, use_checkpoint=False, path=None):
        super(SwinUnetv3RestorationNet, self).__init__()
        self.net_extractor = ContrasExtractorSep()
        self.net_extractor.load_state_dict(torch.load(path))

        self.net_map = FlowSimCorrespondenceGenerationArch()
        self.content_extractor = ContentExtractor(in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(ngf=ngf, groups=groups, depths=depths, num_heads=num_heads,
                                                             window_size=window_size, use_checkpoint=use_checkpoint)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, ref):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            ref
        """
        base = F.interpolate(x, None, 4, 'bilinear', False)
        features = self.net_extractor(base, ref)
        pre_offset_flow_sim, img_ref_feat = self.net_map(features, ref)

        # (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
        #                 and relu1_1. depths of the maps are 256, 128 and 64
        #                 respectively.
        upscale_restore = self.dyn_agg_restore(base, pre_offset_flow_sim, img_ref_feat)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self,
                 ngf=64,
                 groups=8,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 mlp_ratio=2.,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False):
        super(DynamicAggregationRestoration, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)
        self.embed_dim = ngf
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)

        # ---------------------- Down ----------------------

        # dynamic aggregation module for relu1_1 reference feature
        self.down_large_offset_conv1 = nn.Conv2d(ngf + 64 * 2, 64, 3, 1, 1, bias=True)
        self.down_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.down_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        # for large scale
        self.down_head_large = nn.Sequential(nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.down_body_large = SwinBlock(img_size=160, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)
        self.down_tail_large = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)

        # dynamic aggregation module for relu2_1 reference feature
        self.down_medium_offset_conv1 = nn.Conv2d(ngf + 128 * 2, 128, 3, 1, 1, bias=True)
        self.down_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.down_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.down_head_medium = nn.Sequential(nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.down_body_medium = SwinBlock(img_size=80, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size)
        self.down_tail_medium = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)

        # ---------------------- Up ----------------------
        # dynamic aggregation module for relu3_1 reference feature
        self.up_small_offset_conv1 = nn.Conv2d(ngf + 256 * 2, 256, 3, 1, 1, bias=True)  # concat for diff
        self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        # for small scale restoration
        self.up_head_small = nn.Sequential(nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.up_body_small = SwinBlock(img_size=40, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size)
        self.up_tail_small = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
                                           nn.PixelShuffle(2),
                                           nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.up_medium_offset_conv1 = nn.Conv2d(ngf + 128 * 2, 128, 3, 1, 1, bias=True)
        self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.up_head_medium = nn.Sequential(nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.up_body_medium = SwinBlock(img_size=80, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size)
        self.up_tail_medium = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
                                            nn.PixelShuffle(2),
                                            nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.up_large_offset_conv1 = nn.Conv2d(ngf + 64 * 2, 64, 3, 1, 1, bias=True)
        self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        # for large scale
        self.up_head_large = nn.Sequential(nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.up_body_large = SwinBlock(img_size=160, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)
        self.up_tail_large = nn.Sequential(nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
                                           nn.LeakyReLU(0.1, True),
                                           nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output

    def forward(self, base, pre_offset_flow_sim, img_ref_feat):
        pre_offset = pre_offset_flow_sim[0]
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]

        pre_relu1_swapped_feat = self.flow_warp(img_ref_feat['relu1_1'], pre_flow['relu1_1'])
        pre_relu2_swapped_feat = self.flow_warp(img_ref_feat['relu2_1'], pre_flow['relu2_1'])
        pre_relu3_swapped_feat = self.flow_warp(img_ref_feat['relu3_1'], pre_flow['relu3_1'])

        # Unet
        x0 = self.unet_head(base)  # [B, 64, 160, 160]

        # -------------- Down ------------------
        # large scale
        down_relu1_offset = torch.cat([x0, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        down_relu1_offset = self.lrelu(self.down_large_offset_conv1(down_relu1_offset))
        down_relu1_offset = self.lrelu(self.down_large_offset_conv2(down_relu1_offset))
        down_relu1_swapped_feat = self.lrelu(self.down_large_dyn_agg([img_ref_feat['relu1_1'], down_relu1_offset],
                                                                     pre_offset['relu1_1'],
                                                                     pre_similarity['relu1_1']))

        h = torch.cat([x0, down_relu1_swapped_feat], 1)
        h = self.down_head_large(h)
        h = self.down_body_large(h) + x0
        x1 = self.down_tail_large(h)  # [B, 64, 80, 80]

        # medium scale
        down_relu2_offset = torch.cat([x1, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv1(down_relu2_offset))
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv2(down_relu2_offset))
        down_relu2_swapped_feat = self.lrelu(self.down_medium_dyn_agg([img_ref_feat['relu2_1'], down_relu2_offset],
                                                                      pre_offset['relu2_1'],
                                                                      pre_similarity['relu2_1']))

        h = torch.cat([x1, down_relu2_swapped_feat], 1)
        h = self.down_head_medium(h)
        h = self.down_body_medium(h) + x1
        x2 = self.down_tail_medium(h)  # [9, 128, 40, 40]

        # -------------- Up ------------------

        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([x2, pre_relu3_swapped_feat, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(self.up_small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],
                                                              pre_offset['relu3_1'],
                                                              pre_similarity['relu3_1']))

        # small scale
        h = torch.cat([x2, relu3_swapped_feat], 1)
        h = self.up_head_small(h)
        h = self.up_body_small(h) + x2
        x = self.up_tail_small(h)  # [9, 64, 80, 80]

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(self.up_medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                                               pre_offset['relu2_1'],
                                                               pre_similarity['relu2_1']))
        # medium scale
        h = torch.cat([x + x1, relu2_swapped_feat], 1)
        h = self.up_head_medium(h)
        h = self.up_body_medium(h) + x
        x = self.up_tail_medium(h)  # [9, 64, 160, 160]

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(self.up_large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                                                              pre_offset['relu1_1'],
                                                              pre_similarity['relu1_1']))
        # large scale
        h = torch.cat([x + x0, relu1_swapped_feat], 1)
        h = self.up_head_large(h)
        h = self.up_body_large(h) + x
        x = self.up_tail_large(h)

        return x
