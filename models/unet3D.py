import torch
from einops import einsum
import torch.nn as nn
from models.blocks import get_time_embedding
from models.blocks import DownBlock3d, MidBlock3d, UpBlockUnet3d
from utils.config_utils import *


class Unet3D(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.cross_attns = model_config["cross_attn"]
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ######## Class, Mask and Text Conditioning Config #####
        self.cond_dim = None
        self.class_cond = False
        self.image_cond = False
        self.value_cond = False

        self.condition_config = get_config_value(model_config, 'condition_config', None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'value' in condition_types:
                self.value_cond = True
                self.value_in_dim = self.condition_config['value_condition_config']['len_vector']
            if 'image' in condition_types:
                self.image_cond = True
                self.cond_dim = self.condition_config['image_condition_config']['context_dim']
                # self.im_cond_input_ch = self.condition_config['image_condition_config'][
                #     'image_condition_input_channels']
                # self.im_cond_output_ch = self.condition_config['image_condition_config'][
                #     'image_condition_output_channels']

        if self.class_cond:
            # Rather than using a special null class we dont add the
            # class embedding information for unconditional generation
            self.class_emb = nn.Embedding(self.num_classes,
                                          self.t_emb_dim)

        if self.image_cond:
            self.im_encoder = nn.Sequential(
                nn.Conv2d(3, int(self.im_context_dim//4), 3, stride=2, padding=1),  # (b, 3, h, w) → (b, c/2, h/2, w/2)
                nn.ReLU(),
                nn.Conv2d(self.im_context_dim//4, self.im_context_dim//2, 3, stride=2, padding=1), # → (b, c, h/4, w/4)
                nn.ReLU(),
                nn.Conv2d(self.im_context_dim // 2, self.im_context_dim, 3, stride=2, padding=1),  # → (b, c, h/4, w/4)
                nn.ReLU()
                )
            self.im_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj_im2time = nn.Linear(self.im_context_dim, self.t_emb_dim)

        if self.value_cond:
            # from (b, n, 1) to (b, n, t_emb_dim)
            self.value_encoder = nn.Sequential(
                        nn.Linear(self.value_in_dim, 128),
                        nn.SiLU(),
                        nn.Linear(128, self.t_emb_dim)
                        )
        self.cond = self.image_cond or self.class_cond or self.value_cond
        ###################################

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.conv_in = nn.Conv3d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock3d(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        cross_attn=self.cross_attns[i],
                                        context_dim=self.cond_dim,
                                        attn=self.attns[i], norm_channels=self.norm_channels))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock3d(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      cross_attn=self.cross_attns[i],
                                      context_dim=self.cond_dim,
                                      norm_channels=self.norm_channels))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet3d(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            cross_attn=self.cross_attns[i],
                            context_dim=self.cond_dim,
                            norm_channels=self.norm_channels))

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv3d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_input=None):

        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"

        cond_tokens = None
        if self.image_cond:
            # cond
            image_latent = self.im_encoder(cond_input['image']) # (b, c, h, w)
            b, c, h, w = image_latent.shape
            cond_tokens = image_latent.reshape(b, c, h*w)
            cond_tokens = cond_tokens.transpose(1, 2)

            # add the image latent to time emb vector

            image_embd = self.im_avgpool(image_latent).view(image_latent.size(0), -1)
            image_embd = self.proj_im2time(image_embd)
            # print("image condition added to time vector. ")
            t_emb += image_embd

        ######## Class Conditioning ########
        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)
            class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
            t_emb += class_embed

        if self.value_cond:
            value_embd = self.value_encoder(cond_input['value']) # (b, n, 1) to (b, n, t_emd_dim)
            t_emb += value_embd

        out = self.conv_in(x)
        # B x C1 x H x W

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, cond_tokens)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb, cond_tokens)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, cond_tokens)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
