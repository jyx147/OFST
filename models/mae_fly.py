from functools import partial

import torch
import torch.nn as nn
import numpy as np

from models.vit import PatchEmbed, Block
from models.mg_memAE_ofd import Memory

from models.temporal import Temporal
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class PatchEmbed2(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channel=2, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # [14, 14]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # Batch_size, Channel, High, Weight = 8, 3, 224, 224
        assert H == self.img_size[0] and W == self.img_size[1]  # 确定H, W是224, 224
        # [B, C, H, W] flatten==> [B, C, H*W] transpose==> [B, H*W, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  # x通过一个卷积核为16*16的2d卷积将3通道的x转为768通道，并合并HW
        x = self.norm(x)
        return x

class Autoencodervit(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,num_slots=2000,shrink_thres=0.0005,
                 embed_dim=256, depth=4, num_heads=16,qk_scale=None,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        self.num_slots=num_slots
        self.shrink_thres=shrink_thres
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed2=PatchEmbed2(img_size,patch_size,2,embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # block_layer = []
        # for i in range(depth):
        #     block_layer += [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer), Temporal(embed_dim, 65)]
        # self.blocks = nn.Sequential(*block_layer)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.outliner = nn.Linear(embed_dim * 4, embed_dim, bias=True)
        # --------------------------------------------------------------------------
        #self.flowLinear=nn.Linear(260,4,bias=True)
        # --------------------------------------------------------------------------
        self.temporal = Temporal(embed_dim,65)
        self.temporaldecoder = Temporal(decoder_embed_dim,65)

        self.earlyfusion = nn.Conv1d(in_channels=130,out_channels=65,kernel_size=1)
        self.convfusion = nn.Conv1d(in_channels=520,out_channels=260,kernel_size=1)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.conv = nn.Conv2d(8,2,kernel_size=3,stride=1,padding=1)
        self.conv1=nn.Conv1d(8,2,kernel_size=2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed2=nn.Linear(2*decoder_embed_dim,decoder_embed_dim,bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.dilation = nn.Conv1d(decoder_embed_dim, decoder_embed_dim, kernel_size=4, padding=0, dilation=65)
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        #memory
        self.memvit=Memory(num_slots=self.num_slots, slot_dim=4*65*embed_dim,
                           shrink_thres=self.shrink_thres)
        # self.memvit = Memory(num_slots=self.num_slots, slot_dim=65 * embed_dim,
        #                      shrink_thres=self.shrink_thres)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        c=imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_embed(self,x,y):
        # embed patches
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:6, :, :]
        x3 = x[:, 6:9, :, :]
        x4 = x[:, 9:12, :, :]  # 128,3,32,32
        x1 = self.patch_embed(x1)

        x2 = self.patch_embed(x2)
        x3 = self.patch_embed(x3)
        x4 = self.patch_embed(x4)  # 128,64,768

        # add pos embed w/o cls token
        x1 = x1 + self.pos_embed[:, 1:, :]
        x2 = x2 + self.pos_embed[:, 1:, :]
        x3 = x3 + self.pos_embed[:, 1:, :]
        x4 = x4 + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 1,1,768
        cls_tokens1 = cls_token.expand(x1.shape[0], -1, -1)
        cls_tokens2 = cls_token.expand(x2.shape[0], -1, -1)
        cls_tokens3 = cls_token.expand(x3.shape[0], -1, -1)
        cls_tokens4 = cls_token.expand(x4.shape[0], -1, -1)  # 128,1,768
        x1 = torch.cat((cls_tokens1, x1), dim=1)
        x2 = torch.cat((cls_tokens2, x2), dim=1)
        x3 = torch.cat((cls_tokens3, x3), dim=1)
        x4 = torch.cat((cls_tokens4, x4), dim=1)  # 128,65,768
        #flow embed
        y1 = y[:, :2, :, :]
        y2 = y[:, 2:4, :, :]
        y3 = y[:, 4:6, :, :]
        y4 = y[:, 6:8, :, :]  # 128,2,32,32
        y1 = self.patch_embed2(y1)
        y2 = self.patch_embed2(y2)
        y3 = self.patch_embed2(y3)
        y4 = self.patch_embed2(y4)  # 128,64,768

        # add pos embed w/o cls token
        y1 = y1 + self.pos_embed[:, 1:, :]
        y2 = y2 + self.pos_embed[:, 1:, :]
        y3 = y3 + self.pos_embed[:, 1:, :]
        y4 = y4 + self.pos_embed[:, 1:, :]

        # append cls token
        cls_tokeny = self.cls_token + self.pos_embed[:, :1, :]  # 1,1,768
        cls_tokensy1 = cls_tokeny.expand(y1.shape[0], -1, -1)
        cls_tokensy2 = cls_tokeny.expand(y2.shape[0], -1, -1)
        cls_tokensy3 = cls_tokeny.expand(y3.shape[0], -1, -1)
        cls_tokensy4 = cls_tokeny.expand(y4.shape[0], -1, -1)  # 128,1,768
        y1 = torch.cat((cls_tokensy1, y1), dim=1)
        y2 = torch.cat((cls_tokensy2, y2), dim=1)
        y3 = torch.cat((cls_tokensy3, y3), dim=1)
        y4 = torch.cat((cls_tokensy4, y4), dim=1)  # 128,65,768
        return x1,x2,x3,x4,y1,y2,y3,y4

    def forward_encoder(self, x1,x2,x3,x4):

        # apply Transformer blocks
        for blk in self.blocks:#self.blocks
            x1 = blk(x1)#2，197，768
            x2 = blk(x2)
            x3 = blk(x3)
            x4 = blk(x4)
            x1, x2, x3, x4 = self.temporal(x1, x2, x3, x4)

        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x3 = self.norm(x3)
        x4 = self.norm(x4)

        return x1, x2, x3, x4


    def forward_flow(self,y1,y2,y3,y4):
        # apply Transformer blocks
        for blk in self.blocks:  # self.blocks
            y1 = blk(y1)  # 2，197，768
            y2 = blk(y2)
            y3 = blk(y3)
            y4 = blk(y4)
            y1, y2, y3, y4 = self.temporal(y1, y2, y3, y4)

        y1 = self.norm(y1)
        y2 = self.norm(y2)
        y3 = self.norm(y3)
        y4 = self.norm(y4)

        return y1, y2, y3, y4


    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)
        x1 = x[:, 0:65, :]
        x2 = x[:, 65:130, :]
        x3 = x[:, 130:195, :]
        x4 = x[:, 195:260, :]

        # add pos embed
        x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed
        x3 = x3 + self.decoder_pos_embed
        x4 = x4 + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x1 = blk(x1)
            x2 = blk(x2)
            x3 = blk(x3)
            x4 = blk(x4)
            x1, x2, x3, x4 = self.temporaldecoder(x1, x2, x3, x4)

        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.dilation(x.permute(0,2,1))
        x = self.decoder_norm(x.permute(0,2,1))

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x


    def forward(self, input_dict):
        flows = input_dict['motion']#reconstruct
        imgs = input_dict['appearance']#128,12,32,32
        _, _, fh, fw = imgs.shape
        imgs = torch.fft.rfft2(imgs, dim=(-2, -1))
        imgs[:, :, :, -(fw // 4):] = 0
        imgs = torch.fft.irfft2(imgs, dim=(-2, -1))

        x1, x2, x3, x4, y1, y2, y3, y4 = self.forward_embed(imgs,flows)
        x1 = self.earlyfusion(torch.cat((x1, y1), dim=1))
        x2 = self.earlyfusion(torch.cat((x2, y2), dim=1))
        x3 = self.earlyfusion(torch.cat((x3, y3), dim=1))
        x4 = self.earlyfusion(torch.cat((x4, y4), dim=1))
        y1, y2, y3, y4 = self.forward_flow(y1, y2, y3, y4)  # 128 260 768
        x1, x2, x3, x4 = self.forward_encoder(x1, x2, x3, x4)


        z1 = torch.cat((x1,y1),dim=1)
        z2 = torch.cat((x2,y2),dim=1)
        z3 = torch.cat((x3,y3),dim=1)
        z4 = torch.cat((x4,y4),dim=1)
        z = torch.cat((z1,z2,z3,z4),dim=1)

        z  = self.convfusion(z)
        # bs, C,dims = z.shape
        # x_img_1=z.view(bs, -1)
        # mem_out = self.memvit(x_img_1)
        # x_img_1 = mem_out["out"]
        # att_weight_vit = mem_out["att_weight"]

        # unflatten
        # x_img = x_img_1.view(bs, C, dims)
        pred = self.forward_decoder(z)# [N, L, p*p*3]
        predimg=self.unpatchify(pred)


        return  predimg


def mae_vitnew(**kwargs):
    model = Autoencodervit(
        patch_size=4, embed_dim=256, depth=4, num_heads=8,num_slots=2000,shrink_thres=0.005,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    x=torch.randn(128,12,32,32)
    y=torch.randn(128,8,32,32)
    mae=mae_vitnew()
    input_dict=dict(appearance=x,motion=y)
    frame_pred=mae(input_dict)
    print(frame_pred.shape)