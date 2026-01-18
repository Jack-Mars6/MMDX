import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()  #(B, C, N)->(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  #(B, H * W, C)
        return x

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, activation='relu'):
        super(MSDC, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.use_skip_connection = True

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.pconv2 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        # Apply the convolution layers in a loop
        output = 0
        for dwconv in self.dwconvs:
            dout = dwconv(x)
            output += dout
        output = self.pconv1(channel_shuffle(output, gcd(self.in_channels, self.out_channels)))
        if self.in_channels != self.out_channels:
            x = self.pconv2(x)

        return output + x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        # Global Average Pooling and Max Pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # MLP
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        # Channel attention
        out = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Average and Max pooling across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along the channel dimension
        cat = torch.cat([avg_pool, max_pool], dim=1)

        # Spatial attention
        out = torch.sigmoid(self.conv1(cat))
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=12, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # print(x.shape)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LBM(nn.Module):
    def __init__(self, in_channels, in_resolution):
        super(LBM, self).__init__()
        self.in_channels = in_channels
        self.height = in_resolution
        self.width = in_resolution
        # self.alpha = nn.Parameter(torch.full((self.in_channels, self.height, self.width), 1.4), requires_grad=True)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(2*self.in_channels, self.in_channels, 1, 1, 0, bias=False),
            nn.LayerNorm([self.in_channels, self.height, self.width]),
            # nn.ReLU(inplace=False)
            nn.Sigmoid()
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0, bias=False),
            nn.LayerNorm([self.in_channels, self.height, self.width]),
            # nn.ReLU(inplace=False)
            nn.Sigmoid()
        )
        self.lp = nn.Parameter(torch.zeros(self.in_channels, self.height, self.width), requires_grad=True)
        self.lc = nn.Parameter(torch.zeros(self.in_channels, self.height, self.width), requires_grad=True)
        self.hp = nn.Parameter(torch.zeros(self.in_channels, self.height, self.width), requires_grad=True)
        self.hc = nn.Parameter(torch.zeros(self.in_channels, self.height, self.width), requires_grad=True)
        self.dist = nn.Parameter(torch.ones(self.in_channels, self.height, self.width), requires_grad=True)

    def forward(self, xl, xh):
        alpha = self.pconv1(torch.cat((xl, xh), dim=1))
        alpha = self.pconv2(self.pconv2(alpha))
        alpha = 1.4 * alpha
        lb_para = alpha * torch.exp((self.lp + self.lc - self.hp - self.hc) * self.dist)
        Xh = xl * lb_para
        Xl = xh * torch.reciprocal(lb_para)
        return Xl, Xh


class Aggregation(nn.Module):
    def __init__(self, dims, resolution, Ablation_mix=False):
        super(Aggregation, self).__init__()
        self.flag = Ablation_mix
        self.cbam = CBAM(dims)
        self.linear = nn.Linear(2 * dims, dims)
        self.mscb = MSDC(2 * dims, dims)
        self.lbm = LBM(in_channels=dims, in_resolution=resolution)

    def forward(self, x_100, x_140):
        if not self.flag:
            # fusion
            xl, xh = self.lbm(x_100, x_140)
            x_100 = torch.add(xl, x_100)
            x_140 = torch.add(xh, x_140)
            x = torch.cat((x_100, x_140), dim=1)
            x = self.mscb(x)
        else:
            x = torch.cat((x_100, x_140), dim=1)
            x = self.linear(x.permute(0, 2, 3, 1).contiguous())
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)  #将隐藏特征的数量调整为输入特征的三分之二
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)  #将输出沿最后一个维度分成两个部分
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw, indexing='ij')
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw, indexing='ij')

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    # (128, 128) (16, 16) torch.Size([61504, 2]) torch.Size([4194304])
    # print(query_size, key_size, relative_coords_table.shape, idx_map.shape)

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2),
                        stride=1)  #生成一个张量 attn_map，其形状为 [1, 1, num_windows, window_size * window_size]
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)  #形状为 [num_windows, 1] 的张量
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0  #生成一个布尔型掩码[num_windows, 1, window_size * window_size]
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_classes=4, is_up=False, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False, Ablation_att=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.is_extrapolation = is_extrapolation
        self.num_classes = num_classes
        self.is_up = is_up
        self.Ablation_att = Ablation_att

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        if self.is_up & (not self.Ablation_att):
            self.query_embedding = nn.Parameter(
                nn.init.trunc_normal_(torch.empty(self.num_heads, self.num_classes, self.head_dim), mean=0, std=0.02))
        else:
            self.query_embedding = nn.Parameter(
                nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.qe_fc = nn.Linear(num_classes, 1)
        self.ql_fc = nn.Linear(num_classes, 1)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # dynamic_local_bias:
        if self.is_up & (not self.Ablation_att):
            self.learnable_tokens = nn.Parameter(
                nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len, self.num_classes), mean=0, std=0.02))
        else:
            self.learnable_tokens = nn.Parameter(
                nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        if self.is_up & (not self.Ablation_att):
            qe_list = []
            for i in range(self.num_classes):
                qe_list.append(q_norm + self.query_embedding[:, i, :].unsqueeze(1))
            qe = self.qe_fc(torch.stack(qe_list, dim=-1).softmax(dim=-1))
            qe = qe.squeeze(-1)
        else:
            qe = q_norm + self.query_embedding
        q_norm_scaled = qe * F.softplus(self.temperature) * seq_length_scale

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)

        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2)
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                    self.trained_H,
                                                                                                    self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)

        # class-wise learnable_tokens
        if self.is_up & (not self.Ablation_att):
            ql_list = []
            for i in range(self.num_classes):
                ql_list.append(q_norm @ self.learnable_tokens[:, :, :, i])
            ql = self.ql_fc(torch.stack(ql_list, dim=-1).softmax(dim=-1))
            ql = ql.squeeze(-1)
        else:
            ql = q_norm @ self.learnable_tokens

        x_local = ((ql + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)

        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   #定义一个线性变换层，用来将输入张量 x 映射到查询（Query）、键（Key）和值（Value）三个部分
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02)) #每个头的查询嵌入（query_embedding）是一个训练参数，经过截断正态初始化，用于添加到查询向量 q 中
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        if self.is_extrapolation:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, N, N)
        else:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_len)
            # bilinear interpolation:
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear')
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(-1, N, self.trained_H,
                                                                                           self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear').reshape(-1, N, N).transpose(-1, -2)

        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, is_extrapolation=False, num_classes=4, is_up=False, Ablation_att=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                is_extrapolation=is_extrapolation)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                num_classes,
                is_up,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation,
                Ablation_att=Ablation_att)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))  # (H+2*3-7)/4 + 1
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class OverlapPatchExpand(nn.Module):
    """ Image to Patch Expanding"""
    def __init__(self, dim=576, scale=2):
        super().__init__()
        self.expand = nn.Linear(dim, dim * scale)
        self.norm = nn.LayerNorm(dim // scale)
        self.scale = scale

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.expand(x)  #(B, H, W, C)->(B, H, W, 2*C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.scale, p2=self.scale, c=C // self.scale)  #(B, H, W, 2*C)->(B, 2H, 2W, C/2)
        x = x.permute(0, 3, 1, 2)
        # _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim//scale)
        # x = self.norm(x)
        return x


class FinalPatchExpand(nn.Module):
    """ Image to Patch Expanding"""
    def __init__(self, dim=72, scale=4):
        super().__init__()
        self.expand = nn.Linear(dim, dim * scale)
        self.norm = nn.LayerNorm(dim // scale)
        self.scale = scale

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.expand(x)  #(B, H, W, C)->(B, H, W, 4*C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.scale, p2=self.scale, c=C // self.scale)  #(B, H, W, 4*C)->(B, 4H, 4W, C/4)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class TransNeXt(nn.Module):
    """
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation can accept multi-scale inputs,
    it is recommended to set the "img size" parameter to a value close to the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    """

    def __init__(self, img_size=512, pretrain_size=None, window_size=[3, 3, 3, None],
                 patch_size=4, in_chans=1, num_classes=4, embed_dims=[72, 144, 288, 576],
                 num_heads=[3, 6, 12, 24], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.3, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=False, is_extrapolation=False,
                 Ablation_mix=False, Ablation_att=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size
        self.Ablation_mix = Ablation_mix
        self.Ablation_att = Ablation_att

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # encoder
        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation, num_classes=num_classes,
                is_up=False, Ablation_att=self.Ablation_att)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            aggregation = Aggregation(embed_dims[i], img_size // (2 ** (i + 2)), Ablation_mix=self.Ablation_mix)
            setattr(self, f"aggregation{i + 1}", aggregation)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # decoder
        for k in range(num_stages):
            i = 3 - k
            if not self.is_extrapolation:
                relative_pos_index_up, relative_coords_table_up = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index_up{k + 1}", relative_pos_index_up, persistent=False)
                self.register_buffer(f"relative_coords_table_up{k + 1}", relative_coords_table_up, persistent=False)

            patch_expand = OverlapPatchExpand(dim=embed_dims[i], scale=4 if i == 0 else 2)
            concat_back_dim = nn.Linear(2 * embed_dims[i], embed_dims[i])

            block_up = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))),
                window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation, num_classes=num_classes,
                is_up=True, Ablation_att=self.Ablation_att)
                for j in range(depths[i])])
            norm_up = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_expand{k + 1}", patch_expand)
            setattr(self, f"block_up{k + 1}", block_up)
            setattr(self, f"norm_up{k + 1}", norm_up)
            setattr(self, f"concat_back_dim{k + 1}", concat_back_dim)

        self.final_up = FinalPatchExpand(embed_dims[0], 4)
        self.final_conv = nn.Conv2d(embed_dims[0] // 4, num_classes, 1)

        for n, m in self.named_modules():
            self._init_weights(m, n)
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x_100 = x[:, 0, :, :].unsqueeze(dim=1)
        x_140 = x[:, 1, :, :].unsqueeze(dim=1)
        skip_list = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            aggregation = getattr(self, f"aggregation{i + 1}")

            x_100, H, W = patch_embed(x_100)
            x_140, H, W = patch_embed(x_140)
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table \
                    = get_relative_position_cpb(query_size=(H, W),
                                                key_size=(H // sr_ratio, W // sr_ratio),
                                                pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))),
                                                device=x.device)
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))
                    padding_mask = None
            for blk in block:
                x_100 = blk(x_100, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)
                x_140 = blk(x_140, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)
            x_100 = norm(x_100)
            x_140 = norm(x_140)
            x_100 = x_100.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
            x_140 = x_140.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # print(x.shape)
            x = aggregation(x_100, x_140)
            skip_list.append(x)

        return x, skip_list


    def forward_features_up(self, x, skip_list):
        B = x.shape[0]  #(B, C, H, W)

        for k in range(self.num_stages):
            i = 3 - k
            block_up = getattr(self, f"block_up{k + 1}")
            norm_up = getattr(self, f"norm_up{k + 1}")
            concat_back_dim = getattr(self, f"concat_back_dim{k + 1}")

            #跳跃连接
            if k == 0:
                x = skip_list[3 - k]

            else:
                patch_expand = getattr(self, f"patch_expand{k}")
                x = patch_expand(x)
                x = torch.cat([x, skip_list[3 - k]], 1)
                x = x.permute(0, 2, 3, 1).contiguous()
                x = concat_back_dim(x)
                x = x.permute(0, 3, 1, 2).contiguous()

            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
            x = norm_up(x)

            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index_up, relative_coords_table_up \
                    = get_relative_position_cpb(query_size=(H, W),
                                                key_size=(H // sr_ratio, W // sr_ratio),
                                                pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))),
                                                device=x.device)
            else:
                relative_pos_index_up = getattr(self, f"relative_pos_index_up{k + 1}")
                relative_coords_table_up = getattr(self, f"relative_coords_table_up{k + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))
                    padding_mask = None
            for blk in block_up:
                x = blk(x, H, W, relative_pos_index_up, relative_coords_table_up, seq_length_scale, padding_mask)
            x = norm_up(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        # print(x.shape)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x


def transnext_tiny(in_chans: int, num_classes: int, Ablation_mix=False, Ablation_att=False):
    model = TransNeXt(img_size=512, window_size=[3, 3, 3, None],
                      patch_size=4, in_chans=in_chans, num_classes=num_classes, embed_dims=[72, 144, 288, 576],
                      num_heads=[3, 6, 12, 24],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1],
                      drop_rate=0.0, drop_path_rate=0.3, Ablation_mix=Ablation_mix, Ablation_att=Ablation_att)
    return model



if __name__ == "__main__":
    x = torch.randn(1, 2, 512, 512).cuda()
    print('x', x.shape)
    net = transnext_tiny(in_chans=1, num_classes=4, Ablation_mix=False, Ablation_att=False).cuda()
    y = net(x)
    print('y', y.shape)

    # param count
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")


