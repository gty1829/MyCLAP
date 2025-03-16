import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .get_audio_features import get_audio_features

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

@dataclass
class ModelDimensions:
    # Whisper
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    # MoE
    n_shared_experts: int = 1   # 共享专家数量
    n_routed_experts: int = 3   # 路由专家数量(不含共享专家数量)
    topk: int = 1 # 激活专家数量
    score_func: Literal["softmax", "sigmoid"] = "softmax" # 路由函数
    route_scale: float = 1.0    # 路由函数缩放因子
    #n_inter_state: int =   # 专家网络中间层维度
    

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.n_layer = n_layer
        self.x_stacked = []
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.linear = Linear(n_layer, 1) # 对所有层的输出进行加权平均(线性变换)
        self.ln_post = LayerNorm(n_state)
        self.ln_post2 = LayerNorm(n_state)
        

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        Return:
            x: torch.Tensor, shape = (batch_size, n_frames, n_state * 2) 
                                   = (batch_size, n_frames, n_state_concat)
        """
        #print('\n----------------------Audio Encoder-----------------------')
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        self.x_stacked.append(x)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:     # 获得从卷积层输出到倒数第二个层的输出向量的拼接
                self.x_stacked.append(x)     

        #print("len(self.x_stacked) after loop: \n", len(self.x_stacked)) # (n_layers, batch_size, n_frames, n_state)

        self.x_stacked = torch.stack(self.x_stacked, dim=0) 
        #print("x_stacked.shape after list -> tensor(n_layers, batch_size, n_frames, n_state): \n", self.x_stacked.shape)

        n_layer, batch_size, n_frames, n_state = self.x_stacked.shape
        assert (n_layer == self.n_layer)

        # (batch_size, n_frames, n_state, n_layers)
        self.x_stacked = self.x_stacked.permute(1, 2, 3, 0).view(-1, n_state, n_layer)
        #print("self.x_stacked.shape after permute(1, 2, 3, 0).reshape(-1, n_state, n_layer)(batch_size * n_frames, n_state, n_layer):\n", self.x_stacked.shape)
        
        self.x_stacked = self.linear(self.x_stacked).squeeze(-1)    # 对所有层的输出进行加权平均, 去掉最后一维
        #print("self.x_stacked.shape after squeeze: \n", self.x_stacked.shape)
        self.x_stacked = self.x_stacked.view(batch_size, n_frames, n_state)  
        # 目前不确定是否需要对x_stacked加入layer norm
        #print("x_stacked.shape after linear(batch_size, n_frames, n_state): \n", self.x_stacked.shape)

        x = self.ln_post(x)
        self.x_stacked = self.ln_post2(self.x_stacked)

        #print("x.shape after ln_post(batch_size, n_frames, n_state): \n", x.shape)
        #print("x_stacked.shape after ln_post2(batch_size, n_frames, n_state): \n", self.x_stacked.shape)

        x = torch.concat([x, self.x_stacked], dim=-1)   # 在特征维度拼接
        #print("x.shape after concat(batch_size, n_frames, 2 * n_state), \n", x.shape)
        return x    

class Gate(nn.Module):
    """
    Router, 用于路由输入到不同的专家
    """
    def __init__(
            self,
            n_state_concat: int,
            topk: int,
            score_func: Literal["softmax", "sigmoid"],
            route_scale: float,
            n_routed_experts: int
    ):
        super().__init__()
        self.n_state_concat = n_state_concat # encoder是拼接的输出，维度为2 * n_state
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale
        self.n_routed_experts = n_routed_experts
        # Linear默认有bias
        self.linear = Linear(self.n_state_concat, n_routed_experts, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向过程
        Args:
            x: torch.Tensor, shape = (batch_size * n_frames, 2*n_state)
                输入张量
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                - weights: 选中的专家对应的权重 (batch_size * n_frames, topk)。
                - indices: 选中的专家索引 (batch_size * n_frames, topk)。
        """
        #print("\n-----------------Gate------------------")
        #print("x.shape before get Gate scores(batch_size * n_frames, 2*n_state): \n", x.shape)
        # 对每个batch的每一个音频的每一帧根据特征维度打分
        scores = self.linear(x) # scores.shape = (batch_size * n_frames, n_routed_experts)
        #print("scores.shape(batch_size * n_frames, n_routed_experts): \n", scores.shape)
        # 根据分数函数归一化
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        else:
            scores = torch.sigmoid()

        # 取前topk个专家的权重
        weights, indices = torch.topk(scores, self.topk, dim=-1)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)  # 确保所有 `weights` 之和为 1

        # 乘缩放因子
        weights *= self.route_scale
        #print("weights.shape(batch_size * n_frames, topk==1): \n", weights.shape)
        #print("indices.shape(batch_size * n_frames, topk==1): \n", indices.shape)
        return weights, indices

class Expert(nn.Module):
    def __init__(
            self, 
            n_state_concat: int, 
            n_inter_state: int
        ):
        super().__init__()
        self.n_inter_state = n_inter_state
        self.n_state_concat = n_state_concat
        self.n_state = n_state_concat // 2
        self.linear1 = Linear(self.n_state_concat, self.n_inter_state)
        #self.linear2 = Linear(self.n_inter_state, self.n_inter_state)
        self.linear2 = Linear(self.n_inter_state, self.n_state)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向过程
        Args:
            x: torch.Tensor, shape = (batch_size * n_frames, n_state_concat)
                输入张量
        Returns:
            torch.Tensor, shape = (batch_size * n_frames, n_state)
                输出张量
        """
        #print('-----------------Expert-----------------')
        #print("x.shape before expert processed(batch_size * n_frames, n_state_concat): \n", x.shape)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        #print("x.shape after expert processed(batch_size * n_frames, n_state): \n", x.shape)
        return x


class MoE(nn.Module):
    def __init__(
            self,
            n_state: int,
            #n_inter_state: int,
            n_shared_experts: int,
            n_routed_experts: int,
            topk: int,
            score_func: Literal["softmax", "sigmoid"],
            route_scale: float,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_state_concat = n_state * 2
        self.n_inter_state = 4 * n_state
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale
        self.gate = Gate(n_state_concat=self.n_state_concat, topk=topk, score_func=score_func, route_scale=route_scale, n_routed_experts=n_routed_experts)
        self.routed_experts = nn.ModuleList(
            [Expert(n_state_concat=self.n_state_concat, n_inter_state=self.n_inter_state) for _ in range(n_routed_experts)]
        )

        self.shared_experts_linear1 = Linear(self.n_state_concat, self.n_inter_state)
        self.shared_experts_linear2 = Linear(self.n_inter_state, self.n_shared_experts * self.n_state)
        self.layer_norm = LayerNorm(self.n_state)

    def forward(self, x: Tensor) -> Tensor:
        """
        MoE模块前向过程。
        Args:
            x: torch.Tensor, shape = (batch_size, n_frames, n_state_concat)
                输入张量
        Returns:
            torch.Tensor, shape = (batch_size, n_frames, n_state)
                输出张量
        """
        #print('\n-----------------MoE-----------------')
        #print("x.shape before MoE processed(batch_size, n_frames, n_state_concat): \n", x.shape)
        batch_size, n_frames, n_state_concat = x.shape
        assert n_state_concat == self.n_state_concat, "输入张量的维度不正确"

        x = x.view(-1, n_state_concat)
        #print("x.shape after view(batch_size * n_frames, n_state_concat): \n", x.shape)

        weights, indices = self.gate(x) # x.shape: (batch_size * n_frames, topk)
                                        # gate不改变x
        #print("weights.shape after gate(batch_size * n_frames, topk==1): \n", weights.shape)
        # 存储计算结果 y.shape(batch_size * n_frames, n_state)
        y = torch.zeros(batch_size * n_frames, self.n_state, device=x.device, dtype=x.dtype)
        #print("y.shape(batch_size * n_frames, n_state): \n", y.shape)
        # torch.bincount计算一维整数张量中每个值(0~max(input))出现的次数
        # 计算每个专家被分配的tokens数
        counts = torch.bincount(input=indices.flatten(), minlength=self.n_routed_experts).tolist()
        #print("len(counts): ", len(counts))
        #print("len(self.experts): ", len(self.routed_experts))
        # 这里有个bug，下面遍历i会溢出，再看一遍
        # 遍历每个专家
        for i, expert in enumerate(self.routed_experts):
            if counts[i] == 0: # 当前专家没有被分配tokens
                continue
            # idx是torch.tensor, 元素含义是(batch_size * n_frames, 1)
            # 表明是当前批次中的当前帧中选择了专家i(无论专家i是在topk的第几个)
            # top输出是二维向量, 每个一维向量表示每一帧i专家在topk个专家中的索引
            # 如果indices是按照topk排好序的, 则top选择的就是每一帧选择的topk个专家中第i个专家的排名
            # 如果indices不是按照topk排好序的, 则top仅仅表示indices==i的索引(top本身不具有排序功能)
            idx, top = torch.where(indices == i)
            #print(f"\nExpert_{i}: ")
            #print("idx.shape: ", idx.shape)
            #print("top.shape: ", top.shape)
            #weights_sum = weights.sum()
            #print("weights_sum: ", weights_sum) tensor(2672., device='cuda:0', dtype=torch.float16, grad_fn=<SumBackward0>)
            # 对当前expert输入进行加权平均(加None主要是为了正确广播并且逐元素相乘)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
            #print(f"y[idx].shape: ", y[idx].shape)

        z = self.shared_experts_linear1(x)
        z = self.shared_experts_linear2(z)
        #print("z.shape(batch_size * n_frames, n_shared_experts * n_state), n_shared_experts==1: \n", z.shape)
        ret = ((y + z) / 2.0).view(batch_size, n_frames, self.n_state)
        ret = self.layer_norm(ret)
        #print("MoE ret.shape(batch_size, n_frames, n_state): \n", ret.shape)
        return ret

class Whisper_MoE(nn.Module):
    embed_audio = get_audio_features

    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,            
        )
        self.MoE = MoE(
            n_state = self.dims.n_audio_state,
            n_shared_experts = self.dims.n_shared_experts,
            n_routed_experts = self.dims.n_routed_experts,
            topk = self.dims.topk,
            score_func = self.dims.score_func,
            route_scale = self.dims.route_scale
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
