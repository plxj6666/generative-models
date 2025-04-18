import logging
import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Dict
logpy = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warn("no module 'xformers'. Processing without...")

# from .diffusionmodules.util import mixed_checkpoint as checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "math")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_mode: str = "xformers",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        qkv = self.qkv(x)
        if self.attn_mode == "torch":
            qkv = rearrange(
                qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
            ).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "B H L D -> B L (H D)")
        elif self.attn_mode == "xformers":
            qkv = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = rearrange(x, "B L H D -> B L (H D)", H=self.num_heads)
        elif self.attn_mode == "math":
            qkv = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            logpy.warn(
                f"Attention mode '{attn_mode}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            logpy.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                logpy.info("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if self.checkpoint:
            # inputs = {"x": x, "context": context}
            return checkpoint(self._forward, x, context)
            # return checkpoint(self._forward, inputs, self.parameters(), self.checkpoint)
        else:
            return self._forward(**kwargs)

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # inputs = {"x": x, "context": context}
        # return checkpoint(self._forward, inputs, self.parameters(), self.checkpoint)
        return checkpoint(self._forward, x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


# --- 修改 SpatialTransformer 类 ---
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    Modfied to accept f_attr_context_dim and use BasicTransformerBlockWithHifiVFSContext.
    """
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None, # DIL context_dim
        f_attr_context_dim=None, # <<<--- 新增 FAL context_dim
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax", # 注意：这里是字符串，传递给 BasicBlock
        use_checkpoint=True,
        sdp_backend=None, # <<<--- 新增 sdp_backend
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )
        # --- 处理 context_dim 列表 (保持不变) ---
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth # 确保 context_dim 是列表
        elif context_dim is None:
            context_dim = [None] * depth
        elif len(context_dim) != depth:
             logpy.warning(f"{self.__class__.__name__}: context_dim list length ({len(context_dim)}) != depth ({depth}). Using first element.")
             context_dim = [context_dim[0]] * depth

        # --- 新增：处理 f_attr_context_dim 列表 ---
        if exists(f_attr_context_dim) and not isinstance(f_attr_context_dim, list):
            f_attr_context_dim = [f_attr_context_dim] * depth
        elif f_attr_context_dim is None:
            f_attr_context_dim = [None] * depth
        elif len(f_attr_context_dim) != depth:
             logpy.warning(f"{self.__class__.__name__}: f_attr_context_dim list length ({len(f_attr_context_dim)}) != depth ({depth}). Using first element.")
             f_attr_context_dim = [f_attr_context_dim[0]] * depth
        # --- 新增结束 ---

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # --- 实例化修改后的 Transformer Block ---
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockWithHifiVFSContext( # <<<--- 使用修改后的 Block
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d], # DIL context
                    f_attr_context_dim=f_attr_context_dim[d], # <<<--- 传入 FAL context
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type, # 传递字符串模式
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend, # 传递 backend
                )
                for d in range(depth)
            ]
        )
        # --- 修改结束 ---

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor: # <<<--- 修改 context 类型为 Dict
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        # --- 将完整的 context 字典传递给每个 block ---
        for i, block in enumerate(self.transformer_blocks):
            # 不需要再按索引取 context[i] 了，直接传递字典
            x = block(x, context=context) # <<<--- 传递字典
        # --- 修改结束 ---

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                BasicTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attn_mode="softmax-xformers",
                    checkpoint=checkpoint,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return x

# --- 1. 修改 BasicTransformerBlock ---
class BasicTransformerBlockWithHifiVFSContext(nn.Module):
    """
    修改后的 BasicTransformerBlock，用于处理包含 'crossattn' 和 'f_attr_tokens' 的 context 字典。
    """
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None, # DIL context dim (e.g., 768)
        f_attr_context_dim=None, # FAL context dim (e.g., 768 or 1280 if not projected)
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None, # 通常在 __init__ 中不直接使用，但在 forward 中可能影响 F.scaled_dot_product_attention
    ):
        super().__init__()
        # --- 选择 Attention 实现 ---
        attn_cls = self.ATTENTION_MODES.get(attn_mode)
        if attn_cls is None or (attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE):
            if attn_mode != "softmax": logpy.warn(f"Attention mode '{attn_mode}' not available, falling back to softmax.")
            attn_cls = CrossAttention # 使用内置的 scaled_dot_product
            attn_mode = "softmax" # 标记使用内置 SDP

        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls( # Self-Attention or Cross-Attention (if disable_self_attn)
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None, # 如果禁用自注意，则使用 DIL context
            backend=sdp_backend if attn_mode=="softmax" else None # 传递 backend 给 softmax 实现
        )
        self.norm1 = nn.LayerNorm(dim)

        # --- DIL Cross-Attention ---
        self.attn_dil = attn_cls(
            query_dim=dim,
            context_dim=context_dim, # 使用 DIL context dim
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend if attn_mode=="softmax" else None
        )
        self.norm_dil = nn.LayerNorm(dim)

        # --- FAL Attribute Cross-Attention ---
        # 如果提供了 f_attr context dim，则添加这一层
        self.attn_fal = None
        self.norm_fal = None
        self.f_attr_context_dim = f_attr_context_dim
        if self.f_attr_context_dim is not None:
            self.attn_fal = attn_cls(
                query_dim=dim,
                context_dim=self.f_attr_context_dim, # 使用 FAL context dim
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                backend=sdp_backend if attn_mode=="softmax" else None
            )
            self.norm_fal = nn.LayerNorm(dim)
            logpy.debug(f"BasicTransformerBlockWithHifiVFSContext: Added FAL Cross-Attention with context_dim={f_attr_context_dim}")


        # --- FeedForward ---
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm_ff = nn.LayerNorm(dim) # 通常在 FF 之前加 Norm

        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        if self.checkpoint and self.training: # 只在训练时启用 checkpoint
            # --- 从 context 提取 Tensor ---
            context = default(context, {})
            dil_context_tensor = context.get('crossattn')
            fal_context_tensor = context.get('f_attr_tokens')

            # --- 将 Tensor 作为单独参数传递给 checkpoint ---
            # 注意：如果某个 context 为 None，也需要传递 None
            return checkpoint(self._forward_for_checkpoint, x, dil_context_tensor, fal_context_tensor, use_reentrant=False)
        else:
            # 非 checkpoint 或 eval 模式下，直接调用 _forward
            return self._forward(x, context)

    # --- 新增一个用于 checkpoint 的内部 forward 方法 ---
    def _forward_for_checkpoint(self, x: torch.Tensor, dil_context_tensor: Optional[torch.Tensor], fal_context_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        # --- 在这里重新组合 context 字典 ---
        context_dict = {}
        if dil_context_tensor is not None:
            context_dict['crossattn'] = dil_context_tensor
        if fal_context_tensor is not None:
            context_dict['f_attr_tokens'] = fal_context_tensor

        # --- 调用原始的 _forward 逻辑 ---
        return self._forward(x, context_dict)
    # --- 新增结束 ---

    def _forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        # 处理输入 context
        if context is None:
            context = {}
        elif isinstance(context, torch.Tensor):
            # 兼容处理：如果传入 tensor，当作 dil_context
            context = {'crossattn': context}
        
        # 获取不同类型的 context
        dil_context = context.get('crossattn')
        fal_context = context.get('f_attr_tokens')
        
        # 1. 自注意力
        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=dil_context) + x
        else:
            x = self.attn1(self.norm1(x)) + x
        
        # 2. DIL 交叉注意力
        if dil_context is not None:
            x = self.attn_dil(self.norm_dil(x), context=dil_context) + x
        
        # 3. FAL 交叉注意力 (如果有)
        if hasattr(self, 'attn_fal') and self.attn_fal is not None and fal_context is not None:
            x = self.attn_fal(self.norm_fal(x), context=fal_context) + x
        
        # 4. FeedForward - 修复：使用 norm_ff 而不是不存在的 norm3
        x = self.ff(self.norm_ff(x)) + x
        
        return x