import torch

from ..modules.attention import *
from ..modules.diffusionmodules.util import (AlphaBlender, linear,
                                             timestep_embedding)
import logging
logger = logging.getLogger(__name__)

class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):
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
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x)) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x)) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight

# ... (其他导入和类) ...
try:
     # 导入修改后的 SpatialTransformer (假设都在 attention.py 或可访问路径)
     from .attention import SpatialTransformer, BasicTransformerBlockWithHifiVFSContext # 可能需要导入新的Block
     # ... 其他需要的 attention 模块 ...
except ImportError:
     try:
          from ..modules.attention import SpatialTransformer, BasicTransformerBlockWithHifiVFSContext
          # ... 其他需要的 attention 模块 ...
     except ImportError as e:
          print(f"导入修改后的 SpatialTransformer 失败: {e}")
          raise


class SpatialVideoTransformer(SpatialTransformer): # 继承修改后的 SpatialTransformer
    # 在 SpatialVideoTransformer 类定义内部
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1, # <<<--- 接收 depth
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        f_attr_context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        use_checkpoint=False, # <<<--- 注意这里和测试脚本统一为 checkpoint
        time_depth=1, # <<<--- 接收 time_depth (虽然下面没用它来循环)
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        sdp_backend=None,
    ):
        print("DEBUG: Entering MODIFIED SpatialVideoTransformer.__init__")
        # --- 调用修改后的父类 __init__ ---
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth, # <<<--- 传递 depth 给父类
            dropout=dropout,
            context_dim=context_dim,
            f_attr_context_dim=f_attr_context_dim,
            disable_self_attn=disable_self_attn,
            use_linear=use_linear,
            attn_type=attn_mode,
            use_checkpoint=use_checkpoint, # <<<--- 传递 use_checkpoint
            sdp_backend=sdp_backend,
        )
        # --- 父类调用结束 ---

        # --- 显式设置 self.depth ---
        self.depth = depth # <<<--- 取消注释或添加这行
        # --- 修改结束 ---

        self.time_depth = time_depth # (可选) 这个参数似乎没有在循环中使用，如果确实需要分开控制时间块深度，循环应改为 range(self.time_depth)
        self.max_time_embed_period = max_time_embed_period

        # --- 时间混合部分初始化 ... ---
        time_mix_d_head = d_head
        n_time_mix_heads = n_heads
        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)
        inner_dim = n_heads * d_head

        actual_time_context_dim = time_context_dim if not use_spatial_context else context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=actual_time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint, # <<<--- 确保这里传递的是 checkpoint
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn, # 可能不需要传递这个给时间块？
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                # 使用 self.depth 来保持时间块数量与空间块一致
                for _ in range(self.depth) # <<<--- 现在 self.depth 应该已定义
                # 如果需要独立的 time_depth, 使用 range(self.time_depth)
            ]
        )

        # assert len(self.time_stack) == len(self.transformer_blocks) # 这行断言现在应该可以通过

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )
        # --- 时间混合部分结束 ---

    # --- 修改 forward 方法以接收字典 context ---
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[Dict] = None, # <<<--- 确认接收字典
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- 在方法开始时打印接收到的 context 类型 ---
        logger.debug(f"[SpatialVideoTransformer.forward] Received context type: {type(context)}")
        if not isinstance(context, dict) and context is not None:
             logger.error(f"[SpatialVideoTransformer.forward] ERROR: Expected context to be a Dict, but got {type(context)}!")
        # ---

        b, c_in, h, w = x.shape
        x_in = x
        spatial_context_dict = context # 假设传入的是字典

        # --- 处理 time_context (逻辑不变，但可以加日志) ---
        actual_time_context = None
        if self.use_spatial_context:
             main_context_tensor = context.get('crossattn') if isinstance(context, dict) else None # 安全获取
             if main_context_tensor is not None and main_context_tensor.ndim == 3:
                 if timesteps is None or timesteps == 0: logger.warning(...)
                 else: # ... (repeat 逻辑) ...
                      time_context_first_timestep = main_context_tensor[::timesteps]
                      actual_time_context = repeat( time_context_first_timestep, "b seq d -> (b n) seq d", n=h * w )
             elif main_context_tensor is not None: logger.warning(...)
        elif time_context is not None: # ... (repeat 逻辑) ...
            actual_time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if actual_time_context.ndim == 2: actual_time_context = rearrange(actual_time_context, "b c -> b 1 c")
        logger.debug(f"[SpatialVideoTransformer.forward] Prepared actual_time_context type: {type(actual_time_context)}")
        # --- time_context 处理结束 ---


        x = self.norm(x)
        if not self.use_linear: x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear: x = self.proj_in(x)

        # --- 核心循环 ---
        for it_, (block, mix_block) in enumerate( zip(self.transformer_blocks, self.time_stack) ):
            # --- 在调用 block 前检查 spatial_context_dict 类型 ---
            logger.debug(f"[SpatialVideoTransformer.forward -> block {it_}] Passing context type: {type(spatial_context_dict)}")
            if not isinstance(spatial_context_dict, dict) and spatial_context_dict is not None:
                 logger.error(f"[SpatialVideoTransformer.forward -> block {it_}] ERROR: Context is not a Dict!")
            # ---
            # 1. 空间处理
            x = block(x, context=spatial_context_dict) # <<<--- 传递字典

            # 2. 时间混合
            x_mix = x
            # 调用时间混合块
            # --- 在调用 mix_block 前检查 actual_time_context 类型 ---
            logger.debug(f"[SpatialVideoTransformer.forward -> mix_block {it_}] Passing time_context type: {type(actual_time_context)}")
            # ---
            x_mix = mix_block(x_mix, context=actual_time_context, timesteps=timesteps) # <<<--- 传递 time_context

            # 应用 AlphaBlender
            x = self.time_mixer( x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator )
        # --- 循环结束 ---

        if self.use_linear: x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear: x = self.proj_out(x)
        out = x + x_in
        return out