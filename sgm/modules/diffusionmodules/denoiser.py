from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization
from typing import Optional
import logging
logger = logging.getLogger(__name__)

class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        num_video_frames: Optional[int] = None,
        f_attr_low: Optional[torch.Tensor] = None,
        **additional_model_inputs,
    ) -> torch.Tensor:
        """
        修改后的 forward 方法，改进了参数处理和错误检测
        
        Args:
            network: 要调用的 UNet 模型
            input: 输入张量 (噪声 + VAE潜变量)
            sigma: 噪声水平
            cond: 条件字典
            num_video_frames: 视频帧数
            f_attr_low: 低层次属性特征，用于在 UNet 内部注入
        """
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        
        # 自动检测和处理视频帧数
        final_num_frames = num_video_frames
        
        # 尝试多种方式确定视频帧数
        if final_num_frames is None:
            # 1. 检查 additional_model_inputs
            final_num_frames = additional_model_inputs.get('num_video_frames')
            
            # 2. 检查 cond 字典
            if final_num_frames is None and isinstance(cond, dict) and 'num_frames' in cond:
                final_num_frames = cond['num_frames']
                logger.info(f"从cond字典获取num_video_frames={final_num_frames}")
            
            # 3. 从输入形状推断
            if final_num_frames is None and len(input.shape) == 5:  # (B,T,C,H,W)
                final_num_frames = input.shape[1]
                logger.info(f"从input形状推断num_video_frames={final_num_frames}")
            
            # 4. 从 f_attr_low 推断（如果可能）
            if final_num_frames is None and f_attr_low is not None:
                if len(f_attr_low.shape) == 5:  # (B,T,C,H,W)
                    final_num_frames = f_attr_low.shape[1]
                    logger.info(f"从f_attr_low形状推断num_video_frames={final_num_frames}")
        
        # 如果仍然无法确定，使用默认值并发出警告
        if final_num_frames is None:
            logger.warning("无法确定视频帧数，使用默认值2")
            final_num_frames = 2
        
        # 检查 f_attr_low 批次大小与输入的匹配性
        if f_attr_low is not None:
            input_batch_size = input.shape[0]
            if len(input.shape) == 5:
                input_batch_size = input.shape[0] * input.shape[1]
                
            f_attr_batch_size = f_attr_low.shape[0]
            if len(f_attr_low.shape) == 5:
                f_attr_batch_size = f_attr_low.shape[0] * f_attr_low.shape[1]
                
            if f_attr_batch_size != input_batch_size:
                logger.warning(f"f_attr_low批次大小({f_attr_batch_size})与输入({input_batch_size})不匹配")
                if f_attr_batch_size == 1:
                    # 单个样本，扩展到所有批次
                    if len(f_attr_low.shape) == 4:  # (1,C,H,W)
                        f_attr_low = f_attr_low.repeat(input_batch_size, 1, 1, 1)
                        logger.info(f"f_attr_low已扩展至匹配输入批次: {f_attr_low.shape}")
        
        # 确保 cond 是一个字典
        if cond is None:
            cond = {}
        elif not isinstance(cond, dict):
            logger.error(f"cond不是字典类型: {type(cond)}，尝试转换")
            try:
                if hasattr(cond, 'keys'):
                    # 尝试转换为字典
                    cond = {k: cond[k] for k in cond.keys()}
                else:
                    # 无法转换，创建新字典
                    cond = {"context": cond}
            except Exception as e:
                logger.error(f"转换cond为字典失败: {e}")
                cond = {}
        
        # 记录 f_attr_low 信息
        if f_attr_low is not None:
            logger.debug(f"[Denoiser.forward] 传递f_attr_low，形状: {f_attr_low.shape}")
        
        # 调用网络
        return (
            network(
                input * c_in, 
                c_noise, 
                cond, 
                num_video_frames=final_num_frames, 
                f_attr_low=f_attr_low,
                **additional_model_inputs
            ) * c_out
            + input * c_skip
        )


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
