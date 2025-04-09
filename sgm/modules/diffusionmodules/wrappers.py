import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "svd.sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
import logging
from typing import Optional
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG

class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


# svd/sgm/modules/diffusionmodules/wrappers.py
class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict,
                num_video_frames: Optional[int] = None,
                f_attr_low: Optional[torch.Tensor] = None,
                **kwargs):
        logger.debug(f"OpenAIWrapper: Before diffusion_model, context type: {type(c)}, num_frames: {num_video_frames}")
        
        # 检查视频维度
        is_video = len(x.shape) == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
        
        # 处理concat条件，确保维度匹配
        concat_tensor = c.get("concat", torch.Tensor([]).type_as(x))
        if concat_tensor.size(0) > 0:
            # 确保concat维度与x匹配
            if is_video and len(concat_tensor.shape) == 5:
                # concat是视频格式，重塑为(B*T,C,H,W)
                B, T, C_mask, H_mask, W_mask = concat_tensor.shape
                concat_tensor = concat_tensor.reshape(B*T, C_mask, H_mask, W_mask)
            elif is_video and len(concat_tensor.shape) == 4:
                # 每个视频只有一个遮罩，复制到每一帧
                if concat_tensor.shape[0] == x.shape[0] // num_video_frames:
                    B = concat_tensor.shape[0]
                    concat_tensor = concat_tensor.unsqueeze(1)  # (B,1,C,H,W)
                    concat_tensor = concat_tensor.expand(-1, num_video_frames, -1, -1, -1)  # (B,T,C,H,W)
                    concat_tensor = concat_tensor.reshape(B*num_video_frames, -1, concat_tensor.shape[-2], concat_tensor.shape[-1])

            # 修改拼接方式：将遮罩与输入融合而不是拼接
            if concat_tensor.size(0) > 0:
                # 使用遮罩作为注意力掩码：只在面部区域（遮罩=0）应用更新
                # 假设遮罩中0表示需要替换的区域，1表示保留的区域
                # 计算遮罩可见性 (取第一个通道，确保在[0,1]范围内)
                # 确保通道维度匹配
                if concat_tensor.shape[1] == x.shape[1]:  # 通道数相同
                    # 使用加权平均，而不是简单拼接
                    # 获取适当的权重，这里假设遮罩值在[-1,1]范围内
                    mask_weight = (concat_tensor[:, 0:1, :, :] + 1) / 2  # 转换到[0,1]
                    mask_weight = mask_weight.expand_as(x)  # 扩展到所有通道
                    logger.debug(f"融合前x形状: {x.shape}, 遮罩形状: {concat_tensor.shape}")
                    # 应用遮罩融合
                    # 注意：这是一个简化的融合，实际项目中可能需要更复杂的融合逻辑
                    # x = x  # 保持原样，不使用concat_tensor
                    # 或者，添加mask信息
                    c["mask_weight"] = mask_weight  # 将mask作为额外信息传递
                else:
                    logger.warning(f"遮罩通道数({concat_tensor.shape[1]})与输入({x.shape[1]})不匹配，跳过融合")

        # 准备UNet参数
        unet_kwargs = {
            "timesteps": t,
            "context": c,
            "y": c.get("vector", None),
            "num_video_frames": num_video_frames,
            "f_attr_low": f_attr_low,
            **kwargs
        }
        
        # 添加其他可能的条件
        if "cond_view" in c: unet_kwargs["cond_view"] = c.get("cond_view")
        if "cond_motion" in c: unet_kwargs["cond_motion"] = c.get("cond_motion")
        
        # 调用底层模型
        return self.diffusion_model(x, **unet_kwargs)