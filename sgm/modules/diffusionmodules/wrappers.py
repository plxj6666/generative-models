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
    def forward( self, x: torch.Tensor, t: torch.Tensor, c: dict,
                 num_video_frames: Optional[int] = None, # <<<--- 添加显式参数
                 **kwargs ):
        logger.debug(f"OpenAIWrapper: Before diffusion_model, context type: {type(c)}, num_frames: {num_video_frames}")
        # --- 移除从 kwargs 中提取 num_video_frames 的逻辑 (如果之前有的话) ---
        # --- 直接将接收到的 num_video_frames 传递给 diffusion_model ---
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # --- 准备传递给 VideoUNet 的参数 ---
        unet_kwargs = {
            "timesteps": t,
            "context": c.get("crossattn", None), # 传递字典或 None
            "y": c.get("vector", None),
            "num_video_frames": num_video_frames, # <<<--- 传递 num_video_frames
            **kwargs # 传递其他未显式处理的 kwargs
        }
        # 根据需要添加 cond_view, cond_motion
        if "cond_view" in c: unet_kwargs["cond_view"] = c.get("cond_view")
        if "cond_motion" in c: unet_kwargs["cond_motion"] = c.get("cond_motion")

        return self.diffusion_model(x, **unet_kwargs) # <<<--- 使用解包传递