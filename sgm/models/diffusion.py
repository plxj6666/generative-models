# sgm/models/diffusion.py

import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
import logging
import numpy as np  
import cv2
import torch.nn as nn
from ..modules.encoders.modules import AbstractEmbModel # <--- 顶部的导入仍然需要
from ..modules.diffusionmodules.denoiser import DiscreteDenoiser
import sys
from pathlib import Path

# --- 强制优先使用本地项目目录 ---
# 获取当前脚本文件所在的目录
script_dir = Path(__file__).resolve().parent
# 假设脚本就在 /root/HiFiVFS/ 下，则项目根目录就是 script_dir
# 如果脚本在 /root/HiFiVFS/scripts/ 下，则项目根目录是 script_dir.parent
project_root = script_dir # 或者 script_dir.parent，取决于脚本位置
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"--- INFO: Prepended project root to sys.path: {project_root} ---")
else:
    # 如果已存在，确保它在最前面
    sys.path.remove(str(project_root))
    sys.path.insert(0, str(project_root))
    print(f"--- INFO: Moved project root to the front of sys.path: {project_root} ---")
print(f"--- INFO: Current sys.path[0]: {sys.path[0]} ---")
# --- 路径修改结束 ---


# --- 导入 HiFiVFS 相关模块 ---
# 确保路径正确
try:
    from fal_dil.models.encoder import AttributeEncoder
    from fal_dil.utils.face_recognition import DeepFaceRecognizer
    from fal_dil.models.dit import DetailedIdentityTokenizer
    from fal_dil import losses as hifivfs_losses # <--- 注意这里
    from fal_dil.utils.vae_utils import decode_with_vae, convert_tensor_to_cv2_images
    print("成功导入 HiFiVFS 相关模块。")
except ImportError as e:
    print(f"警告：导入 HiFiVFS FAL/DIL 相关模块失败: {e}。Lid 和 LFAL 损失将不可用。")
    # AttributeEncoder = None
    # DeepFaceRecognizer = None
    # hifivfs_losses = None
    # decode_with_vae = None
    # convert_tensor_to_cv2_images = None
# --- 导入结束 ---


from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.denoiser import Denoiser # <--- 确保导入 Denoiser
from ..modules.encoders.modules import GeneralConditioner # <--- 确保导入 GeneralConditioner
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img, append_dims)

logger = logging.getLogger(__name__)

class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config: Dict,
        denoiser_config: Dict,
        first_stage_config: Dict,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        input_key: str = "vt", # 默认输入键为 'vt' (VAE latent)
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,

        # --- HiFiVFS 相关参数 ---
        attribute_encoder_config: Optional[Dict] = None,
        face_recognizer_config: Optional[Dict] = None,
        hifivfs_loss_config: Optional[Dict] = None,
        f_low_injection: bool = True,
                # --- 新增：直接接收实例 ---
        attribute_encoder_instance: Optional[AttributeEncoder] = None,
        face_recognizer_instance: Optional[DeepFaceRecognizer] = None,
        dit_instance: Optional[DetailedIdentityTokenizer] = None, # <--- 新增 DIT 实例
        # --- 新增结束 ---
    ):
        super().__init__()
        ignore_list = ['ckpt_path', 'attribute_encoder_instance', 'face_recognizer_instance', 'dit_instance']
        # 保存超参数时，也忽略 conditioner_config，因为我们将手动处理它
        ignore_list.append('conditioner_config')
        self.save_hyperparameters(ignore=ignore_list)

        from svd.sgm.modules.encoders.modules import AbstractEmbModel, GeneralConditioner

        # --- 其他初始化 ---
        self.log_keys = log_keys; self.input_key = input_key; # ... (省略)
        self.optimizer_config = default( optimizer_config, {"target": "torch.optim.AdamW"} )

        model_instance = instantiate_from_config(network_config)
        self.model: nn.Module = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))( model_instance, compile_model=compile_model )
        self.denoiser: Denoiser = instantiate_from_config(denoiser_config)
        self.sampler = ( instantiate_from_config(sampler_config) if sampler_config is not None else None )
        self._init_first_stage(first_stage_config)
        self.loss_fn = ( instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None )
        self.use_ema = use_ema;
        if self.use_ema: self.model_ema = LitEma(self.model, decay=ema_decay_rate); logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.scale_factor = scale_factor; self.disable_first_stage_autocast = disable_first_stage_autocast; self.no_cond_log = no_cond_log; self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time; self.scheduler_config = scheduler_config

        # --- f_low 注入和投影层初始化 ---
        self.f_low_injection = f_low_injection
        # 移除f_low_proj_layer初始化和相关逻辑
        # 不再需要将f_low投影到VAE潜变量通道数
        # --- 初始化结束 ---

        # --- 直接使用传入的实例 ---
        self.attribute_encoder = attribute_encoder_instance
        self.face_recognizer = face_recognizer_instance

        # --- 彻底手动创建 Conditioner ---
        logger.info("Manually creating GeneralConditioner...")
        cond_cfg = default(conditioner_config, UNCONDITIONAL_CONFIG) # 获取原始配置
        cond_cfg_dict = OmegaConf.to_container(cond_cfg, resolve=True) if OmegaConf.is_config(cond_cfg) else cond_cfg

        initialized_embedders = []
        if isinstance(cond_cfg_dict, dict) and cond_cfg_dict.get('target') == 'svd.sgm.modules.encoders.modules.GeneralConditioner':
            emb_model_cfgs = cond_cfg_dict.get('params', {}).get('emb_models', [])
            logger.info(f"Found {len(emb_model_cfgs)} embedder configurations. Instantiating manually...")
            for n, emb_cfg_dict in enumerate(emb_model_cfgs):
                emb_target = emb_cfg_dict.get('target')
                if not emb_target: continue
                emb_params = emb_cfg_dict.get('params', {})
                embedder_instance = None
                try:
                    # 注入实例
                    if emb_target.endswith('DILEmbedder'):
                        emb_params['face_recognizer'] = face_recognizer_instance
                        emb_params['dit'] = dit_instance
                    elif emb_target.endswith('AttributeEmbedder'):
                        emb_params['attribute_encoder'] = self.attribute_encoder

                    # 直接实例化
                    embedder_cls = get_obj_from_str(emb_target)
                    embedder_instance = embedder_cls(**emb_params)

                    if isinstance(embedder_instance, nn.Module):
                        # 设置属性
                        embedder_instance.is_trainable = emb_cfg_dict.get("is_trainable", False)
                        embedder_instance.ucg_rate = emb_cfg_dict.get("ucg_rate", 0.0)
                        if "input_key" in emb_cfg_dict: embedder_instance.input_key = emb_cfg_dict["input_key"]
                        if "output_key" in emb_cfg_dict: setattr(embedder_instance, 'output_key', emb_cfg_dict["output_key"])
                        if not embedder_instance.is_trainable: # 冻结
                             embedder_instance.train = disabled_train
                             for param in embedder_instance.parameters(): param.requires_grad = False
                             embedder_instance.eval()
                        initialized_embedders.append(embedder_instance)
                        logger.info(f"  Successfully created instance for: {emb_target}")
                    else:
                         logger.error(f"  Instantiated object for {emb_target} is not an nn.Module. Skipping.")

                except Exception as e_emb:
                    logger.error(f"  Failed to create instance for {emb_target}: {e_emb}", exc_info=True)

            # 手动创建 GeneralConditioner 实例
            try:
                self.conditioner = GeneralConditioner(emb_models=initialized_embedders)
                logger.info(f"Manual GeneralConditioner initialized with {len(initialized_embedders)} embedders.")
            except Exception as e_gc:
                 logger.error(f"Manual initialization of GeneralConditioner failed: {e_gc}", exc_info=True)
                 self.conditioner = GeneralConditioner(emb_models=[]) # Fallback
        else:
             # 如果配置不是 GC 或无效，则创建一个空的
             logger.warning(f"Conditioner config invalid or not GeneralConditioner. Creating empty GeneralConditioner.")
             self.conditioner = GeneralConditioner(emb_models=[])
        # --- 手动创建 Conditioner 结束 ---


        
        # --- HiFiVFS 损失配置 ---
        self.hifivfs_loss_config = default(hifivfs_loss_config, {})
        self.lambda_lid = self.hifivfs_loss_config.get('lambda_lid', 0.0)
        self.lambda_fal = self.hifivfs_loss_config.get('lambda_fal', 0.0)
        self.fal_loss_weights = self.hifivfs_loss_config.get('fal_weights', {})
        self.f_low_injection = f_low_injection
        # 在这里重新检查依赖，因为现在实例是直接传入的
        self._recheck_hifivfs_dependencies()

        logger.info(f"HiFiVFS Losses (final): lambda_lid={self.lambda_lid}, lambda_fal={self.lambda_fal}")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def _recheck_hifivfs_dependencies(self):
         """根据传入的实例重新检查损失是否可用"""
         if self.lambda_lid > 0 and self.face_recognizer is None:
              logger.warning("Lid loss enabled, but face_recognizer instance is missing. Disabling Lid loss.")
              self.lambda_lid = 0.0
         if self.lambda_fal > 0 and self.attribute_encoder is None:
              logger.warning("LFAL loss enabled, but attribute_encoder instance is missing. Disabling LFAL loss.")
              self.lambda_fal = 0.0
         if self.lambda_fal > 0 and self.fal_loss_weights.get('identity', 0.0) > 0 and self.face_recognizer is None:
              logger.warning("Ltid loss enabled, but face_recognizer instance is missing. Disabling Ltid loss.")
              self.fal_loss_weights['identity'] = 0.0


    def _initialize_hifivfs_components(self, attr_enc_cfg, face_rec_cfg):
        """初始化 HiFiVFS 特定的模型组件"""
        if self.lambda_lid <= 0 and self.lambda_fal <= 0:
            logger.info("Lid 和 LFAL 损失权重均为 0，跳过 HiFiVFS 组件初始化。")
            return

        if AttributeEncoder is None or DeepFaceRecognizer is None or hifivfs_losses is None or decode_with_vae is None:
            logger.error("需要计算 Lid 或 LFAL 损失，但必要的 HiFiVFS 模块未能导入。禁用这些损失。")
            self.lambda_lid = 0.0
            self.lambda_fal = 0.0
            return

        # 初始化 Attribute Encoder
        if self.lambda_fal > 0:
            if attr_enc_cfg:
                try:
                    self.attribute_encoder = instantiate_from_config(attr_enc_cfg)
                    # 不需要手动 .to(self.device)，Pytorch Lightning 会处理
                    if any(p.requires_grad for p in self.attribute_encoder.parameters()):
                        self.attribute_encoder.train()
                    else:
                        self.attribute_encoder.eval()
                    logger.info("AttributeEncoder 初始化成功。")
                except Exception as e:
                    logger.error(f"初始化 AttributeEncoder 失败: {e}", exc_info=True)
                    self.attribute_encoder = None
                    self.lambda_fal = 0.0 # 禁用 FAL loss
            else:
                logger.warning("未提供 attribute_encoder_config，LFAL 损失将被禁用。")
                self.lambda_fal = 0.0

        # 初始化 Face Recognizer
        if self.lambda_lid > 0 or (self.lambda_fal > 0 and self.fal_loss_weights.get('identity', 0.0) > 0):
             # 如果需要 Lid 或 Ltid，则需要 face_recognizer
            if face_rec_cfg:
                try:
                    self.face_recognizer = instantiate_from_config(face_rec_cfg)
                    if not self.face_recognizer.initialized: raise RuntimeError("人脸识别器初始化失败")
                    logger.info("DeepFaceRecognizer 初始化成功。")
                except Exception as e:
                    logger.error(f"初始化 DeepFaceRecognizer 失败: {e}", exc_info=True)
                    self.face_recognizer = None
                    self.lambda_lid = 0.0
                    if 'identity' in self.fal_loss_weights:
                        logger.warning("由于 Face Recognizer 初始化失败，Ltid 损失也将被禁用。")
                        self.fal_loss_weights['identity'] = 0.0 # 注意这里修改的是实例属性
            else:
                logger.warning("未提供 face_recognizer_config，Lid 和 Ltid 损失将被禁用。")
                self.lambda_lid = 0.0
                if 'identity' in self.fal_loss_weights: self.fal_loss_weights['identity'] = 0.0


    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError(f"不支持的检查点格式: {path}")

        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info( # 使用 logger
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            # 只记录部分缺失键以避免日志过长
            logger.info(f"Missing Keys (sample): {missing[:10]}...")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys (sample): {unexpected[:10]}...")

    def _init_first_stage(self, config): # 实现
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters(): param.requires_grad = False
        self.first_stage_model = model
    def get_input(self, batch): # 实现
        x = batch[self.input_key]
        if x.ndim == 5: x = x.view(-1, *x.shape[2:])
        elif x.ndim != 4: raise ValueError(f"Input tensor ndim expected 4 or 5, got {x.ndim}")
        return x

    @torch.no_grad()
    def decode_first_stage(self, z):
        """确保能处理视频格式的VAE latent"""
        original_shape = z.shape
        is_video = len(original_shape) == 5  # (B,T,C,H,W)格式
        
        if is_video:
            B, T, C, H, W = original_shape
            z = z.reshape(B*T, C, H, W)
        
        # 正常解码
        result = self.first_stage_model.decode(z)
        
        # 恢复视频格式
        if is_video:
            _, C_out, H_out, W_out = result.shape
            result = result.reshape(B, T, C_out, H_out, W_out)
        
        return result
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        """编码，支持视频格式"""
        is_video = len(x.shape) == 5
        original_shape = x.shape
        
        if is_video:
            # 视频输入(B,T,C,H,W) -> 重塑为(B*T,C,H,W)
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
        
        if self.disable_first_stage_autocast:
            encoder_posterior = self.first_stage_model.encode(x)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                encoder_posterior = self.first_stage_model.encode(x)
        
        # 处理不同类型的返回值
        if hasattr(encoder_posterior, 'sample'):
            z = encoder_posterior.sample() * self.scale_factor
        else:
            # 如果直接返回Tensor，直接使用
            z = encoder_posterior * self.scale_factor
        
        # 恢复视频形状
        if is_video:
            # (B*T,C_latent,H_latent,W_latent) -> (B,T,C_latent,H_latent,W_latent)
            _, C_latent, H_latent, W_latent = z.shape
            z = z.reshape(B, T, C_latent, H_latent, W_latent)
        
        return z

    def forward(self, x: torch.Tensor, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        核心前向传播和损失计算。

        Args:
            x (torch.Tensor): 干净的目标视频 VAE latent (B*T, C, H, W)。
            batch (Dict): 包含条件的批次数据。

        Returns:
            Tuple[torch.Tensor, Dict]: 总损失 (标量) 和包含各损失分量的字典。
        """
        loss_dict = {}
        # 1. 采样时间和噪声
        # 1. 采样时间步 t
        num_train_timesteps = 1000 # 默认或从配置获取
        try:
             if isinstance(self.denoiser, DiscreteDenoiser): num_train_timesteps = self.denoiser.num_idx
             elif hasattr(self.loss_fn, "sigma_sampler") and hasattr(self.loss_fn.sigma_sampler, "num_steps"): num_train_timesteps = self.loss_fn.sigma_sampler.num_steps
             else: logger.warning(...)
        except AttributeError: logger.warning(...)

        t = torch.randint(0, num_train_timesteps, (x.shape[0],), device=self.device).long()

        # --- 修改：获取 sigma_t ---
        try:
             # 优先尝试 DiscreteDenoiser 的方法
             if isinstance(self.denoiser, DiscreteDenoiser):
                  sigmas_t = self.denoiser.idx_to_sigma(t) # <<<--- 使用 idx_to_sigma
             # 备选：如果 loss_fn 有 sigma_sampler 且能根据 t 获取 sigma (不常见)
             # elif hasattr(self.loss_fn, "sigma_sampler") and hasattr(self.loss_fn.sigma_sampler, "get_sigmas"):
             #      sigmas_t = self.loss_fn.sigma_sampler.get_sigmas(t)
             # 最后的备选：直接使用 loss_fn.sigma_sampler 采样，但这可能与 t 不对应
             elif hasattr(self.loss_fn, "sigma_sampler"):
                 logger.warning("Using sigmas sampled by loss_fn.sigma_sampler, may not correspond to t.")
                 sigmas_t = self.loss_fn.sigma_sampler(x.shape[0]).to(x.device) # 重新采样
             else:
                  raise RuntimeError("Cannot determine sigma from t.")
        except Exception as e_sigma:
             logger.error(f"获取 sigma_t 失败: {e_sigma}. 无法继续。")
             return torch.tensor(0.0, device=self.device, requires_grad=True), loss_dict
        # --- 修改结束 ---
        noise = torch.randn_like(x)

        # 2. 计算加噪输入 xt
        sigmas_bc = append_dims(sigmas_t, x.ndim)
        xt = x + noise * sigmas_bc # 直接使用 sigma 加噪 (对应 VE SDE 或简化的 VP)

        # 3. 获取条件 c (包含 DIL 和 f_attr tokens)
        # 在 shared_step 中已经准备好 batch
        # try:
        #     c, _ = self.conditioner.get_unconditional_conditioning(batch)
        # except Exception as e_cond:
        #      logger.error(f"获取条件失败: {e_cond}", exc_info=True)
        #      # 返回零损失和空字典
        #      return torch.tensor(0.0, device=self.device, requires_grad=True), loss_dict
        # --- 改为在外部获取条件，forward 只负责计算 ---
        c = batch.get("c", {}) # 假设条件在 batch 中准备好
        uc = batch.get("uc", {}) # 假设 UC 在 batch 中准备好

                # --- 修改：从原始 batch 数据获取 num_frames ---
        # --- 修改：强制从 batch 获取 num_frames ---
        num_frames = batch.get('num_video_frames')
        if num_frames is None:
             logger.error("Batch is missing the required key 'num_video_frames'!")
             # 你需要确定 Dataset 确实返回了这个键
             # 返回错误，训练无法继续
             return torch.tensor(0.0, device=self.device, requires_grad=True), loss_dict
        # 确保 num_frames 是 Python int (如果 Dataset 返回 Tensor 的话)
        if isinstance(num_frames, torch.Tensor):
             num_frames = num_frames.item() # 或者 .tolist()[0] 取决于形状
        logger.debug(f"[DiffusionEngine.forward] Got num_frames = {num_frames} from batch.")
        if num_frames is None: logger.error(...); return ...
        logger.debug(f"[DiffusionEngine.forward] Determined num_frames = {num_frames}")
        
        # 4. 处理 f_low 注入 (如果启用)
        model_input = xt
        f_attr_low = None
        if self.f_low_injection and self.attribute_encoder is not None:
            with torch.no_grad():
                # 从AttributeEncoder获取f_attr_low
                f_attr_low = self.attribute_encoder.extract_low_level_features(x)

        try:
            x_pred = self.denoiser(self.model, model_input, sigmas_t, cond=c,
                                   num_video_frames=num_frames, f_attr_low=f_attr_low) # <<<--- 直接传递显式参数
        except Exception as e_unet:
             logger.error(f"U-Net/Denoiser 前向传播失败: {e_unet}", exc_info=True)
             return torch.tensor(0.0, device=self.device, requires_grad=True), loss_dict

        # 6. 计算核心 LDM 损失
        ldm_loss_mean = torch.tensor(0.0, device=self.device)
        if self.loss_fn:
            try:
                # loss_fn(model_output, target, w)
                # model_output 是 x_pred, target 是 x
                w = append_dims(self.loss_fn.loss_weighting(sigmas_t), x.ndim)
                # 直接调用 get_loss (因为 loss_fn 的 forward 包含采样等步骤)
                ldm_loss = self.loss_fn.get_loss(x_pred, x, w)
                ldm_loss_mean = ldm_loss.mean() # StandardDiffusionLoss 返回的是每个样本的 loss
                loss_dict['loss_ldm'] = ldm_loss_mean.item()
            except Exception as e_ldm:
                logger.error(f"计算 LDM 损失失败: {e_ldm}", exc_info=True)
                # 如果 LDM 失败，后续损失意义不大，可以提前返回
                return torch.tensor(0.0, device=self.device, requires_grad=True), loss_dict
        else:
            logger.warning("loss_fn 未配置，无法计算 LDM 损失。")


        # 7. 计算附加损失 (Lid, LFAL)
        lid_loss_mean = torch.tensor(0.0, device=self.device)
        lfal_loss_mean = torch.tensor(0.0, device=self.device)

        # 7.1 计算 Lid
        if self.lambda_lid > 0 and self.face_recognizer is not None and decode_with_vae is not None:
            try:
                with torch.no_grad():
                    # 缓存 VAE 到 CPU 以节省 GPU 显存
                    # vae_device = self.first_stage_model.device
                    # self.first_stage_model.to('cpu')
                    # # 解码，确保输入是 float
                    # vr_pixel_pred = decode_with_vae(self.first_stage_model, x_pred.float().cpu(), self.scale_factor)
                    # self.first_stage_model.to(vae_device) # 移回原设备

                    # **优化：如果显存足够，直接在 GPU 解码**
                    vae_device = next(self.first_stage_model.parameters()).device # 获取VAE当前设备
                    vr_pixel_pred = decode_with_vae(self.first_stage_model, x_pred.to(vae_device), self.scale_factor)


                    # 转换图像格式 (需要确保函数可用)
                    if convert_tensor_to_cv2_images:
                         vr_cv2_images = convert_tensor_to_cv2_images(vr_pixel_pred)
                    else:
                         raise RuntimeError("convert_tensor_to_cv2_images 函数不可用")

                # 提取 fgid (假设 face_recognizer 在 CPU 或能处理 GPU Tensor)
                fgid_pred = self._extract_batch_fgid(vr_cv2_images) # 封装提取逻辑
                fgid_pred = fgid_pred.to(self.device) # 确保在主设备

                # 获取源 fgid
                fgid_source = batch['fgid'].to(self.device)
                if fgid_source.ndim == 3: fgid_source = fgid_source.view(-1, fgid_source.shape[-1])

                # 计算损失
                if fgid_source.shape == fgid_pred.shape:
                    lid_loss = 1.0 - F.cosine_similarity(fgid_source, fgid_pred, dim=1)
                    lid_loss_mean = lid_loss.mean()
                    if torch.isnan(lid_loss_mean): lid_loss_mean = torch.tensor(0.0, device=self.device)
                    loss_dict['loss_lid'] = lid_loss_mean.item()
                else: logger.warning(f"Lid 计算失败：fgid_source ({fgid_source.shape}) 和 fgid_pred ({fgid_pred.shape}) 形状不匹配。")

            except Exception as e_lid:
                logger.warning(f"计算 Lid 损失失败: {e_lid}", exc_info=True)
                lid_loss_mean = torch.tensor(0.0, device=self.device)
                if 'fgid_pred' in locals(): del fgid_pred # 清理可能存在的变量

        # 7.2 计算 LFAL
        if self.lambda_fal > 0 and self.attribute_encoder is not None and hifivfs_losses is not None:
            try:
                # 确保 fgid_pred 存在 (如果需要 Ltid)
                lambda_tid = self.fal_loss_weights.get('identity', 0.0)
                if lambda_tid > 0 and 'fgid_pred' not in locals():
                     # 如果 Lid 没计算或失败，尝试再次计算 fgid_pred
                     if self.face_recognizer and decode_with_vae and convert_tensor_to_cv2_images:
                          logger.info("为 Ltid 重新计算 fgid_pred...")
                          try:
                              with torch.no_grad():
                                   # vae_device = self.first_stage_model.device
                                   # self.first_stage_model.to('cpu')
                                   # vr_pixel_pred = decode_with_vae(self.first_stage_model, x_pred.float().cpu(), self.scale_factor)
                                   # self.first_stage_model.to(vae_device)
                                   vae_device = next(self.first_stage_model.parameters()).device
                                   vr_pixel_pred = decode_with_vae(self.first_stage_model, x_pred.to(vae_device), self.scale_factor)
                                   vr_cv2_images = convert_tensor_to_cv2_images(vr_pixel_pred)
                              fgid_pred = self._extract_batch_fgid(vr_cv2_images).to(self.device)
                          except Exception as e_fgid_retry:
                              logger.warning(f"为 Ltid 重计算 fgid_pred 失败: {e_fgid_retry}")
                     else:
                          logger.warning("无法为 Ltid 计算 fgid_pred，相关模块缺失。")

                # 获取其他所需数据
                frid = batch['frid'].to(self.device)
                is_same_identity = batch['is_same_identity'].to(self.device)
                if frid.ndim == 3: frid = frid.view(-1, frid.shape[-1])
                if is_same_identity.ndim == 3: is_same_identity = is_same_identity.view(-1, is_same_identity.shape[-1])
                if 'fgid_source' not in locals(): # 确保 fgid_source 存在
                     fgid_source = batch['fgid'].to(self.device)
                     if fgid_source.ndim == 3: fgid_source = fgid_source.view(-1, fgid_source.shape[-1])


                # 计算各分量损失
                loss_attr = torch.tensor(0.0, device=self.device)
                loss_rec = torch.tensor(0.0, device=self.device)
                loss_tid = torch.tensor(0.0, device=self.device)

                lambda_attr = self.fal_loss_weights.get('attribute', 0.0)
                if lambda_attr > 0:
                    # attribute_encoder 需要梯度，不能用 no_grad
                    f_attr_orig, _ = self.attribute_encoder(x)
                    f_attr_pred, _ = self.attribute_encoder(x_pred)
                    loss_attr = hifivfs_losses.compute_attribute_loss(f_attr_orig, f_attr_pred)
                    loss_dict['loss_attr'] = loss_attr.item()

                lambda_rec = self.fal_loss_weights.get('reconstruction', 0.0)
                if lambda_rec > 0:
                    loss_rec = hifivfs_losses.compute_reconstruction_loss(x, x_pred, is_same_identity, loss_type='l1')
                    loss_dict['loss_rec'] = loss_rec.item()

                if lambda_tid > 0:
                    if 'fgid_pred' in locals():
                         loss_tid = hifivfs_losses.compute_triplet_identity_loss(
                             fgid_source, fgid_pred, frid, is_same_identity,
                             margin=self.fal_loss_weights.get('identity_margin', 0.5)
                         )
                         loss_dict['loss_tid'] = loss_tid.item()
                    else:
                         logger.warning("无法计算 Ltid，因为 fgid_pred 不可用。")

                lfal_loss_mean = (lambda_attr * loss_attr +
                                  lambda_rec * loss_rec +
                                  lambda_tid * loss_tid)
                loss_dict['loss_lfal'] = lfal_loss_mean.item()

            except Exception as e_fal:
                logger.warning(f"计算 LFAL 损失失败: {e_fal}", exc_info=True)
                lfal_loss_mean = torch.tensor(0.0, device=self.device)


        # 8. 计算总损失
        total_loss_mean = ldm_loss_mean + self.lambda_lid * lid_loss_mean + self.lambda_fal * lfal_loss_mean
        loss_dict['loss'] = total_loss_mean.item()

        return total_loss_mean, loss_dict

    def _extract_batch_fgid(self, cv2_images: List[np.ndarray]) -> torch.Tensor:
        """辅助函数：批量提取 fgid"""
        if self.face_recognizer is None:
             logger.warning("Face recognizer 未初始化，无法提取 fgid。返回零向量。")
             # 需要知道 embedding size
             embed_size = 512 # 默认值，或者从配置读取
             try: embed_size = self.face_recognizer.embedding_size
             except: pass
             return torch.zeros((len(cv2_images), embed_size), dtype=torch.float32)

        fgid_list = []
        for img in cv2_images:
            fgid_p = self.face_recognizer.get_embedding(img) # 假设接受 BGR uint8
            if fgid_p is None:
                fgid_p = np.zeros(self.face_recognizer.embedding_size, dtype=np.float32)
            fgid_list.append(fgid_p)
        fgid_batch = torch.from_numpy(np.stack(fgid_list)).float()
        return fgid_batch

    def shared_step(self, batch: Dict) -> Any:
        """处理一个训练批次，计算损失"""
        try:
            # 获取输入
            x = self.get_input(batch)
            
            # 获取条件
            try:
                c, uc = self.conditioner.get_unconditional_conditioning(batch)
            except Exception as e:
                logger.error(f"获取条件失败", exc_info=True)
                # 修改：返回统一格式的元组
                return torch.tensor(0.0, device=self.device), {}
            
            # 1. 获取输入（VAE潜变量）
            x = self.get_input(batch)
            
            # 2. 获取条件（DIL、属性token等）
            try:
                c, uc = self.conditioner.get_unconditional_conditioning(batch)
            except:
                # 如果条件处理失败，记录详细错误
                logger.error(f"获取条件失败", exc_info=True)
                return torch.tensor(0.0, device=self.device)
            
            # 3. 获取时间步和噪声
            num_train_timesteps = 1000  # 默认值
            try:
                if isinstance(self.denoiser, DiscreteDenoiser):
                    num_train_timesteps = self.denoiser.num_idx
                elif hasattr(self.loss_fn, "sigma_sampler") and hasattr(self.loss_fn.sigma_sampler, "num_steps"):
                    num_train_timesteps = self.loss_fn.sigma_sampler.num_steps
                else:
                    logger.warning("无法确定时间步总数，使用默认值 1000")
            except AttributeError:
                logger.warning("无法确定时间步总数，使用默认值 1000")
            
            t = torch.randint(0, num_train_timesteps, (x.shape[0],), device=self.device).long()
            
            # 获取 sigma_t
            try:
                if isinstance(self.denoiser, DiscreteDenoiser):
                    sigmas_t = self.denoiser.idx_to_sigma(t)
                elif hasattr(self.loss_fn, "sigma_sampler"):
                    sigmas_t = self.loss_fn.sigma_sampler(x.shape[0]).to(x.device)
                else:
                    raise RuntimeError("无法确定 sigma_t")
            except Exception as e_sigma:
                logger.error(f"获取 sigma_t 失败: {e_sigma}")
                return torch.tensor(0.0, device=self.device)
            
            # 4. 添加噪声到VAE潜变量 - 这就是Vmt+Noisy的部分
            noise = torch.randn_like(x)
            sigmas_bc = append_dims(sigmas_t, x.ndim) 
            noised_input = x + noise * sigmas_bc  # 这是将要输入UNet的基础内容
            
            # 5. 【修改】提取f_attr_low，用于在UNet内部注入
            # 修改shared_step方法中f_attr_low提取部分

            f_attr_low = None
            if self.f_low_injection and self.attribute_encoder is not None:
                with torch.no_grad():
                    try:
                        # 从V'prime中提取f_attr_low特征
                        v_prime_latent = batch.get('v_prime_latent')
                        if v_prime_latent is not None:
                            # 处理维度
                            if v_prime_latent.ndim == 5:  # [B, T, C, H, W]
                                B, T, C, H, W = v_prime_latent.shape
                                v_prime_latent_reshaped = v_prime_latent.reshape(B*T, C, H, W)
                            else:
                                v_prime_latent_reshaped = v_prime_latent
                            
                            # 提取特征
                            f_attr_low = self.attribute_encoder.extract_low_level_features(v_prime_latent_reshaped.float())
                            
                            # 特征维度匹配检查：确保与输入批次大小匹配
                            input_batch_size = noised_input.shape[0]  # 应该是B*T
                            if f_attr_low.shape[0] != input_batch_size:
                                logger.warning(f"f_attr_low批次大小({f_attr_low.shape[0]})与输入({input_batch_size})不匹配，调整中...")
                                if f_attr_low.shape[0] == 1:
                                    # 如果只有一个样本，复制到所有帧
                                    f_attr_low = f_attr_low.repeat(input_batch_size, 1, 1, 1)
                                    logger.info(f"f_attr_low已扩展至匹配输入批次: {f_attr_low.shape}")
                            
                            logger.info(f"从V'提取f_attr_low特征，形状: {f_attr_low.shape}")
                        else:
                            logger.warning("未找到v_prime_latent，无法提取正确的f_attr_low")
                    except Exception as e:
                        logger.error(f"提取f_attr_low特征失败: {e}")
                        f_attr_low = None

            # 在shared_step方法中处理面部遮罩的部分
            # 6. 处理面部遮罩（如果有）
            mask = batch.get("mask")
            if mask is not None:
                # 使用提供的面部遮罩
                mask_lat = self.encode_first_stage(mask)
                c["concat"] = mask_lat
                logger.info("使用提供的面部遮罩")
            else:
                # 尝试动态生成面部遮罩
                try:
                    mask_lat = self.generate_face_mask(x)
                    c["concat"] = mask_lat
                    logger.debug(f"动态生成的面部遮罩形状: {mask_lat.shape}")
                except Exception as e:
                    logger.warning(f"动态生成面部遮罩失败: {e}")

            # 7. 确定视频帧数
            num_frames = batch.get('num_video_frames')
            if num_frames is None and x.ndim == 4:
                # 推断视频帧数
                if 'video_length' in batch:
                    num_frames = batch['video_length']
                else:
                    # 假设批次是(B*T, C, H, W)格式，猜测T
                    batch_size = batch.get('batch_size', 1)
                    if x.shape[0] % batch_size == 0:
                        num_frames = x.shape[0] // batch_size
                    else:
                        num_frames = 1  # 默认为单帧
            
            # 8. 调用模型 - 传递f_attr_low给UNetModel
            model_output = self.model(
                x=noised_input,
                t=t,
                c=c,
                num_video_frames=num_frames,
                f_attr_low=f_attr_low  # 【关键】传递f_attr_low给OpenAIWrapper
            )
            
            # 计算损失
            if self.loss_fn is not None:
                # 准备计算损失
                # 传递所有必要参数给forward方法
                loss, loss_dict = self.forward(x, {
                    'c': c,
                    'uc': uc,
                    'noise': noise,
                    'sigmas_t': sigmas_t,
                    't': t,
                    'noised_input': noised_input,
                    'num_video_frames': num_frames,
                    'fgid': batch.get('fgid', None),  # 用于Lid损失
                    'frid': batch.get('frid', None),  # 用于FAL损失
                    'is_same_identity': batch.get('is_same_identity', None)  # 用于FAL损失
                })
                return loss, loss_dict
            else:
                # 如果没有定义损失函数，则返回模型输出
                return model_output, {}
        except Exception as e:
            # 添加全局异常处理
            logger.error(f"shared_step发生未处理异常: {e}", exc_info=True)
            return torch.tensor(0.0, device=self.device), {}

    def training_step(self, batch: Dict, batch_idx: int) -> Optional[torch.Tensor]:
        """Pytorch Lightning 训练步骤"""
        loss, loss_dict = self.shared_step(batch)

        # 记录日志
        log_dict_filtered = {k: v for k, v in loss_dict.items() if isinstance(v, (int, float)) and not np.isnan(v)}
        self.log_dict(log_dict_filtered, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True) # sync_dist for DDP

        # 记录学习率
        self.log_learning_rate()

        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
             logger.error(f"训练步骤 {self.global_step} 检测到 NaN/Inf 损失，跳过优化。损失值: {loss.item()}")
             return None # 返回 None 会跳过优化器步骤

        return loss

    def log_learning_rate(self):
         """记录学习率"""
         if self.scheduler_config is not None:
              optimizers = self.optimizers()
              if optimizers:
                   opt = optimizers[0] if isinstance(optimizers, list) else optimizers
                   if opt and opt.param_groups:
                       lr = opt.param_groups[0]['lr']
                       self.log("lr_abs", lr, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
                   # else: logger.warning("无法记录学习率，优化器或参数组无效。") # 可能过于频繁
              # else: logger.warning("无法记录学习率，未找到优化器。")


    # --- configure_optimizers ---
    def configure_optimizers(self):
        # --- 从 optimizer_config 获取学习率 ---
        lr = self.optimizer_config.get('lr') # 尝试直接获取 lr
        if lr is None:
            # 如果 optimizer_config 中没有直接的 'lr'，尝试从 params 获取
            lr = self.optimizer_config.get('params', {}).get('lr')
        if lr is None:
            # 如果仍然没有，可能需要设置一个默认值或从 hparams 获取 (但不推荐)
            logger.warning("Learning rate not found directly in optimizer_config. Attempting to use self.hparams.learning_rate (may fail).")
            # 尝试从 hparams 获取作为最后的手段
            lr = self.hparams.get('learning_rate', 1e-4) # 提供一个默认值以防万一
            if lr is None: # 如果 hparams 也没有
                raise ValueError("Learning rate ('lr') must be defined in optimizer_config or hyperparameters.")
        # --- 修改结束 ---  
        params_to_optimize = []

        # 1. U-Net 参数 (self.model)
        params_to_optimize.extend(filter(lambda p: p.requires_grad, self.model.parameters()))
        logger.info(f"已添加 {len(list(filter(lambda p: p.requires_grad, self.model.parameters())))} 个可训练参数来自 U-Net (self.model)。")

        # 2. 可训练的 Conditioner Embedders
        trainable_embedders = []
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                trainable_params = list(filter(lambda p: p.requires_grad, embedder.parameters()))
                if trainable_params:
                     params_to_optimize.extend(trainable_params)
                     trainable_embedders.append(embedder.__class__.__name__)
                     logger.info(f"已添加 {len(trainable_params)} 个可训练参数来自 Embedder '{embedder.__class__.__name__}'。")

        # 3. AttributeEncoder (如果可训练)
        if self.attribute_encoder is not None and any(p.requires_grad for p in self.attribute_encoder.parameters()):
            trainable_params = list(filter(lambda p: p.requires_grad, self.attribute_encoder.parameters()))
            if trainable_params:
                 params_to_optimize.extend(trainable_params)
                 logger.info(f"已添加 {len(trainable_params)} 个可训练参数来自 AttributeEncoder。")

        if not params_to_optimize:
            logger.warning("优化器未配置，因为没有找到任何可训练的参数。")
            return None

        logger.info(f"总计 {len(params_to_optimize)} 组参数将被优化。")

        # 实例化优化器
        try:
            # 确保 optimizer_config 是字典
            opt_cfg_dict = OmegaConf.to_container(self.optimizer_config, resolve=True) if OmegaConf.is_config(self.optimizer_config) else self.optimizer_config
            opt = self.instantiate_optimizer_from_config(params_to_optimize, lr, opt_cfg_dict) # <<<--- 传递 lr
        except Exception as e_opt:
            logger.error(f"实例化优化器失败: {e_opt}", exc_info=True)
            raise

        # 配置学习率调度器
        if self.scheduler_config is not None:
            try:
                 scheduler_config = instantiate_from_config(self.scheduler_config)
                 # 使用 Pytorch Lightning 推荐的字典格式
                 scheduler_dict = {
                      "scheduler": LambdaLR(opt, lr_lambda=scheduler_config.schedule),
                      "interval": "step",
                      "frequency": 1,
                 }
                 logger.info(f"配置优化器和 LambdaLR 调度器。Trainable embedders: {trainable_embedders}")
                 return [opt], [scheduler_dict]
            except Exception as e_sched:
                 logger.error(f"初始化学习率调度器失败: {e_sched}", exc_info=True)
                 # 即使调度器失败，也返回优化器
                 logger.info(f"配置优化器 (无调度器)。Trainable embedders: {trainable_embedders}")
                 return opt
        else:
             logger.info(f"配置优化器 (无调度器)。Trainable embedders: {trainable_embedders}")
             return opt

    # --- 其他方法保持不变 ---
    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store()
            self.model_ema.copy_to()
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore()
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        target = cfg.get("target", "torch.optim.AdamW")
        logger.info(f"Instantiating optimizer {target} with learning rate {lr}")
        # 确保 cfg['params'] 存在且是字典
        optimizer_params = cfg.get("params", {})
        # 确保 lr 参数被传递
        optimizer_params['lr'] = lr # 显式设置 lr
        return get_obj_from_str(target)(
            params, **optimizer_params # 解包包含 lr 的参数字典
        )
    @torch.no_grad()
    def sample( self, cond: Dict, uc: Optional[Dict] = None, batch_size: int = 16, shape: Optional[Union[Tuple, List]] = None, **kwargs ):
        """模型推理采样"""
        if self.sampler is None:
            raise ValueError("Sampler is not defined.")
        if shape is None:
             # 尝试从 VAE 获取默认潜变量形状
             # 假设 VAE 有 latent_dim 和下采样因子
             try:
                  default_h = self.first_stage_model.encoder.ch * self.first_stage_model.encoder.resolution // self.first_stage_model.encoder.downsample_factor # 这可能不准确
                  default_w = default_h
                  default_c = self.first_stage_model.encoder.z_channels
                  shape = (default_c, default_h, default_w)
                  logger.info(f"Sample shape not provided, using default latent shape: {shape}")
             except AttributeError:
                  raise ValueError("Cannot determine default sample shape. Please provide 'shape'.")

        randn = torch.randn(batch_size, *shape).to(self.device)

        # 定义去噪器函数闭包
        denoiser_fn = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )

        # 调用 Sampler
        # 注意：CFG 通常在 Sampler 内部处理 uc
        samples_latent = self.sampler(denoiser_fn, randn, cond, uc=uc)
        return samples_latent # 返回潜变量样本

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """记录条件信息的可视化 (如果需要)"""
        # ... (可以根据需要实现或保留原样) ...
        return {}

    @torch.no_grad()
    def log_images( self, batch: Dict, N: int = 8, sample: bool = True, ucg_keys: Optional[List[str]] = None, **kwargs ) -> Dict:
        """记录输入、重建和样本的可视化"""
        log = dict()
        # 获取输入 (假设是 VAE latent)
        x = self.get_input(batch) # (B*T, C, H, W)
        N = min(x.shape[0], N) # 取 B*T 或 N 中较小者
        x = x[:N]

        # 获取条件
        if ucg_keys is None:
            ucg_keys = [e.input_key for e in self.conditioner.embedders if hasattr(e, 'input_key')]

        try:
            c, uc = self.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=ucg_keys
            )
            # 调整 c 和 uc 的 batch size 以匹配 N
            for k in c:
                 if isinstance(c[k], torch.Tensor) and c[k].shape[0] > N:
                      c[k] = c[k][:N]
            for k in uc:
                 if isinstance(uc[k], torch.Tensor) and uc[k].shape[0] > N:
                      uc[k] = uc[k][:N]

        except Exception as e_cond:
            logger.error(f"log_images 中获取条件失败: {e_cond}")
            c, uc = {}, {}


        log["inputs_latent"] = x
        # 解码输入潜变量得到原始图像近似值
        log["inputs"] = self.decode_first_stage(x)

        # 计算并记录重建 (可选，但有助于调试 VAE)
        try:
            # TODO: 这里简单地用 x 作为重建目标，实际可能需要更复杂的逻辑
            log["reconstructions"] = self.decode_first_stage(x)
        except Exception as e_rec:
             logger.warning(f"log_images 中解码失败: {e_rec}")


        # 记录条件信息 (可选)
        # log.update(self.log_conditionings(batch, N))

        # 生成样本
        if sample:
            with self.ema_scope("Sampling"):
                # 使用 x 的形状作为目标形状
                samples_latent = self.sample(
                    cond=c, uc=uc, batch_size=N, shape=x.shape[1:], **kwargs
                )
            log["samples_latent"] = samples_latent
            log["samples"] = self.decode_first_stage(samples_latent)

        # 将所有 Tensor 转换为 CPU 上的 float32，以方便 Pytorch Lightning 记录
        for k in log:
             if isinstance(log[k], torch.Tensor):
                  log[k] = log[k].detach().float().cpu()

        return log
    
    def generate_face_mask(self, x):
        """生成面部遮罩并编码到VAE潜空间"""
        # 确保有FaceParser
        if not hasattr(self, 'face_parser'):
            from fal_dil.utils.face_parsing import FaceParser
            self.face_parser = FaceParser(device=self.device)
        
        # 保存原始形状
        original_shape = x.shape
        is_video = len(original_shape) == 5
        
        # 先解码到像素空间 - 使用修改后的工具函数
        with torch.no_grad():
            # 替换直接调用为使用工具函数
            from fal_dil.utils.vae_utils import decode_with_vae
            pixel_x = decode_with_vae(self.first_stage_model, x, self.scale_factor)
        
        # 处理视频格式
        if is_video:
            B, T, C, H, W = pixel_x.shape
            masks = []
            for b in range(B):
                frame_masks = []
                for t in range(T):
                    frame = pixel_x[b, t]  # (C,H,W)
                    # 将tensor转换为numpy数组供FaceParser处理
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                    # 范围从[-1,1]转到[0,255]
                    frame_np = ((frame_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
                    # 执行面部解析
                    mask = self.face_parser.parse(frame_np)  # (H,W,1)
                    
                    # 将遮罩转换为Tensor
                    mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1).to(device=self.device)  # (1,H,W)
                    # 复制通道，转换为3通道格式
                    mask_tensor = mask_tensor.repeat(3, 1, 1)  # (3,H,W) <-- 关键修改
                    frame_masks.append(mask_tensor)
                
                batch_masks = torch.stack(frame_masks, dim=0)  # (T,3,H,W)
                masks.append(batch_masks)
            
            mask_tensor = torch.stack(masks, dim=0)  # (B,T,3,H,W)
        else:
            # 处理单帧图像
            masks = []
            for b in range(x.shape[0]):
                # 转换为numpy
                frame = pixel_x[b].permute(1, 2, 0).cpu().numpy()
                frame = ((frame + 1.0) / 2.0 * 255.0).astype(np.uint8)
                # 获取遮罩
                mask = self.face_parser.parse(frame)  # (H,W,1)
                mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1).to(device=self.device)  # (1,H,W)
                # 复制通道，转换为3通道格式
                mask_tensor = mask_tensor.repeat(3, 1, 1)  # (3,H,W) <-- 关键修改
                masks.append(mask_tensor)
            mask_tensor = torch.stack(masks, dim=0)  # (B,3,H,W)
        
        # 通过VAE编码到潜空间
        # 将遮罩值范围从[0,1]转换到[-1,1]（注意：虽然我们现在有3个通道，但值应该都一样）
        mask_normalized = mask_tensor * 2.0 - 1.0
        mask_latent = self.encode_first_stage(mask_normalized)
        
        return mask_latent