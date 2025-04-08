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
        self.f_low_proj_layer = None
        if self.f_low_injection:
             # --- 获取 VAE latent 通道数 (通常是 4) ---
             # 尝试从 VAE 配置获取
             vae_latent_channels = first_stage_config.get('params',{}).get('ddconfig',{}).get('z_channels', 4)
             # --- 获取 f_low 通道数 (来自 AttributeEncoder，硬编码或从配置读取) ---
             # 假设 f_low 通道固定为 320 (根据 encoder.py)
             f_low_channels = 320
             if attribute_encoder_instance is not None:
                  # (可选) 尝试从实例获取，如果 Encoder 有属性的话
                  # f_low_channels = getattr(attribute_encoder_instance, 'output_low_channels', 320)
                  pass

             if f_low_channels != vae_latent_channels:
                  logger.info(f"Initializing 1x1 Conv layer to project f_low from {f_low_channels} to {vae_latent_channels} channels.")
                  self.f_low_proj_layer = nn.Conv2d(
                      in_channels=f_low_channels,
                      out_channels=vae_latent_channels,
                      kernel_size=1
                  )
                  # 这个投影层是否需要训练？通常需要，将其参数添加到优化器
                  # (configure_optimizers 会自动处理，因为它是 self 的属性)
             else:
                  logger.info(f"f_low channels ({f_low_channels}) match VAE latent channels ({vae_latent_channels}). No projection needed.")
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
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor: # 完整实现
        vae_device = next(self.first_stage_model.parameters()).device
        z = 1.0 / self.scale_factor * z.to(vae_device)
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        try:
            with torch.autocast(self.device.type, enabled=not self.disable_first_stage_autocast):
                for n in range(n_rounds):
                    z_batch = z[n * n_samples : (n + 1) * n_samples]
                    out = self.first_stage_model.decode(z_batch)
                    all_out.append(out.cpu()) # Decode on device, move to CPU after
            out = torch.cat(all_out, dim=0)
        except Exception as e:
            logger.error(f"VAE decoding failed: {e}", exc_info=True)
            bs, _, h, w = z.shape
            out_c = getattr(self.first_stage_model.decoder, 'out_channels', 3) # Guess output channels
            out_res_mult = getattr(self.first_stage_model, 'scale_factor_reciprocal', 8) # Guess scale factor
            out = torch.zeros(bs, out_c, h * out_res_mult, w * out_res_mult, dtype=torch.float32)
        return out.to(self.device) # Move final result to main device

    @torch.no_grad()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        # self.first_stage_model.to(x.device) # PL 会处理
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        try:
            with torch.autocast(self.device.type, enabled=not self.disable_first_stage_autocast):
                for n in range(n_rounds):
                    x_batch = x[n * n_samples : (n + 1) * n_samples]
                    # VAE encoder 通常返回分布对象
                    encoder_posterior = self.first_stage_model.encode(x_batch)
                    # 获取潜变量样本
                    if isinstance(encoder_posterior, torch.Tensor): # 有些 VAE 直接返回 Tensor
                         out = encoder_posterior
                    elif hasattr(encoder_posterior, 'sample'):
                         out = encoder_posterior.sample()
                    elif hasattr(encoder_posterior, 'mode'):
                         out = encoder_posterior.mode()
                    else:
                         raise TypeError(f"VAE encode 返回了未知类型: {type(encoder_posterior)}")
                    all_out.append(out)
            z = torch.cat(all_out, dim=0)
            z = self.scale_factor * z
        except Exception as e:
            logger.error(f"VAE 编码失败: {e}", exc_info=True)
            z = torch.zeros(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8, device=x.device) # 假设 latent 通道为4
            # raise e
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
        if self.f_low_injection and self.attribute_encoder is not None:
            f_low = None # 初始化 f_low
            # 使用 no_grad 计算 f_low，因为我们通常不希望 f_low 注入影响 Eattr 梯度
            # 如果 Eattr 浅层也需要训练，需要移除 no_grad
            with torch.no_grad(): # <<<--- 推荐使用 no_grad 计算 f_low
                try:
                    if x.ndim == 4:
                        _, f_low = self.attribute_encoder(x) # (B*T, 320, H, W)
                    else: logger.warning(...)
                except Exception as e_flow: logger.warning(...); f_low = None

            if f_low is not None:
                # --- 应用投影层 (如果存在) ---
                if self.f_low_proj_layer is not None:
                    try:
                        # 投影层也应该在 no_grad 上下文中，除非你想训练它
                        # 但通常 f_low 注入的目的是提供信息，不参与梯度
                        f_low_projected = self.f_low_proj_layer(f_low) # (B*T, 4, H, W)
                        logger.debug(f"Projected f_low shape: {f_low_projected.shape}")
                    except Exception as e_proj:
                        logger.error(f"Failed to project f_low: {e_proj}", exc_info=True)
                        f_low_projected = None
                else:
                    # 如果不需要投影 (通道数已匹配)
                    f_low_projected = f_low
                # --- 投影结束 ---

                # --- 注入投影后的 f_low ---
                if f_low_projected is not None:
                    if model_input.shape == f_low_projected.shape:
                        model_input = model_input + f_low_projected # <<<--- 使用投影后的 f_low
                        logger.debug("Projected f_low added to U-Net input.")
                    else:
                        # 检查通道数和空间尺寸
                        if model_input.shape[1] != f_low_projected.shape[1]:
                             logger.warning(f"Model input channels ({model_input.shape[1]}) and projected f_low channels ({f_low_projected.shape[1]}) mismatch after projection! Cannot inject.")
                        elif model_input.shape[2:] != f_low_projected.shape[2:]:
                             logger.warning(f"Model input spatial size ({model_input.shape[2:]}) and projected f_low ({f_low_projected.shape[2:]}) mismatch! Cannot inject.")
                        else: # 其他未知形状问题
                             logger.warning(f"Model input shape ({model_input.shape}) and projected f_low shape ({f_low_projected.shape}) mismatch! Cannot inject.")
                # --- 注入结束 ---
        # --- U-Net 调用 (现在不需要 additional_model_inputs 字典了) ---
        try:
            x_pred = self.denoiser(self.model, model_input, sigmas_t, cond=c,
                                   num_video_frames=num_frames) # <<<--- 直接传递显式参数
            # --- 修改结束 ---
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
        """准备数据并调用 forward 计算损失"""
        # 1. 获取 VAE latent 输入 x (来自 dataset 的 'vt')
        try:
            x_orig_shape = batch[self.input_key].shape # 记录原始形状 (B, T, C, H, W)
            x = self.get_input(batch) # get_input 应该返回 (B*T, C, H, W)
            if not (x.ndim == 4 and x.shape[0] == x_orig_shape[0] * x_orig_shape[1]):
                 logger.warning(f"get_input 返回的形状 ({x.shape}) 可能不符合预期的 B*T 格式 (原始: {x_orig_shape})")
                 # 根据需要添加错误处理或 reshape
                 if x.ndim == 5 and x.shape[0] == x_orig_shape[0] and x.shape[1] == x_orig_shape[1]:
                      x = x.view(-1, *x.shape[2:]) # 手动 reshape
                 else:
                      raise ValueError("无法将输入 reshape 为 (B*T, C, H, W)")

        except ValueError as e: logger.error(f"获取或 reshape 输入失败: {e}"); return torch.tensor(0.0, device=self.device), {}
        except KeyError as e: logger.error(f"获取输入失败: 输入键 '{e}' 不在 batch 中。"); return torch.tensor(0.0, device=self.device), {}

        # 2. --- 修改：确保 batch 中其他需要 Embedder 处理的数据也被 reshape ---
        batch_for_cond = {}
        batch_size = x_orig_shape[0] # B
        num_frames = x_orig_shape[1] # T

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 如果是 5D 张量 (B, T, ...)， reshape 成 (B*T, ...)
                if value.ndim == 5 and value.shape[0] == batch_size and value.shape[1] == num_frames:
                    batch_for_cond[key] = value.view(batch_size * num_frames, *value.shape[2:])
                # 如果是 3D 张量 (B, C, H, W) - 例如 source_image
                elif value.ndim == 4 and value.shape[0] == batch_size:
                    # 需要为每个时间步复制 B 次
                    # 假设 Embedder 会处理 B*T 的输入
                    # 如果 Embedder 只处理 B，这里需要调整
                    batch_for_cond[key] = value.repeat_interleave(num_frames, dim=0) # (B*T, C, H, W)
                # 如果是 2D 张量 (B, D) - 例如 fgid, frid
                elif value.ndim == 2 and value.shape[0] == batch_size:
                     batch_for_cond[key] = value.repeat_interleave(num_frames, dim=0) # (B*T, D)
                # 如果已经是 (B*T, ...) 形状 (例如我们之前 repeat 的 fgid)
                elif value.shape[0] == batch_size * num_frames:
                     batch_for_cond[key] = value
                # 其他情况暂时保持不变 (例如 is_same_identity 可能已经是 B*T)
                else:
                    batch_for_cond[key] = value
                    # logger.debug(f"Tensor '{key}' shape {value.shape} 未被 reshape。")
            else:
                batch_for_cond[key] = value # 非 Tensor 数据直接复制

        # 使用 reshape 后的 batch_for_cond 获取条件
        try:
             c, uc = self.conditioner.get_unconditional_conditioning(batch_for_cond)
             # 将 c 和 uc 添加回 batch_for_cond (或原始 batch) 以便 forward 访问
             batch_for_cond["c"] = c
             batch_for_cond["uc"] = uc
        except Exception as e_cond:
             logger.error(f"获取条件失败: {e_cond}", exc_info=True)
             return torch.tensor(0.0, device=self.device), {}
        # --- 修改结束 ---


        # 3. 添加全局步数
        batch_for_cond["global_step"] = self.global_step

        # 4. 调用 forward 计算损失 (传递 reshape 后的 batch_for_cond)
        loss, loss_dict = self.forward(x, batch_for_cond) # <<<--- 传递修改后的 batch

        return loss, loss_dict


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