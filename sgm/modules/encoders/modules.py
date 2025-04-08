import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (ByT5Tokenizer, CLIPTextModel, CLIPTokenizer,
                          T5EncoderModel, T5Tokenizer)

from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import (extract_into_tensor,
                                              make_beta_schedule)
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)
from contextlib import nullcontext

class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

import logging
logger = logging.getLogger(__name__)
class GeneralConditioner(nn.Module): # 保留类定义和其他方法
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat"}
    # 定义默认的拼接维度
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
    # 可以为自定义键预定义拼接维度，如果需要的话
    CUSTOM_KEY2CATDIM = {"f_attr_tokens": 2} # 假设 f_attr_tokens 也按 sequence 维拼接

    # __init__ 和 possibly_get_ucg_val 方法保持不变
    def __init__(self, emb_models: List[AbstractEmbModel]): # <<<--- 参数类型改为实例列表
        super().__init__()
        # 直接使用传入的实例化好的 embedder 列表
        self.embedders = nn.ModuleList(emb_models)
        logger.info(f"GeneralConditioner received {len(self.embedders)} embedder instances.")

        
    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        # ... (代码不变) ...
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        # 确保 batch[embedder.input_key] 是列表或类似可迭代对象
        if not isinstance(batch[embedder.input_key], list):
             # 如果不是列表，可能需要特殊处理或发出警告
             # logger.warning(f"Input key '{embedder.input_key}' is not a list for legacy UCG.")
             # 假设它是一个单一的值或Tensor，直接处理
             if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                  batch[embedder.input_key] = val
        else:
             for i in range(len(batch[embedder.input_key])):
                  if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                       batch[embedder.input_key][i] = val
        return batch


    # --- 修改后的 forward 方法 ---
    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        """
        处理输入批次，调用所有 embedder，并将它们的输出整理到字典中。
        根据 embedder 的 output_key 属性或输出维度确定输出键。
        """
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        for embedder in self.embedders:
            # --- 修改这里：访问 _is_trainable ---

            is_trainable = getattr(embedder, '_is_trainable', False)
            embedding_context = nullcontext() if is_trainable else torch.no_grad()
            # logger.debug(f"Processing {embedder.__class__.__name__}, is_trainable={is_trainable}, context={type(embedding_context)}") # 添加调试日志

            with embedding_context:
                # --- 获取输入和处理 legacy_ucg_val ---
                try:
                    input_val = None
                    if hasattr(embedder, "input_key") and embedder.input_key:
                        if embedder.input_key not in batch:
                             logger.error(f"Embedder {embedder.__class__.__name__} input_key '{embedder.input_key}' not in batch!")
                             continue
                        input_val = batch[embedder.input_key]
                        # 访问 legacy_ucg_val 前先检查是否存在
                        if hasattr(embedder, 'legacy_ucg_val') and embedder.legacy_ucg_val is not None:
                             # possibly_get_ucg_val 可能需要修改以处理非列表情况
                             batch = self.possibly_get_ucg_val(embedder, batch)
                             input_val = batch[embedder.input_key] # 更新 input_val
                        emb_out = embedder(input_val)
                    elif hasattr(embedder, "input_keys"):
                        inputs = [batch[k] for k in embedder.input_keys]
                        emb_out = embedder(*inputs)
                    else:
                        logger.error(f"Embedder {embedder.__class__.__name__} has no input_key or input_keys.")
                        continue
                except KeyError as e:
                     logger.error(f"处理 Embedder {embedder.__class__.__name__} 时发生 KeyError: 输入键 '{e}' 不在 batch 中！可用的键: {list(batch.keys())}")
                     continue # 跳过这个 embedder
                except Exception as e:
                     logger.error(f"调用 Embedder {embedder.__class__.__name__} 时发生错误: {e}", exc_info=True)
                     continue # 跳过这个 embedder

            # 检查和处理输出
            if emb_out is None:
                 logger.warning(f"Embedder {embedder.__class__.__name__} 返回了 None，跳过。")
                 continue
            if not isinstance(emb_out, (torch.Tensor, list, tuple)):
                logger.error(f"Embedder {embedder.__class__.__name__} 的输出必须是 Tensor 或序列，但得到了 {type(emb_out)}。")
                continue
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out] # 统一处理为列表

            # 处理每个输出 Tensor
            for emb in emb_out:
                if not isinstance(emb, torch.Tensor):
                     logger.warning(f"Embedder {embedder.__class__.__name__} 的输出列表中包含非 Tensor 元素 ({type(emb)})，跳过。")
                     continue

                # 确定输出键 (out_key)
                out_key = None
                if hasattr(embedder, 'output_key') and embedder.output_key:
                    out_key = embedder.output_key
                elif emb.dim() in self.OUTPUT_DIM2KEYS:
                    out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                else:
                    # 尝试根据 input_key 猜测 (可选，或者直接报错/警告)
                    if hasattr(embedder, 'input_key'):
                         logger.warning(f"无法根据维度 {emb.dim()} 为 Embedder {embedder.__class__.__name__} (input: '{embedder.input_key}') 确定标准输出键。将尝试使用 input_key 作为 out_key。")
                         out_key = embedder.input_key # 使用 input_key 作为备选
                    else:
                         logger.error(f"无法确定 Embedder {embedder.__class__.__name__} 输出的 out_key (shape: {emb.shape})。跳过此输出。")
                         continue

                # 应用 UCG rate
                ucg_rate = getattr(embedder, '_ucg_rate', 0.0) # <<<--- 访问 _ucg_rate
                if ucg_rate > 0.0 and getattr(embedder, 'legacy_ucg_val', None) is None:
                    # 确保 emb 是浮点类型以应用 Bernoulli mask
                    if not torch.is_floating_point(emb):
                         logger.warning(f"Embedder {embedder.__class__.__name__} 的输出不是浮点类型 ({emb.dtype})，无法应用 UCG rate。")
                    else:
                         try:
                              # 需要处理 batch size 可能为 0 的情况
                              if emb.shape[0] > 0:
                                   mask = torch.bernoulli(
                                       (1.0 - embedder.ucg_rate)
                                       * torch.ones(emb.shape[0], device=emb.device)
                                   )
                                   emb = expand_dims_like(mask, emb) * emb
                              else:
                                   logger.debug(f"Embedder {embedder.__class__.__name__} 输出 batch size 为 0，跳过 UCG。")
                         except Exception as e_ucg:
                              logger.error(f"应用 UCG 到 Embedder {embedder.__class__.__name__} 时出错: {e_ucg}")


                # 应用 force_zero_embeddings
                if hasattr(embedder, 'input_key') and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)

                # 合并到 output 字典
                if out_key in output:
                    # 确定拼接维度
                    cat_dim = self.KEY2CATDIM.get(out_key) or self.CUSTOM_KEY2CATDIM.get(out_key)
                    if cat_dim is None:
                        # 默认行为：如果是 3D tensor 且键名含 'attn' 或 'token'，则按 sequence 维 (2) 拼，否则按 channel 维 (1)
                        if emb.ndim == 3 and ('attn' in out_key or 'token' in out_key): cat_dim = 2
                        else: cat_dim = 1
                        logger.debug(f"未找到键 '{out_key}' 的预定义拼接维度，使用默认值: {cat_dim}")

                    # 安全拼接
                    try:
                        if output[out_key].ndim == emb.ndim:
                            if output[out_key].shape[0] == emb.shape[0]: # 检查 Batch 维度是否一致
                                output[out_key] = torch.cat((output[out_key], emb), dim=cat_dim)
                            else:
                                logger.warning(f"键 '{out_key}' 的现有输出 Batch Size {output[out_key].shape[0]} 与新输出 Batch Size {emb.shape[0]} 不匹配，无法拼接。新输出被忽略。")
                        else:
                            logger.warning(f"键 '{out_key}' 的现有输出维度 {output[out_key].ndim} 与新输出维度 {emb.ndim} 不匹配，无法拼接。新输出被忽略。")
                    except Exception as e:
                        logger.error(f"拼接键 '{out_key}' 时出错 (现有形状: {output[out_key].shape}, 新形状: {emb.shape}, 维度: {cat_dim}): {e}")
                else:
                    output[out_key] = emb # 直接赋值

        # print(f"Debug: GeneralConditioner output keys: {list(output.keys())}")
        # for k, v in output.items():
        #      if isinstance(v, torch.Tensor): print(f"Debug: Output['{k}'] shape: {v.shape}")

        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        from pytorch_fid import inception

        kwargs["resize_input"] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    def forward(self, inp):
        outp = self.model(inp)

        if len(outp) == 1:
            return outp[0].squeeze()

        return outp


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-xxl", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self, version="google/byt5-base", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        init_device=None,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(default(init_device, "cpu")),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEmbModel):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
        wrap_video=False,
        kernel_size=1,
        remap_output=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LowScaleEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        linear_start,
        linear_end,
        timesteps=1000,
        max_noise_level=250,
        output_size=64,
        scale_factor=1.0,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):
    def __init__(
        self,
        n_cond_frames: int,
        n_copies: int,
        encoder_config: dict,
        sigma_sampler_config: Optional[dict] = None,
        sigma_cond_config: Optional[dict] = None,
        is_ae: bool = False,
        scale_factor: float = 1.0,
        disable_encoder_autocast: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = (
            instantiate_from_config(sigma_sampler_config)
            if sigma_sampler_config is not None
            else None
        )
        self.sigma_cond = (
            instantiate_from_config(sigma_cond_config)
            if sigma_cond_config is not None
            else None
        )
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_autocast = disable_encoder_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def forward(
        self, vid: torch.Tensor
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, dict],
        Tuple[Tuple[torch.Tensor, torch.Tensor], dict],
    ]:
        if self.sigma_sampler is not None:
            b = vid.shape[0] // self.n_cond_frames
            sigmas = self.sigma_sampler(b).to(vid.device)
            if self.sigma_cond is not None:
                sigma_cond = self.sigma_cond(sigmas)
                if self.n_cond_frames == 1:
                    sigma_cond = repeat(sigma_cond, "b d -> (b t) d", t=self.n_copies)
                else:
                    sigma_cond = repeat(sigma_cond, "b d -> (b t) d", t=self.n_cond_frames) # For SV4D
            sigmas = repeat(sigmas, "b -> (b t)", t=self.n_cond_frames)
            noise = torch.randn_like(vid)
            vid = vid + noise * append_dims(sigmas, vid.ndim)

        with torch.autocast("cuda", enabled=not self.disable_encoder_autocast):
            n_samples = (
                self.en_and_decode_n_samples_a_time
                if self.en_and_decode_n_samples_a_time is not None
                else vid.shape[0]
            )
            n_rounds = math.ceil(vid.shape[0] / n_samples)
            all_out = []
            for n in range(n_rounds):
                if self.is_ae:
                    out = self.encoder.encode(vid[n * n_samples : (n + 1) * n_samples])
                else:
                    out = self.encoder(vid[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)

        vid = torch.cat(all_out, dim=0)
        vid *= self.scale_factor

        if self.n_cond_frames == 1:
            vid = rearrange(vid, "(b t) c h w -> b () (t c) h w", t=self.n_cond_frames)
            vid = repeat(vid, "b 1 c h w -> (b t) c h w", t=self.n_copies)

        return_val = (vid, sigma_cond) if self.sigma_cond is not None else vid

        return return_val


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    def __init__(
        self,
        open_clip_embedding_config: Dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def forward(self, vid):
        vid = self.open_clip(vid)
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        return vid
