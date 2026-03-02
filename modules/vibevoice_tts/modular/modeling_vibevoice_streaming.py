"""
VibeVoice Streaming Model Architecture (0.5B)

Cherry-picked from vibevoice-community/VibeVoice (MIT License).
Import paths adjusted for vendored use in Voice Clone Studio.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from modules.vibevoice_tts.schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig


logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


class BinaryClassifier(nn.Module):
    """Binary classifier for end-of-speech detection in streaming TTS."""

    def __init__(self, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpeechConnector(nn.Module):
    """Connector module that projects speech latents to language model hidden dimension."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class VibeVoiceStreamingPreTrainedModel(PreTrainedModel):
    """Base class for VibeVoice Streaming models."""

    config_class = VibeVoiceStreamingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, VibeVoiceDiffusionHead):
            module.initialize_weights()
            return

        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class VibeVoiceStreamingModel(VibeVoiceStreamingPreTrainedModel):
    """
    VibeVoice Streaming Model for real-time TTS.

    The model uses a split architecture:
    - language_model: Lower transformer layers for text encoding
    - tts_language_model: Upper transformer layers for TTS generation
    """

    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32

        lm_config = copy.deepcopy(config.decoder_config)
        lm_backbone_num_hidden_layers = getattr(lm_config, 'num_hidden_layers', 24) - config.tts_backbone_num_hidden_layers
        lm_config.num_hidden_layers = lm_backbone_num_hidden_layers
        self.language_model = AutoModel.from_config(lm_config)
        self.language_model.norm = nn.Identity()

        tts_lm_config = copy.deepcopy(lm_config)
        tts_lm_config.num_hidden_layers = config.tts_backbone_num_hidden_layers
        self.tts_language_model = AutoModel.from_config(tts_lm_config)

        self.tts_input_types = nn.Embedding(num_embeddings=2, embedding_dim=config.decoder_config.hidden_size)

        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).to(dtype)
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size).to(dtype)

        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        self.prediction_head = AutoModel.from_config(config.diffusion_head_config).to(dtype)

        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )

    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            return self.language_model.embed_tokens

        for name, attr in self.language_model.fullmap.items():
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer

        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "VibeVoiceStreamingModel.forward is intentionally disabled. "
            "Use `model.language_model(...)` or `model.tts_language_model(...)` instead."
        )


AutoModel.register(VibeVoiceStreamingConfig, VibeVoiceStreamingModel)

__all__ = [
    "VibeVoiceStreamingPreTrainedModel",
    "VibeVoiceStreamingModel",
    "BinaryClassifier",
    "SpeechConnector",
]
