""" VibeVoice Streaming model configuration

Cherry-picked from vibevoice-community/VibeVoice (MIT License).
Import paths adjusted for vendored use in Voice Clone Studio.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .configuration_vibevoice import VibeVoiceAcousticTokenizerConfig, VibeVoiceDiffusionHeadConfig

logger = logging.get_logger(__name__)


class VibeVoiceStreamingConfig(PretrainedConfig):
    """
    Configuration class for the VibeVoice Streaming model (0.5B).

    The streaming model differs from the multi-speaker model:
    - No semantic tokenizer (only acoustic)
    - Split language model: lower layers for text encoding, upper layers for TTS
    - Optimized for low-latency real-time generation
    """
    model_type = "vibevoice_streaming"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        tts_backbone_num_hidden_layers=20,
        **kwargs
    ):

        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, (Qwen2Config,)):
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            self.diffusion_head_config = diffusion_head_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers

        # Expose decoder's num_hidden_layers at top level so that
        # transformers DynamicCache (which reads config.num_hidden_layers)
        # works without requiring get_text_config() to resolve correctly.
        self.num_hidden_layers = self.decoder_config.num_hidden_layers

        super().__init__(**kwargs)

__all__ = [
    "VibeVoiceStreamingConfig"
]
