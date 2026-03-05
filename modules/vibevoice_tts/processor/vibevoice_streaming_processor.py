"""
VibeVoice Streaming Processor

Cherry-picked from vibevoice-community/VibeVoice (MIT License).
Import paths adjusted for vendored use in Voice Clone Studio.

This processor handles input preparation for the streaming 0.5B model,
including text tokenization and cached voice prompt handling.
"""

import math
import warnings
import os
import json
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from transformers.tokenization_utils_base import (
    BatchEncoding, PaddingStrategy, TruncationStrategy
)
from transformers.utils import TensorType, logging, cached_file
from .vibevoice_tokenizer_processor import AudioNormalizer

logger = logging.get_logger(__name__)


class VibeVoiceStreamingProcessor:
    """
    Processor for the VibeVoice Streaming 0.5B model.

    Wraps a VibeVoice tokenizer and audio processor. Uses pre-computed voice
    embeddings (.pt files) instead of live audio conditioning.

    Args:
        tokenizer: The tokenizer for text processing.
        audio_processor: The audio processor for speech processing.
        speech_tok_compress_ratio: Compression ratio for speech tokenization.
        db_normalize: Whether to apply decibel normalization.
    """

    def __init__(self, tokenizer=None, audio_processor=None,
                 speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Instantiate processor from a pretrained path or HuggingFace Hub."""
        from .vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
        from modules.vibevoice_tts.modular.modular_vibevoice_text_tokenizer import (
            VibeVoiceTextTokenizer,
            VibeVoiceTextTokenizerFast
        )

        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path,
                    "preprocessor_config.json",
                    **kwargs
                )
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json: {e}")
                logger.warning("Using default configuration")
                config = {
                    "speech_tok_compress_ratio": 3200,
                    "db_normalize": True,
                }

        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer
        language_model_pretrained_name = (
            config.get("language_model_pretrained_name", None)
            or kwargs.pop("language_model_pretrained_name", "Qwen/Qwen2.5-0.5B")
        )
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        if 'qwen' in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
                language_model_pretrained_name, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported tokenizer type for {language_model_pretrained_name}. Supported: Qwen."
            )

        # Load audio processor
        if "audio_processor" in config:
            audio_config = config["audio_processor"]
            audio_processor = VibeVoiceTokenizerProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
                eps=audio_config.get("eps", 1e-6),
            )
        else:
            audio_processor = VibeVoiceTokenizerProcessor()

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Save processor to a directory."""
        os.makedirs(save_directory, exist_ok=True)

        processor_config = {
            "processor_class": "VibeVoiceStreamingProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "VibeVoiceTokenizerProcessor",
                "sampling_rate": getattr(self.audio_processor, 'sampling_rate', 24000),
                "normalize_audio": getattr(self.audio_processor, 'normalize_audio', True),
                "target_dB_FS": getattr(self.audio_processor, 'target_dB_FS', -25),
                "eps": getattr(self.audio_processor, 'eps', 1e-6),
            }
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)

        logger.info(f"Processor configuration saved in {config_path}")

    def __call__(self):
        raise NotImplementedError(
            "VibeVoiceStreamingProcessor.__call__ is not implemented. "
            "Use process_input_with_cached_prompt for streaming inputs."
        )

    def process_input_with_cached_prompt(
        self,
        text=None,
        cached_prompt=None,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_attention_mask=True,
        **kwargs,
    ):
        """
        Process text with cached voice prompt for streaming generation.

        Args:
            text: The input text to process.
            cached_prompt: Dict containing pre-computed KV cache from .pt file.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
            max_length: Maximum sequence length.
            return_tensors: Output tensor type ("pt" for PyTorch).
            return_attention_mask: Whether to return attention mask.

        Returns:
            BatchEncoding with input_ids, attention_mask, tts_lm_input_ids, etc.
        """
        texts = [text]
        cached_prompts = [cached_prompt]

        all_encodings = []
        for text_input, cached_prompt_input in zip(texts, cached_prompts):
            script_tokens = self.tokenizer.encode(text_input.strip() + "\n", add_special_tokens=False)
            input_id_length = cached_prompt_input['lm']['last_hidden_state'].size(1)
            tts_lm_input_id_length = cached_prompt_input['tts_lm']['last_hidden_state'].size(1)

            input_ids = [self.tokenizer.pad_id] * input_id_length
            tts_lm_input_ids = [self.tokenizer.pad_id] * tts_lm_input_id_length
            speech_input_mask = [False] * tts_lm_input_id_length

            encoding = {
                "input_ids": input_ids,
                "tts_lm_input_ids": tts_lm_input_ids,
                "tts_text_ids": script_tokens,
                "speech_inputs": None,
                "speech_input_mask": speech_input_mask,
            }
            all_encodings.append(encoding)

        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

        return batch_encoding

    def _batch_encode(self, encodings, padding=True, truncation=False,
                      max_length=None, return_tensors=None, return_attention_mask=True):
        """Combine multiple encodings into a batch with padding."""
        input_ids_list = [enc["input_ids"] for enc in encodings]
        tts_lm_input_ids_list = [enc["tts_lm_input_ids"] for enc in encodings]
        tts_text_ids_list = [enc["tts_text_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]

        attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None
        tts_lm_attention_masks = [[1] * len(ids) for ids in tts_lm_input_ids_list] if return_attention_mask else None

        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True

        batch_encoding = BatchEncoding()

        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            batch_encoding["tts_lm_input_ids"] = torch.tensor(tts_lm_input_ids_list, dtype=torch.long)
            batch_encoding["tts_text_ids"] = torch.tensor(tts_text_ids_list, dtype=torch.long)

            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
                batch_encoding["tts_lm_attention_mask"] = torch.tensor(tts_lm_attention_masks, dtype=torch.long)

            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            batch_encoding["tts_lm_input_ids"] = tts_lm_input_ids_list
            batch_encoding["tts_text_ids"] = tts_text_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
                batch_encoding["tts_lm_attention_mask"] = tts_lm_attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list

        if has_speech:
            speech_dict = self.prepare_speech_inputs(
                all_speech_inputs, return_tensors=return_tensors,
            )
            batch_encoding["speech_tensors"] = speech_dict["padded_speeches"]
            batch_encoding["speech_masks"] = speech_dict["speech_masks"]
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None

        return batch_encoding

    def prepare_speech_inputs(self, speech_inputs, return_tensors=None,
                              device=None, dtype=None):
        """Prepare speech inputs for model consumption."""
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}

        vae_tok_seqlens = [
            math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs
        ]
        max_speech_length = max(s.shape[0] for s in speech_inputs)

        if speech_inputs[0].ndim == 1:
            padded_speeches = np.full(
                (len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32
            )
        else:
            padded_speeches = np.full(
                (len(speech_inputs), max_speech_length, speech_inputs[0].shape[-1]),
                fill_value=0, dtype=np.float32
            )
        speech_masks = np.zeros(
            (len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_
        )

        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, :len(speech)] = speech
            speech_masks[i, :vae_tok_length] = True

        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }

        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(
                padded_speeches, device=device, dtype=dtype or torch.float32
            )
            result["speech_masks"] = torch.tensor(
                speech_masks, device=device, dtype=torch.bool
            )

        return result

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(
            tokenizer_input_names + audio_processor_input_names
            + ["speech_inputs", "speech_input_mask"]
        ))

    def save_audio(self, audio, output_path="output.wav", sampling_rate=None,
                   normalize=False, batch_prefix="audio_"):
        """Save audio data to a file."""
        return self.audio_processor.save_audio(
            audio, output_path=output_path, sampling_rate=sampling_rate,
            normalize=normalize, batch_prefix=batch_prefix
        )


__all__ = [
    "VibeVoiceStreamingProcessor",
]
