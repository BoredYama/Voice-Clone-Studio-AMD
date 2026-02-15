"""
TTS Model Manager

Centralized management for all TTS models (Qwen3, VibeVoice, etc.)
"""

import torch
import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import gc

from .model_utils import (
    get_device, get_dtype, get_attention_implementation,
    check_model_available_locally, empty_device_cache, log_gpu_memory, set_seed,
    run_pre_load_hooks
)

# Suppress noisy info/warning messages from upstream libraries:
# - qwen_tts config init: "speaker_encoder_config is None...", "talker_config is None...", etc.
# - transformers generation: "Setting pad_token_id to eos_token_id..."
# - transformers modeling: "Flash Attention 2 without specifying a torch dtype..."
# - transformers tensor_parallel: "TP rules were not applied...", "layers were not sharded..."
for _logger_name in [
    "qwen_tts",
    "transformers.modeling_utils",
    "transformers.generation.utils",
    "transformers.integrations.tensor_parallel",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


class TTSManager:
    """Manages all TTS models with lazy loading and VRAM optimization."""

    def __init__(self, user_config: Dict = None, samples_dir: Path = None):
        """
        Initialize TTS Manager.

        Args:
            user_config: User configuration dict (with attention_mechanism, low_cpu_mem_usage, offline_mode)
            samples_dir: Path to samples directory for prompt caching
        """
        self.user_config = user_config or {}
        self.samples_dir = samples_dir or Path("samples")

        # Model cache
        self._qwen3_base_model = None
        self._qwen3_base_size = None
        self._qwen3_voice_design_model = None
        self._qwen3_custom_voice_model = None
        self._qwen3_custom_voice_size = None
        self._vibevoice_tts_model = None
        self._vibevoice_tts_size = None
        self._luxtts_model = None

        # Chatterbox models
        self._chatterbox_tts_model = None
        self._chatterbox_vc_model = None
        self._chatterbox_mtl_model = None

        # Prompt cache
        self._voice_prompt_cache = {}
        self._luxtts_prompt_cache = {}
        self._last_loaded_model = None

    def _check_and_unload_if_different(self, model_id):
        """If switching to a different model, unload all. Stops external servers on first/new load."""
        if self._last_loaded_model is not None and self._last_loaded_model != model_id:
            print(f"Switching from {self._last_loaded_model} to {model_id} - unloading all TTS models...")
            self.unload_all()
            run_pre_load_hooks()
        elif self._last_loaded_model is None:
            # First model load — stop external servers (e.g., llama.cpp) to free VRAM
            run_pre_load_hooks()
        self._last_loaded_model = model_id

    def _load_model_with_attention(self, model_class, model_name: str, **kwargs):
        """
        Load a HuggingFace model with best available attention mechanism.

        Returns:
            Tuple: (loaded_model, attention_mechanism_used)
        """
        offline_mode = self.user_config.get("offline_mode", False)

        # Check local availability
        local_path = check_model_available_locally(model_name)
        if local_path:
            print(f"Found local model: {local_path}")
            model_to_load = str(local_path)
        elif offline_mode:
            raise RuntimeError(
                f"❌ Offline mode enabled but model not available locally: {model_name}\n"
                f"To use offline mode, download the model first or disable offline mode in Settings."
            )
        else:
            model_to_load = model_name

        mechanisms = get_attention_implementation(
            self.user_config.get("attention_mechanism", "auto")
        )

        last_error = None
        for attn in mechanisms:
            try:
                model = model_class.from_pretrained(
                    model_to_load,
                    attn_implementation=attn,
                    trust_remote_code=True,
                    **kwargs
                )
                print(f"[OK] Model loaded with {attn}")
                return model, attn
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                is_attn_error = any(
                    keyword in error_msg
                    for keyword in ["flash", "attention", "sdpa", "not supported"]
                )

                if is_attn_error:
                    print(f"  {attn} not available, trying next option...")
                    continue
                else:
                    raise e

        raise RuntimeError(f"Failed to load model: {str(last_error)}")

    def get_qwen3_base(self, size: str = "1.7B"):
        """Load Qwen3 Base TTS model."""
        model_id = f"qwen3_base_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_base_model is None:
            from qwen_tts import Qwen3TTSModel

            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
            print(f"Loading {model_name}...")

            self._qwen3_base_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map=get_device(),
                dtype=get_dtype(),
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            self._qwen3_base_size = size
            print(f"Qwen3 Base TTS ({size}) loaded!")

        return self._qwen3_base_model

    def get_qwen3_voice_design(self):
        """Load Qwen3 VoiceDesign model (1.7B only)."""
        self._check_and_unload_if_different("qwen3_voice_design")

        if self._qwen3_voice_design_model is None:
            from qwen_tts import Qwen3TTSModel

            print("Loading Qwen3 VoiceDesign model (1.7B)...")

            self._qwen3_voice_design_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=get_device(),
                dtype=get_dtype(),
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            print("VoiceDesign model loaded!")

        return self._qwen3_voice_design_model

    def get_qwen3_custom_voice(self, size: str = "1.7B"):
        """Load Qwen3 CustomVoice model."""
        model_id = f"qwen3_custom_voice_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_custom_voice_model is None:
            from qwen_tts import Qwen3TTSModel

            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
            print(f"Loading {model_name}...")

            self._qwen3_custom_voice_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map=get_device(),
                dtype=get_dtype(),
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            self._qwen3_custom_voice_size = size
            print(f"CustomVoice model ({size}) loaded!")

        return self._qwen3_custom_voice_model

    def get_vibevoice_tts(self, size: str = "1.5B"):
        """Load VibeVoice TTS model."""
        model_id = f"vibevoice_tts_{size}"
        self._check_and_unload_if_different(model_id)

        if self._vibevoice_tts_model is None:
            print(f"Loading VibeVoice TTS ({size})...")
            try:
                from modules.vibevoice_tts.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference
                )
                import warnings

                # Map size to model path
                if size == "Large (4-bit)":
                    model_path = "FranckyB/VibeVoice-Large-4bit"
                    try:
                        import bitsandbytes
                    except ImportError:
                        raise ImportError(
                            "bitsandbytes required for 4-bit models. Install with: pip install bitsandbytes"
                        )
                else:
                    model_path = f"FranckyB/VibeVoice-{size}"

                import logging
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    self._vibevoice_tts_model, _ = self._load_model_with_attention(
                        VibeVoiceForConditionalGenerationInference,
                        model_path,
                        dtype=get_dtype(),
                        device_map=get_device(),
                        low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                    )

                self._vibevoice_tts_size = size
                print(f"VibeVoice TTS ({size}) loaded!")

            except ImportError as e:
                print(f"❌ VibeVoice TTS not available: {e}")
                raise
            except Exception as e:
                print(f"❌ Error loading VibeVoice TTS: {e}")
                raise

        return self._vibevoice_tts_model

    def get_luxtts(self):
        """Load LuxTTS model (lazy import to avoid slowing app startup)."""
        self._check_and_unload_if_different("luxtts")

        if self._luxtts_model is None:
            print("Loading LuxTTS model...")
            try:
                import warnings
                import logging

                # Suppress k2 import warning — PyTorch fallback works fine
                k2_logger = logging.getLogger()
                prev_level = k2_logger.level
                k2_logger.setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*k2.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
                    from zipvoice.luxvoice import LuxTTS

                    device = get_device()
                    if device.startswith("cuda"):
                        self._luxtts_model = LuxTTS("YatharthS/LuxTTS", device="cuda")
                    elif device == "mps":
                        self._luxtts_model = LuxTTS("YatharthS/LuxTTS", device="mps")
                    else:
                        threads = int(self.user_config.get("luxtts_cpu_threads", 2))
                        self._luxtts_model = LuxTTS(
                            "YatharthS/LuxTTS", device="cpu", threads=max(1, threads)
                        )

                k2_logger.setLevel(prev_level)

                print("LuxTTS loaded!")

            except ImportError as e:
                raise ImportError(
                    f"LuxTTS not available: {e}\n"
                    "Install with: pip install zipvoice@git+https://github.com/ysharma3501/LuxTTS.git"
                )
            except Exception as e:
                print(f"Error loading LuxTTS: {e}")
                raise

        return self._luxtts_model

    def unload_all(self):
        """Unload all TTS models to free VRAM."""
        freed = []

        if self._qwen3_base_model is not None:
            del self._qwen3_base_model
            self._qwen3_base_model = None
            freed.append("Qwen3 Base")

        if self._qwen3_voice_design_model is not None:
            del self._qwen3_voice_design_model
            self._qwen3_voice_design_model = None
            freed.append("Qwen3 VoiceDesign")

        if self._qwen3_custom_voice_model is not None:
            del self._qwen3_custom_voice_model
            self._qwen3_custom_voice_model = None
            freed.append("Qwen3 CustomVoice")

        if self._vibevoice_tts_model is not None:
            del self._vibevoice_tts_model
            self._vibevoice_tts_model = None
            freed.append("VibeVoice TTS")

        if self._luxtts_model is not None:
            del self._luxtts_model
            self._luxtts_model = None
            freed.append("LuxTTS")

        if self._chatterbox_tts_model is not None:
            del self._chatterbox_tts_model
            self._chatterbox_tts_model = None
            freed.append("Chatterbox TTS")

        if self._chatterbox_vc_model is not None:
            del self._chatterbox_vc_model
            self._chatterbox_vc_model = None
            freed.append("Chatterbox VC")

        if self._chatterbox_mtl_model is not None:
            del self._chatterbox_mtl_model
            self._chatterbox_mtl_model = None
            freed.append("Chatterbox Multilingual")

        if freed:
            gc.collect()
            empty_device_cache()
            print(f"Unloaded TTS models: {', '.join(freed)}")

        return bool(freed)

    # ============================================================
    # GENERATION METHODS
    # ============================================================

    def generate_voice_design(self, text: str, language: str, instruct: str, seed: int = -1,
                              do_sample: bool = True, temperature: float = 0.9, top_k: int = 50,
                              top_p: float = 1.0, repetition_penalty: float = 1.05,
                              max_new_tokens: int = 2048) -> Tuple[str, int]:
        """
        Generate audio using voice design with natural language instructions.

        Args:
            text: Text to generate
            language: Language for TTS
            instruct: Natural language voice design instructions
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        model = self.get_qwen3_voice_design()

        # Generate
        wavs, sr = model.generate_voice_design(
            text=text.strip(),
            language=language if language != "Auto" else "Auto",
            instruct=instruct.strip(),
            do_sample=do_sample,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens)
        )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_custom_voice(self, text: str, language: str, speaker: str, instruct: str = None,
                              model_size: str = "1.7B", seed: int = -1,
                              do_sample: bool = True, temperature: float = 0.9, top_k: int = 50,
                              top_p: float = 1.0, repetition_penalty: float = 1.05,
                              max_new_tokens: int = 2048) -> Tuple[str, int]:
        """
        Generate audio using CustomVoice model with premium speakers.

        Args:
            text: Text to generate
            language: Language for TTS
            speaker: Speaker name
            instruct: Optional style instructions
            model_size: Model size (1.7B, 0.6B, etc.)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        model = self.get_qwen3_custom_voice(model_size)

        # Build kwargs
        kwargs = {
            "text": text.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens)
        }
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct.strip()

        wavs, sr = model.generate_custom_voice(**kwargs)

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_with_trained_model(self, text, language, speaker_name,
                                    checkpoint_path, instruct=None, seed=-1,
                                    do_sample=True, temperature=0.9, top_k=50,
                                    top_p=1.0, repetition_penalty=1.05,
                                    max_new_tokens=2048, user_config=None,
                                    icl_mode=False, voice_sample_path=None, ref_text=None):
        """
        Generate audio using a trained custom voice model checkpoint.

        Supports two modes:
        - Speaker embedding mode (default): Uses the baked-in speaker embedding
        - ICL mode: Uses In-Context Learning with a reference audio sample for
          much more natural and expressive voice cloning

        Args:
            text: Text to generate
            language: Language for TTS
            speaker_name: Speaker name the model was trained with
            checkpoint_path: Path to trained model checkpoint
            instruct: Optional style instructions (only used in speaker embedding mode)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            user_config: User configuration dict
            icl_mode: If True, use In-Context Learning with voice sample
            voice_sample_path: Path to reference audio (required for ICL mode)
            ref_text: Transcript of the reference audio (required for ICL mode)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        if user_config is None:
            user_config = {}

        # Determine device and dtype
        device = get_device()
        dtype = get_dtype(device)

        # Load the trained model checkpoint with attention fallback
        mechanisms = get_attention_implementation(
            user_config.get("attention_mechanism", "auto")
        )

        model = None
        for attn in mechanisms:
            try:
                model = Qwen3TTSModel.from_pretrained(
                    checkpoint_path,
                    device_map=device,
                    torch_dtype=dtype,
                    attn_implementation=attn,
                    low_cpu_mem_usage=user_config.get("low_cpu_mem_usage", False)
                )
                print(f"Trained model loaded with {attn}")
                break
            except Exception as e:
                error_msg = str(e).lower()
                is_attn_error = any(
                    kw in error_msg for kw in ["flash", "attention", "sdpa", "not supported"]
                )
                if is_attn_error:
                    print(f"  {attn} not available for trained model, trying next...")
                    continue
                raise

        if model is None:
            raise RuntimeError("Failed to load trained model with any attention mechanism")

        if icl_mode and voice_sample_path and ref_text:
            # ICL mode: use generate_voice_clone with reference audio
            # Ensure model type allows voice cloning (patch legacy "custom_voice" models)
            if hasattr(model, 'model') and hasattr(model.model, 'tts_model_type'):
                if model.model.tts_model_type != "base":
                    print(f"Patching tts_model_type from '{model.model.tts_model_type}' to 'base' for ICL inference")
                    model.model.tts_model_type = "base"

            # Training drops speaker_encoder weights from checkpoints to save space.
            # ICL needs it to compute speaker embeddings from reference audio.
            # Borrow speaker_encoder from the base model of matching size.
            if model.model.speaker_encoder is None:
                import json as json_mod
                config_path = Path(checkpoint_path) / "config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    ckpt_config = json_mod.load(f)
                base_size = "1.7B" if ckpt_config.get("tts_model_size") == "1b7" else "0.6B"
                base_name = f"Qwen/Qwen3-TTS-12Hz-{base_size}-Base"
                print(f"Speaker encoder missing - borrowing from {base_size} base model...")

                # Try local first, then HuggingFace
                local_path = check_model_available_locally(base_name)
                base_to_load = str(local_path) if local_path else base_name

                base_model = Qwen3TTSModel.from_pretrained(
                    base_to_load,
                    device_map=device,
                    torch_dtype=dtype,
                )
                model.model.speaker_encoder = base_model.model.speaker_encoder
                model.model.speaker_encoder_sample_rate = base_model.model.speaker_encoder_sample_rate
                del base_model
                empty_device_cache()
                print("Speaker encoder transplanted successfully")

            # Create voice clone prompt with ICL (not x-vector only)
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=str(voice_sample_path),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )

            # Build generation kwargs
            gen_kwargs = {
                'max_new_tokens': int(max_new_tokens),
            }
            if do_sample:
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = float(temperature)
                if top_k > 0:
                    gen_kwargs['top_k'] = int(top_k)
                if top_p < 1.0:
                    gen_kwargs['top_p'] = float(top_p)
                if repetition_penalty != 1.0:
                    gen_kwargs['repetition_penalty'] = float(repetition_penalty)

            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=language if language != "Auto" else "Auto",
                voice_clone_prompt=prompt_items,
                **gen_kwargs
            )
        else:
            # Speaker embedding mode: use generate_custom_voice
            # Ensure model type allows custom voice (patch "base" models)
            if hasattr(model, 'model') and hasattr(model.model, 'tts_model_type'):
                if model.model.tts_model_type != "custom_voice":
                    print(f"Patching tts_model_type from '{model.model.tts_model_type}' to 'custom_voice' for speaker embedding inference")
                    model.model.tts_model_type = "custom_voice"

            kwargs = {
                "text": text.strip(),
                "language": language if language != "Auto" else "Auto",
                "speaker": speaker_name,
                "do_sample": do_sample,
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "max_new_tokens": int(max_new_tokens)
            }
            if instruct and instruct.strip():
                kwargs["instruct"] = instruct.strip()

            wavs, sr = model.generate_custom_voice(**kwargs)

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_voice_clone_qwen(self, text: str, language: str, prompt_items, seed: int = -1,
                                  do_sample: bool = True, temperature: float = 0.9, top_k: int = 50,
                                  top_p: float = 1.0, repetition_penalty: float = 1.05,
                                  max_new_tokens: int = 2048, model_size: str = "1.7B") -> Tuple[str, int]:
        """
        Generate audio using Qwen3 voice cloning with cached prompt.

        Args:
            text: Text to generate
            language: Language for TTS
            prompt_items: Pre-computed voice clone prompt
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            model_size: Model size (1.7B or 0.6B)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load BASE model (not CustomVoice - Base supports voice cloning)
        model = self.get_qwen3_base(model_size)

        # Prepare generation kwargs
        gen_kwargs = {
            'max_new_tokens': int(max_new_tokens),
        }
        if do_sample:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = float(temperature)
            if top_k > 0:
                gen_kwargs['top_k'] = int(top_k)
            if top_p < 1.0:
                gen_kwargs['top_p'] = float(top_p)
            if repetition_penalty != 1.0:
                gen_kwargs['repetition_penalty'] = float(repetition_penalty)

        # Generate using the cached prompt
        wavs, sr = model.generate_voice_clone(
            text=text.strip(),
            language=language if language != "Auto" else "Auto",
            voice_clone_prompt=prompt_items,
            **gen_kwargs
        )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    @staticmethod
    def _chunk_text_for_vibevoice(text, sentences_per_chunk):
        """Split text into multi-turn Speaker 1 format for VibeVoice stability.

        Long single-speaker text causes VibeVoice to degrade (screaming/rushing).
        Per Microsoft's official guidance, splitting into repeated Speaker 1 turns
        resets the generation state and prevents quality loss.

        Args:
            text: The full text to chunk
            sentences_per_chunk: Number of sentences per chunk

        Returns:
            Formatted multi-turn script string
        """
        import re
        # Split on sentence-ending punctuation, keeping the delimiter attached
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter empty
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            return f"Speaker 1: {text}"

        # Group into chunks of N sentences
        chunks = []
        for i in range(0, len(parts), sentences_per_chunk):
            chunk = ' '.join(parts[i:i + sentences_per_chunk])
            chunks.append(f"Speaker 1: {chunk}")

        return '\n'.join(chunks)

    def generate_voice_clone_vibevoice(self, text, voice_sample_path, seed=-1,
                                       do_sample=False, temperature=1.0, top_k=50,
                                       top_p=1.0, repetition_penalty=1.0,
                                       cfg_scale=3.0, num_steps=20,
                                       sentences_per_chunk=0,
                                       model_size="Large", user_config=None):
        """
        Generate audio using VibeVoice voice cloning.

        Args:
            text: Text to generate
            voice_sample_path: Path to voice sample WAV file
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            cfg_scale: Classifier-free guidance scale
            num_steps: DDPM inference steps
            sentences_per_chunk: Split into chunks of N sentences (0 = no split).
                Prevents quality degradation on long text.
            model_size: Model size (Large, 1.5B, or Large (4-bit))
            user_config: User configuration dict

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import warnings
        import logging

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        model = self.get_vibevoice_tts(model_size)

        from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor

        # Map model_size to valid HuggingFace repo path
        if model_size == "Large (4-bit)":
            model_path = "FranckyB/VibeVoice-Large-4bit"
        else:
            model_path = f"FranckyB/VibeVoice-{model_size}"

        # Suppress tokenizer mismatch warning
        prev_level = logging.getLogger("transformers.tokenization_utils_base").level
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if user_config is None:
                user_config = {}
            offline_mode = user_config.get("offline_mode", False)
            processor = VibeVoiceProcessor.from_pretrained(model_path, local_files_only=offline_mode)

        logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

        # Build generation config
        gen_config = {'do_sample': do_sample}
        if do_sample:
            gen_config['temperature'] = float(temperature)
            if top_k > 0:
                gen_config['top_k'] = int(top_k)
            if top_p < 1.0:
                gen_config['top_p'] = float(top_p)
            if repetition_penalty != 1.0:
                gen_config['repetition_penalty'] = float(repetition_penalty)

        sr = 24000  # VibeVoice uses 24kHz
        device = get_device()

        # If chunking is enabled, generate each chunk separately and concatenate.
        # Each chunk gets its own inference call so the generation state resets,
        # preventing quality degradation (screaming/rushing) on long text.
        if sentences_per_chunk and sentences_per_chunk > 0:
            import re
            import numpy as np
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) > 1:
                chunks = []
                for i in range(0, len(parts), sentences_per_chunk):
                    chunks.append(' '.join(parts[i:i + sentences_per_chunk]))

                print(f"VibeVoice chunking: {len(chunks)} chunks of ~{sentences_per_chunk} sentence(s)")
                audio_segments = []

                for idx, chunk in enumerate(chunks):
                    chunk_script = f"Speaker 1: {chunk}"

                    chunk_inputs = processor(
                        text=[chunk_script],
                        voice_samples=[[voice_sample_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    for k, v in chunk_inputs.items():
                        if torch.is_tensor(v):
                            chunk_inputs[k] = v.to(device)

                    model.set_ddpm_inference_steps(num_steps=int(num_steps))

                    outputs = model.generate(
                        **chunk_inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config=gen_config,
                        verbose=False,
                    )

                    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                        audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                        audio_segments.append(audio_tensor.squeeze().numpy())
                        print(f"  Chunk {idx + 1}/{len(chunks)} done ({len(chunk.split())} words)")
                    else:
                        print(f"  Chunk {idx + 1}/{len(chunks)} produced no audio, skipping")

                if not audio_segments:
                    raise RuntimeError("VibeVoice failed to generate audio for any chunk")

                audio_data = np.concatenate(audio_segments)
                return audio_data, sr

        # Standard single-pass generation (no chunking)
        formatted_script = f"Speaker 1: {text.strip()}"

        # Process inputs
        inputs = processor(
            text=[formatted_script],
            voice_samples=[[voice_sample_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # Set inference steps
        model.set_ddpm_inference_steps(num_steps=int(num_steps))

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config=gen_config,
            verbose=False,
        )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Convert bfloat16 to float32 for soundfile compatibility
            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
            audio_data = audio_tensor.squeeze().numpy()
            sr = 24000  # VibeVoice uses 24kHz
        else:
            raise RuntimeError("VibeVoice failed to generate audio")

        return audio_data, sr

    # Voice prompt caching
    def get_prompt_cache_path(self, sample_name: str, model_size: str = "1.7B") -> Path:
        """Get path to cached voice prompt."""
        return self.samples_dir / f"{sample_name}_{model_size}.pt"

    def compute_sample_hash(self, wav_path: str, ref_text: str) -> str:
        """Compute hash of sample to detect changes."""
        hasher = hashlib.md5()
        with open(wav_path, 'rb') as f:
            hasher.update(f.read())
        hasher.update(ref_text.encode('utf-8'))
        return hasher.hexdigest()

    def save_voice_prompt(self, sample_name: str, prompt_items, sample_hash: str, model_size: str = "1.7B") -> bool:
        """Save voice prompt to cache."""
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        try:
            # Move tensors to CPU
            if isinstance(prompt_items, dict):
                cpu_prompt = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in prompt_items.items()
                }
            elif isinstance(prompt_items, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in prompt_items
                ]
            else:
                cpu_prompt = prompt_items.cpu() if isinstance(prompt_items, torch.Tensor) else prompt_items

            cache_data = {
                'prompt': cpu_prompt,
                'hash': sample_hash,
                'version': '1.0'
            }
            torch.save(cache_data, cache_path)
            print(f"Saved voice prompt: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to save voice prompt: {e}")
            return False

    def load_voice_prompt(self, sample_name: str, expected_hash: str, model_size: str = "1.7B") -> Optional[dict]:
        """Load voice prompt from cache if valid."""
        cache_key = f"{sample_name}_{model_size}"

        # Check memory cache first
        if cache_key in self._voice_prompt_cache:
            cached = self._voice_prompt_cache[cache_key]
            if cached['hash'] == expected_hash:
                return cached['prompt']

        # Check disk cache
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        if not cache_path.exists():
            return None

        try:
            cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

            if cache_data.get('hash') != expected_hash:
                return None

            # Move to device
            cached_prompt = cache_data['prompt']
            device = get_device()

            if isinstance(cached_prompt, dict):
                prompt_items = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cached_prompt.items()
                }
            elif isinstance(cached_prompt, (list, tuple)):
                prompt_items = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt_items = cached_prompt.to(device) if isinstance(cached_prompt, torch.Tensor) else cached_prompt

            # Store in memory cache
            self._voice_prompt_cache[cache_key] = {
                'prompt': prompt_items,
                'hash': expected_hash
            }

            return prompt_items

        except Exception as e:
            print(f"Failed to load voice prompt cache: {e}")
            return None

    # ============================================================
    # LUXTTS PROMPT CACHING
    # ============================================================

    def compute_audio_hash(self, wav_path):
        """Compute a hash of the raw audio file bytes (used for LuxTTS prompt caching)."""
        hasher = hashlib.md5()
        with open(wav_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def get_luxtts_prompt_cache_path(self, sample_name):
        """Get the path to the cached LuxTTS encoded prompt file."""
        return self.samples_dir / f"{sample_name}_luxtts.pt"

    def save_luxtts_prompt(self, sample_name, encoded_prompt, audio_hash, rms=0.01, ref_duration=30):
        """Save LuxTTS encoded prompt to disk (CPU tensors only)."""
        cache_path = self.get_luxtts_prompt_cache_path(sample_name)

        try:
            if isinstance(encoded_prompt, dict):
                cpu_prompt = {}
                for key, value in encoded_prompt.items():
                    cpu_prompt[key] = value.cpu() if isinstance(value, torch.Tensor) else value
            elif isinstance(encoded_prompt, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in encoded_prompt
                ]
            else:
                cpu_prompt = encoded_prompt.cpu() if isinstance(encoded_prompt, torch.Tensor) else encoded_prompt

            cache_data = {
                "prompt": cpu_prompt,
                "audio_hash": audio_hash,
                "params": {
                    "rms": round(float(rms), 6),
                    "ref_duration": int(ref_duration),
                },
                "version": "luxtts-1.0",
            }
            torch.save(cache_data, cache_path)
            return True
        except Exception as e:
            print(f"Failed to save LuxTTS prompt: {e}")
            return False

    def load_luxtts_prompt(self, sample_name, expected_audio_hash, rms=0.01, ref_duration=30):
        """Load LuxTTS encoded prompt from disk/memory if valid."""
        cache_key = sample_name

        # Check memory cache
        if cache_key in self._luxtts_prompt_cache:
            cached = self._luxtts_prompt_cache[cache_key]
            if cached.get("audio_hash") == expected_audio_hash:
                return cached["prompt"]

        # Check disk cache
        cache_path = self.get_luxtts_prompt_cache_path(sample_name)
        if not cache_path.exists():
            return None

        try:
            device = get_device()
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

            if cache_data.get("audio_hash") != expected_audio_hash:
                return None

            params = cache_data.get("params") or {}
            if round(float(params.get("rms", -1)), 6) != round(float(rms), 6) or int(
                params.get("ref_duration", -1)
            ) != int(ref_duration):
                return None

            cached_prompt = cache_data.get("prompt")
            if isinstance(cached_prompt, dict):
                prompt = {}
                for key, value in cached_prompt.items():
                    prompt[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            elif isinstance(cached_prompt, (list, tuple)):
                prompt = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt = cached_prompt.to(device) if isinstance(cached_prompt, torch.Tensor) else cached_prompt

            self._luxtts_prompt_cache[cache_key] = {
                "prompt": prompt,
                "audio_hash": expected_audio_hash,
            }

            return prompt

        except Exception as e:
            print(f"Failed to load LuxTTS prompt cache: {e}")
            return None

    def _encode_luxtts_prompt_direct(self, wav_path, ref_text, rms=0.01, ref_duration=30):
        """Encode LuxTTS prompt directly using known transcript text (bypasses Whisper).

        Replicates zipvoice's process_audio() but substitutes the known transcript
        instead of running Whisper transcription.
        """
        import librosa
        from zipvoice.utils.infer import rms_norm

        lux_model = self.get_luxtts()

        # Load audio at 24kHz (same as process_audio)
        prompt_wav, sr = librosa.load(str(wav_path), sr=24000, duration=int(ref_duration))
        prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
        prompt_wav, prompt_rms = rms_norm(prompt_wav, float(rms))

        # Extract features
        prompt_features = lux_model.feature_extractor.extract(
            prompt_wav, sampling_rate=24000
        ).to(lux_model.device)
        prompt_features = prompt_features.unsqueeze(0) * 0.1  # feat_scale=0.1

        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=lux_model.device)

        # Tokenize the known transcript directly (no Whisper needed)
        prompt_tokens = lux_model.tokenizer.texts_to_token_ids([ref_text])

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_features_lens": prompt_features_lens,
            "prompt_features": prompt_features,
            "prompt_rms": prompt_rms,
        }

    def get_or_create_luxtts_prompt(self, sample_name, wav_path, rms=0.01, ref_duration=30, ref_text=None, progress_callback=None):
        """Get cached LuxTTS encoded prompt or create a new one."""
        audio_hash = self.compute_audio_hash(wav_path)

        cached = self.load_luxtts_prompt(
            sample_name,
            expected_audio_hash=audio_hash,
            rms=rms,
            ref_duration=ref_duration,
        )
        if cached is not None:
            if progress_callback:
                progress_callback(0.35, desc="Using cached LuxTTS voice prompt...")
            return cached, True

        if progress_callback:
            progress_callback(0.2, desc="Encoding LuxTTS voice prompt (first time)...")

        # Use direct encoding with known text (bypasses Whisper entirely)
        if ref_text:
            encoded_prompt = self._encode_luxtts_prompt_direct(
                wav_path, ref_text, rms=rms, ref_duration=ref_duration
            )
        else:
            raise ValueError(
                f"No transcript found for sample '{sample_name}'. "
                "Please transcribe this sample first in the Prep Audio tab "
                "(using Whisper or VibeVoice ASR), then try again."
            )

        if progress_callback:
            progress_callback(0.35, desc="Caching LuxTTS voice prompt...")

        self.save_luxtts_prompt(
            sample_name, encoded_prompt, audio_hash, rms=rms, ref_duration=ref_duration
        )

        self._luxtts_prompt_cache[sample_name] = {
            "prompt": encoded_prompt,
            "audio_hash": audio_hash,
        }

        return encoded_prompt, False

    def generate_voice_clone_luxtts(
        self, text, voice_sample_path, sample_name,
        num_steps=4, t_shift=0.5, speed=1.0,
        return_smooth=False, rms=0.01, ref_duration=30,
        guidance_scale=3.0, seed=-1, ref_text=None, progress_callback=None
    ):
        """Generate audio using LuxTTS voice cloning.

        Args:
            text: Text to generate
            voice_sample_path: Path to voice sample WAV file
            sample_name: Name of the sample (for caching)
            num_steps: Sampling steps (3-4 recommended)
            t_shift: Sampling parameter (higher = better quality but more pronunciation errors)
            speed: Speed multiplier (lower=slower)
            return_smooth: Smoother output (may reduce metallic artifacts)
            rms: Loudness (0.01 recommended)
            ref_duration: How many seconds of reference audio to use (30 default, increase to 1000 if artifacts)
            guidance_scale: Classifier-free guidance scale (3.0 default)
            seed: Random seed (-1 for random)
            ref_text: Known transcript of the voice sample (bypasses Whisper if provided)
            progress_callback: Optional Gradio progress callback

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Get or create encoded prompt (with caching)
        encoded_prompt, was_cached = self.get_or_create_luxtts_prompt(
            sample_name=sample_name,
            wav_path=voice_sample_path,
            rms=rms,
            ref_duration=ref_duration,
            ref_text=ref_text,
            progress_callback=progress_callback,
        )

        cache_status = "cached" if was_cached else "newly processed"
        if progress_callback:
            progress_callback(0.6, desc=f"Generating audio ({cache_status} prompt)...")

        # Load model and generate
        lux_model = self.get_luxtts()
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
            wav_tensor = lux_model.generate_speech(
                text.strip(),
                encoded_prompt,
                num_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                t_shift=float(t_shift),
                speed=float(speed),
                return_smooth=bool(return_smooth),
            )

        # Convert to numpy
        if isinstance(wav_tensor, torch.Tensor):
            audio_data = wav_tensor.detach().cpu().to(torch.float32).numpy().squeeze()
        else:
            audio_data = np.array(wav_tensor).squeeze()

        return audio_data, 48000, was_cached

    # ============================================================
    # CHATTERBOX METHODS
    # ============================================================

    def get_chatterbox_tts(self):
        """Load Chatterbox TTS model (English)."""
        self._check_and_unload_if_different("chatterbox_tts")

        if self._chatterbox_tts_model is None:
            print("Loading Chatterbox TTS model...")
            try:
                from modules.chatterbox import ChatterboxTTS

                device = get_device()
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_tts_model = ChatterboxTTS.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_tts_model = ChatterboxTTS.from_pretrained(device)
                print("Chatterbox TTS loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox TTS: {e}")
                raise

        return self._chatterbox_tts_model

    def get_chatterbox_multilingual(self):
        """Load Chatterbox Multilingual TTS model (23 languages)."""
        self._check_and_unload_if_different("chatterbox_mtl")

        if self._chatterbox_mtl_model is None:
            print("Loading Chatterbox Multilingual TTS model...")
            try:
                from modules.chatterbox import ChatterboxMultilingualTTS

                device = get_device()
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_mtl_model = ChatterboxMultilingualTTS.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_mtl_model = ChatterboxMultilingualTTS.from_pretrained(device)
                print("Chatterbox Multilingual TTS loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox Multilingual not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox Multilingual TTS: {e}")
                raise

        return self._chatterbox_mtl_model

    def get_chatterbox_vc(self):
        """Load Chatterbox Voice Conversion model."""
        self._check_and_unload_if_different("chatterbox_vc")

        if self._chatterbox_vc_model is None:
            print("Loading Chatterbox Voice Conversion model...")
            try:
                from modules.chatterbox import ChatterboxVC

                device = get_device()
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_vc_model = ChatterboxVC.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_vc_model = ChatterboxVC.from_pretrained(device)
                print("Chatterbox VC loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox VC not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox VC: {e}")
                raise

        return self._chatterbox_vc_model

    def generate_voice_clone_chatterbox(self, text, voice_sample_path, seed=-1,
                                        exaggeration=0.5, cfg_weight=0.5,
                                        temperature=0.8, repetition_penalty=1.2,
                                        top_p=1.0):
        """Generate audio using Chatterbox TTS (English voice cloning).

        Args:
            text: Text to speak
            voice_sample_path: Path to reference voice WAV
            seed: Random seed (-1 for random)
            exaggeration: Emotion intensity (0-2)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            top_p: Top-p sampling

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        model = self.get_chatterbox_tts()

        wav_tensor = model.generate(
            text=text.strip(),
            audio_prompt_path=str(voice_sample_path),
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000

    def generate_voice_clone_chatterbox_multilingual(self, text, language_code,
                                                     voice_sample_path, seed=-1,
                                                     exaggeration=0.5, cfg_weight=0.5,
                                                     temperature=0.8, repetition_penalty=2.0,
                                                     top_p=1.0):
        """Generate audio using Chatterbox Multilingual TTS (23 languages).

        Args:
            text: Text to speak
            language_code: 2-letter ISO language code (e.g. "en", "fr")
            voice_sample_path: Path to reference voice WAV
            seed: Random seed (-1 for random)
            exaggeration: Emotion intensity (0-2)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty (default 2.0 for multilingual)
            top_p: Top-p sampling

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        model = self.get_chatterbox_multilingual()

        wav_tensor = model.generate(
            text=text.strip(),
            language_id=language_code,
            audio_prompt_path=str(voice_sample_path),
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000

    def generate_voice_convert_chatterbox(self, source_audio_path, target_voice_path):
        """Convert voice in source audio to match target voice.

        Args:
            source_audio_path: Path to source audio WAV to convert
            target_voice_path: Path to target voice reference WAV

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        model = self.get_chatterbox_vc()

        wav_tensor = model.generate(
            audio=str(source_audio_path),
            target_voice_path=str(target_voice_path),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000


# Global singleton instance
_tts_manager = None


def get_tts_manager(user_config: Dict = None, samples_dir: Path = None) -> TTSManager:
    """Get or create the global TTS manager."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager(user_config, samples_dir)
    return _tts_manager
