"""
VibeVoice training data utilities.

Adapted from vibevoice-community/VibeVoice (MIT License).
Credits: Juan Pablo Gallego (jpgallegoar) from VoicePowered AI.

Provides dataset wrapper and data collator for VibeVoice LoRA fine-tuning.
"""

import math
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch

try:
    import librosa
except Exception:
    librosa = None

try:
    import resampy
except Exception:
    resampy = None


def _resample_if_needed(wav, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    if resampy is not None:
        return resampy.resample(wav.astype(np.float32), orig_sr, target_sr)
    if librosa is not None:
        return librosa.resample(y=wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    warnings.warn(
        "No resampler available; treating audio as target_sr without resampling. "
        "Install resampy or librosa.",
        RuntimeWarning,
    )
    return wav.astype(np.float32, copy=False)


def _apply_silence_with_crossfade(
    wav,
    sample_rate=24000,
    pre_silence_sec=0.25,
    pre_crossfade_sec=0.25,
    post_crossfade_sec=0.25,
    post_silence_sec=0.75,
):
    """Pad audio with leading/trailing silence and apply crossfades."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    start_sil_samples = int(round(pre_silence_sec * sample_rate))
    end_sil_samples = int(round(post_silence_sec * sample_rate))
    pre_crossfade_samples = int(round(pre_crossfade_sec * sample_rate))
    post_crossfade_samples = int(round(post_crossfade_sec * sample_rate))

    total_len = wav.shape[0]

    if total_len == 0:
        pieces = []
        if start_sil_samples > 0:
            pieces.append(np.zeros(start_sil_samples, dtype=np.float32))
        if end_sil_samples > 0:
            pieces.append(np.zeros(end_sil_samples, dtype=np.float32))
        return np.concatenate(pieces) if pieces else wav

    start_len = min(pre_crossfade_samples, total_len)
    remaining_after_start = max(total_len - start_len, 0)
    end_len = min(post_crossfade_samples, remaining_after_start)
    middle_end_idx = total_len - end_len

    start_segment = wav[:start_len]
    middle_segment = wav[start_len:middle_end_idx]
    end_segment = wav[middle_end_idx:]

    def _linear_fade(num_samples, start, end):
        if num_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        return np.linspace(start, end, num_samples, endpoint=True, dtype=np.float32)

    start_crossfade = start_segment * _linear_fade(start_len, 0.0, 1.0)
    end_crossfade = end_segment * _linear_fade(end_segment.shape[0], 1.0, 0.0)

    pieces = []
    if start_sil_samples > 0:
        pieces.append(np.zeros(start_sil_samples, dtype=np.float32))
    if start_crossfade.size > 0:
        pieces.append(start_crossfade.astype(np.float32, copy=False))
    if middle_segment.size > 0:
        pieces.append(middle_segment.astype(np.float32, copy=False))
    if end_crossfade.size > 0:
        pieces.append(end_crossfade.astype(np.float32, copy=False))
    if end_sil_samples > 0:
        pieces.append(np.zeros(end_sil_samples, dtype=np.float32))

    return np.concatenate(pieces)


def _load_audio_to_24k(audio, target_sr=24000, augment_with_silence=False):
    """Load and resample audio to 24kHz."""
    if isinstance(audio, np.ndarray):
        wav_out = audio.astype(np.float32)
    elif isinstance(audio, torch.Tensor):
        wav_out = audio.detach().cpu().float().numpy()
    elif isinstance(audio, str):
        if librosa is None:
            raise RuntimeError(
                "librosa is required to load audio file paths. "
                "Please pip install librosa."
            )
        wav, sr = librosa.load(audio, sr=None, mono=True)
        wav_out = _resample_if_needed(wav, int(sr), target_sr)
    elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        wav_out = _resample_if_needed(arr, sr, target_sr)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    wav_out = np.asarray(wav_out, dtype=np.float32)

    if augment_with_silence:
        wav_out = _apply_silence_with_crossfade(wav_out, sample_rate=target_sr)

    return wav_out


class VibeVoiceDataset:
    """Lightweight HF-style dataset wrapper for VibeVoice training."""

    def __init__(self, dataset, text_column="text", audio_column="audio",
                 voice_prompts_column="voice_prompts"):
        self.dataset = dataset
        self.text_column = text_column
        self.audio_column = audio_column
        self.voice_prompts_column = voice_prompts_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = {}
        data["text"] = item[self.text_column]
        data["audio"] = item[self.audio_column]

        user_provided_prompt = None
        if self.voice_prompts_column and self.voice_prompts_column in item:
            user_provided_prompt = item[self.voice_prompts_column]

        if user_provided_prompt:
            if not isinstance(user_provided_prompt, list):
                data["voice_prompts"] = [user_provided_prompt]
            else:
                data["voice_prompts"] = user_provided_prompt
        else:
            # Auto-generate voice prompt from target audio
            try:
                target_sr = 24000
                wav_array = _load_audio_to_24k(item[self.audio_column], target_sr=target_sr)
                audio_len_seconds = len(wav_array) / target_sr

                min_len_sec = min(5.0, audio_len_seconds / 4.0)
                max_len_sec = min(15.0, audio_len_seconds / 2.0)

                if min_len_sec > max_len_sec:
                    min_len_sec = max_len_sec
                max_len_sec = min(max_len_sec, audio_len_seconds)

                if max_len_sec > 0.1:
                    prompt_len_sec = random.uniform(min_len_sec, max_len_sec)
                    prompt_len_samples = int(prompt_len_sec * target_sr)
                    max_start_sample = len(wav_array) - prompt_len_samples
                    start_sample = random.randint(0, max_start_sample)
                    prompt_crop = wav_array[start_sample:start_sample + prompt_len_samples]
                    data["voice_prompts"] = [prompt_crop]
                else:
                    data["voice_prompts"] = None
            except Exception as e:
                warnings.warn(f"Could not create voice prompt for item {idx}: {e}")
                data["voice_prompts"] = None

        return data


@dataclass
class VibeVoiceCollator:
    """Data collator for VibeVoice training.

    Handles audio tokenization, sequence construction with speech placeholders,
    and batching with proper padding and masks.
    """
    processor: object
    max_length: object = None
    speech_compress_ratio: int = 3200
    semantic_vae_dim: int = 128
    compute_semantics: bool = False
    debug_checks: bool = False
    text_field: str = "text"
    audio_field: str = "audio"
    voice_prompts_field: str = "voice_prompts"
    voice_prompt_drop_rate: float = 0.0

    def __call__(self, features):
        sample_input_ids = []
        sample_attention_masks = []
        sample_acoustic_input_masks = []
        sample_acoustic_loss_masks = []

        all_speech_waveforms = []
        all_speech_latent_lengths = []
        per_segment_is_target = []

        for ex in features:
            text = ex.get(self.text_field, "")
            voice_prompts = ex.get(self.voice_prompts_field)
            target_audio = ex.get(self.audio_field)

            # Wrap plain text in Speaker format if not already formatted
            if not text.strip().lower().startswith("speaker"):
                text = f"Speaker 0: {text.strip()}"

            # Clamp drop rate
            _drop_rate = self.voice_prompt_drop_rate
            if _drop_rate < 0.0:
                _drop_rate = 0.0
            elif _drop_rate > 1.0:
                _drop_rate = 1.0

            proc = self.processor(
                text=[text],
                voice_samples=[voice_prompts] if voice_prompts is not None and random.random() >= _drop_rate else None,
                padding=False,
                truncation=False,
                max_length=self.max_length,
                return_tensors="pt",
            )

            ids = proc["input_ids"][0].tolist()
            attn = proc.get("attention_mask", torch.ones_like(proc["input_ids"]))[0].tolist()
            speech_input_mask = proc.get("speech_input_mask")
            if speech_input_mask is None:
                speech_input_mask = torch.zeros_like(proc["input_ids"], dtype=torch.bool)
            speech_input_mask_list = speech_input_mask[0].tolist()

            wav_target = _load_audio_to_24k(target_audio, target_sr=24000, augment_with_silence=True)

            # Get target latent length from acoustic tokenizer
            target_latent_len = None
            try:
                acoustic_tok = getattr(self.processor, "acoustic_tokenizer", None)
                if acoustic_tok is not None and hasattr(acoustic_tok, "encode"):
                    enc_out = acoustic_tok.encode(wav_target)
                    T = None
                    try:
                        if hasattr(enc_out, "shape") and len(getattr(enc_out, "shape", [])) >= 1:
                            T = int(enc_out.shape[0])
                        else:
                            cand = enc_out
                            for _ in range(2):
                                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                                    cand = cand[0]
                            if hasattr(cand, "shape") and len(getattr(cand, "shape", [])) >= 1:
                                T = int(cand.shape[0])
                    except Exception:
                        T = None
                    if T is not None and T > 0:
                        target_latent_len = T
            except Exception:
                target_latent_len = None

            if target_latent_len is None:
                target_latent_len = max(1, int(math.ceil(len(wav_target) / float(self.speech_compress_ratio))))

            speech_diff_id = self.processor.tokenizer.speech_diffusion_id
            target_placeholders = [speech_diff_id] * target_latent_len

            ids_extended = ids + target_placeholders
            attn_extended = attn + [1] * target_latent_len
            acoustic_input_mask = speech_input_mask_list + [True] * target_latent_len
            acoustic_loss_mask = ([False] * len(speech_input_mask_list)) + [True] * target_latent_len

            speech_end_id = self.processor.tokenizer.speech_end_id
            ids_extended.append(speech_end_id)
            attn_extended.append(1)
            acoustic_input_mask.append(False)
            acoustic_loss_mask.append(False)

            # Add EOS token
            eos_token_id = getattr(self.processor.tokenizer, "eos_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and eos_token_id >= 0:
                ids_extended.append(eos_token_id)
                attn_extended.append(1)
                acoustic_input_mask.append(False)
                acoustic_loss_mask.append(False)

            # Truncate if needed
            if self.max_length is not None and len(ids_extended) > self.max_length:
                cut = len(ids_extended) - int(self.max_length)
                leading_non_acoustic = 0
                for v in acoustic_input_mask:
                    if v:
                        break
                    leading_non_acoustic += 1
                if cut > leading_non_acoustic:
                    raise ValueError(
                        f"--max_length={self.max_length} would truncate into acoustic tokens. "
                        f"Needed cut={cut}, but only {leading_non_acoustic} leading non-acoustic tokens available. "
                        "Increase max_length or shorten text/voice-prompt preamble."
                    )
                ids_extended = ids_extended[cut:]
                attn_extended = attn_extended[cut:]
                acoustic_input_mask = acoustic_input_mask[cut:]
                acoustic_loss_mask = acoustic_loss_mask[cut:]

            sample_input_ids.append(ids_extended)
            sample_attention_masks.append(attn_extended)
            sample_acoustic_input_masks.append(acoustic_input_mask)
            sample_acoustic_loss_masks.append(acoustic_loss_mask)

            # Collect voice prompt speech segments
            voice_speeches = []
            voice_latent_lengths = []
            if proc.get("speech_tensors") is not None:
                voice_np = proc["speech_tensors"].cpu().numpy()
                voice_masks = proc["speech_masks"].cpu().numpy().astype(bool)
                for seg_idx in range(voice_np.shape[0]):
                    voice_speeches.append(voice_np[seg_idx])
                    voice_latent_lengths.append(int(voice_masks[seg_idx].sum()))

            all_speech_waveforms.extend(voice_speeches)
            all_speech_latent_lengths.extend(voice_latent_lengths)
            per_segment_is_target.extend([False] * len(voice_speeches))

            all_speech_waveforms.append(wav_target)
            all_speech_latent_lengths.append(target_latent_len)
            per_segment_is_target.append(True)

        # Pad sequences to max length
        max_seq_len = max(len(x) for x in sample_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_acoustic_input_masks = []
        padded_acoustic_loss_masks = []

        tok = self.processor.tokenizer
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = getattr(tok, "eos_token_id", None)
            if pad_token_id is None or pad_token_id < 0:
                raise ValueError(
                    "Tokenizer has no pad_token_id or eos_token_id; "
                    "please set one or pass a valid pad id."
                )

        for ids, attn, ain_mask, aloss_mask in zip(
            sample_input_ids, sample_attention_masks,
            sample_acoustic_input_masks, sample_acoustic_loss_masks
        ):
            pad_len = max_seq_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_masks.append(attn + [0] * pad_len)
            padded_acoustic_input_masks.append(ain_mask + [False] * pad_len)
            padded_acoustic_loss_masks.append(aloss_mask + [False] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
        acoustic_input_mask_tensor = torch.tensor(padded_acoustic_input_masks, dtype=torch.bool)
        acoustic_loss_mask_tensor = torch.tensor(padded_acoustic_loss_masks, dtype=torch.bool)

        if all_speech_waveforms:
            max_wave_len = max(w.shape[0] for w in all_speech_waveforms)
            padded_speeches = np.zeros((len(all_speech_waveforms), max_wave_len), dtype=np.float32)
            for i, w in enumerate(all_speech_waveforms):
                L = w.shape[0]
                padded_speeches[i, :L] = w

            max_latent_len = max(all_speech_latent_lengths) if all_speech_latent_lengths else 1
            speech_masks_np = np.zeros((len(all_speech_waveforms), max_latent_len), dtype=np.bool_)
            for i, L_lat in enumerate(all_speech_latent_lengths):
                speech_masks_np[i, :L_lat] = True

            speech_tensors_tensor = torch.tensor(padded_speeches, dtype=torch.float32)
            speech_masks_tensor = torch.tensor(speech_masks_np, dtype=torch.bool)

            speeches_loss_input_np = np.zeros_like(speech_masks_np, dtype=np.bool_)
            for i, is_target in enumerate(per_segment_is_target):
                if is_target:
                    speeches_loss_input_np[i] = speech_masks_np[i]
            speeches_loss_input_tensor = torch.tensor(speeches_loss_input_np, dtype=torch.bool)

            # Semantic features
            if (self.compute_semantics and hasattr(self.processor, "semantic_tokenizer")
                    and self.processor.semantic_tokenizer is not None):
                sem_feats = []
                for w in all_speech_waveforms:
                    try:
                        sem = self.processor.semantic_tokenizer.encode(w)
                        sem = np.asarray(sem, dtype=np.float32)
                    except Exception:
                        sem = np.zeros((0, self.semantic_vae_dim), dtype=np.float32)
                    if sem.ndim != 2:
                        raise RuntimeError(
                            f"Semantic tokenizer returned unexpected shape {sem.shape}. "
                            "Expect [T, D]."
                        )
                    L = sem.shape[0]
                    D = sem.shape[1]
                    if D != self.semantic_vae_dim:
                        if D < self.semantic_vae_dim:
                            pad_d = np.zeros((L, self.semantic_vae_dim - D), dtype=np.float32)
                            sem = np.concatenate([sem, pad_d], axis=1)
                        else:
                            sem = sem[:, :self.semantic_vae_dim]
                    if L < max_latent_len:
                        pad = np.zeros((max_latent_len - L, self.semantic_vae_dim), dtype=np.float32)
                        sem = np.concatenate([sem, pad], axis=0)
                    elif L > max_latent_len:
                        sem = sem[:max_latent_len]
                    sem_feats.append(sem.astype(np.float32))
                speech_semantic_tensors = torch.tensor(np.stack(sem_feats, axis=0), dtype=torch.float32)
            else:
                raise RuntimeError(
                    "Semantic features are required but could not be computed. "
                    "Ensure processor.semantic_tokenizer is available."
                )
        else:
            speech_tensors_tensor = None
            speech_masks_tensor = None
            speeches_loss_input_tensor = None
            speech_semantic_tensors = None

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "speech_tensors": speech_tensors_tensor,
            "speech_masks": speech_masks_tensor,
            "speech_semantic_tensors": speech_semantic_tensors,
            "acoustic_input_mask": acoustic_input_mask_tensor,
            "acoustic_loss_mask": acoustic_loss_mask_tensor,
            "speeches_loss_input": speeches_loss_input_tensor,
        }
