"""
VibeVoice LoRA fine-tuning script.

Adapted from vibevoice-community/VibeVoice (MIT License).
Credits: Juan Pablo Gallego (jpgallegoar) from VoicePowered AI.

Trains LoRA adapters on VibeVoice base model with mixed CE + diffusion loss.
Invoked via subprocess from model_utils.py.

Usage:
    python -m modules.vibevoice_tts.finetune.train_vibevoice \
        --model_name_or_path vibevoice/VibeVoice-1.5B \
        --train_jsonl path/to/train.jsonl \
        --output_dir path/to/output \
        --num_train_epochs 10 \
        --per_device_train_batch_size 1
"""

import logging
import os
import sys
import copy
from dataclasses import dataclass, field

import json as _json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers import TrainingArguments as HfTrainingArguments

from peft import LoraConfig, get_peft_model, TaskType

# Use vendored VibeVoice classes
from modules.vibevoice_tts.modular.modeling_vibevoice import (
    VibeVoiceForConditionalGeneration,
    VibeVoiceCausalLMOutputWithPast,
)
from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
from modules.vibevoice_tts.finetune.data_vibevoice import VibeVoiceDataset, VibeVoiceCollator

logger = logging.getLogger(__name__)


# ============================================================================
# EMA Callback for diffusion head
# ============================================================================

class EmaCallback(TrainerCallback):
    """Exponential Moving Average callback for the diffusion prediction head."""

    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None

    def _get_module(self, model):
        mod = model
        for name in self.attr_path.split('.'):
            mod = getattr(mod, name)
        return mod

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        head = self._get_module(model)
        self.shadow = {
            k: p.detach().to(self.device).clone()
            for k, p in head.state_dict().items()
        }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        head = self._get_module(model)
        with torch.no_grad():
            for k, v in head.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.device), alpha=(1.0 - self.decay)
                )

    def _swap_in_ema(self, model):
        head = self._get_module(model)
        self._orig = copy.deepcopy(head.state_dict())
        head.load_state_dict(self.shadow, strict=False)

    def _swap_back(self, model):
        if self._orig is None:
            return
        head = self._get_module(model)
        head.load_state_dict(self._orig, strict=False)
        self._orig = None

    def on_save(self, args, state, control, model=None, **kwargs):
        self._swap_in_ema(model)

    def on_save_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._swap_in_ema(model)


# ============================================================================
# Arguments
# ============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to VibeVoice base model"}
    )
    processor_name_or_path: str = field(
        default=None, metadata={"help": "Path to processor dir. Defaults to model path."}
    )
    freeze_acoustic_tokenizer: bool = field(default=True)
    freeze_semantic_tokenizer: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target module names"},
    )
    train_diffusion_head: bool = field(
        default=False, metadata={"help": "Train diffusion prediction head (full fine-tune)"}
    )
    train_connectors: bool = field(
        default=False, metadata={"help": "Train acoustic/semantic connectors"}
    )


@dataclass
class DataArguments:
    train_jsonl: str = field(
        default=None, metadata={"help": "Path to local train JSONL with {text, audio}"}
    )
    validation_jsonl: str = field(
        default=None, metadata={"help": "Optional path to local validation JSONL"}
    )
    text_column_name: str = field(default="text")
    audio_column_name: str = field(default="audio")
    voice_prompts_column_name: str = field(default="voice_prompts")
    max_length: int = field(default=None)
    voice_prompt_drop_rate: float = field(
        default=0.0,
        metadata={"help": "Probability to drop voice prompt during training (0.0=keep, 1.0=drop)"},
    )


@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    ddpm_batch_mul: int = field(default=1)
    ce_loss_weight: float = field(default=1.0)
    diffusion_loss_weight: float = field(default=1.0)
    save_interval: int = field(
        default=0,
        metadata={"help": "Save every N epochs (0 = no intermediate saves)"}
    )
    gradient_clipping: bool = field(
        default=False,
        metadata={"help": "Enable gradient clipping using max_grad_norm"},
    )
    ema_decay: float = field(
        default=-1.0,
        metadata={"help": "EMA decay for diffusion head. -1 = auto-calculate based on total steps. 0 = disable EMA."},
    )
    remove_unused_columns: bool = field(default=False)
    save_only_model: bool = field(default=True)
    save_strategy: str = field(default="no")
    report_to: str = field(default="none")


# ============================================================================
# Helpers
# ============================================================================

def build_lora_config(args):
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )


def mask_for_ce(labels, attention_mask, acoustic_input_mask, pad_id=-100):
    """Create CE loss mask: exclude acoustic positions, keep text tokens."""
    shifted = labels[:, 1:].contiguous()
    base_mask = (
        attention_mask[:, 1:].contiguous().eq(1)
        if (attention_mask is not None and attention_mask.numel() > 0)
        else torch.ones_like(shifted, dtype=torch.bool)
    )
    label_is_acoustic = acoustic_input_mask[:, 1:].contiguous()
    final_mask = base_mask & (~label_is_acoustic)
    out = shifted.clone()
    out[~final_mask] = pad_id
    return out


def _patch_acoustic_encode(model_obj, logger_):
    """Patch acoustic_tokenizer.encode() to return [[...]] for legacy indexing."""
    try:
        acoustic = getattr(getattr(model_obj, "model", model_obj), "acoustic_tokenizer", None)
        if acoustic is None or not hasattr(acoustic, "encode"):
            return
        base_encode = acoustic.encode

        def encode_wrapped(*args, **kwargs):
            out = base_encode(*args, **kwargs)
            try:
                _ = out[0][0]
                return out
            except Exception:
                pass
            if isinstance(out, dict):
                for k in ("frames", "codes", "tokens", "latents", "hidden_states"):
                    if k in out:
                        return [[out[k]]]
                if len(out) > 0:
                    return [[next(iter(out.values()))]]
            for attr in ("frames", "codes", "tokens", "latents", "hidden_states"):
                if hasattr(out, attr):
                    return [[getattr(out, attr)]]
            try:
                if isinstance(out, torch.Tensor):
                    return [[out]]
            except Exception:
                pass
            return [[out]]

        acoustic.encode = encode_wrapped
        logger_.info("Patched acoustic_tokenizer.encode() for legacy indexing.")
    except Exception as e:
        logger_.warning(f"Failed to patch acoustic_tokenizer.encode(): {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training parameters %s", training_args)
    set_seed(training_args.seed)

    # Gradient clipping: disable unless explicitly requested via --gradient_clipping
    if not getattr(training_args, "gradient_clipping", False):
        if hasattr(training_args, "max_grad_norm"):
            training_args.max_grad_norm = 0.0
    else:
        if (not hasattr(training_args, "max_grad_norm")) or training_args.max_grad_norm is None or training_args.max_grad_norm <= 0:
            training_args.max_grad_norm = 1.0
        logger.info(f"Gradient clipping enabled: max_grad_norm={training_args.max_grad_norm}")

    # Load processor
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError("--model_name_or_path must be provided")
    processor = VibeVoiceProcessor.from_pretrained(processor_path)

    # Validate special tokens
    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id"]:
        if not hasattr(tok, required) or getattr(tok, required) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    # Load model
    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required")

    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        dtype = torch.float16

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
    )
    _patch_acoustic_encode(model, logger)
    processor.semantic_tokenizer = getattr(model.model, "semantic_tokenizer", None)

    # Hard-tie LM head
    try:
        emb_module = model.get_input_embeddings()
        head_module = model.get_output_embeddings()
        if hasattr(emb_module, "weight") and hasattr(head_module, "weight"):
            if (emb_module.weight.shape == head_module.weight.shape
                    and emb_module.weight.data_ptr() != head_module.weight.data_ptr()):
                with torch.no_grad():
                    head_module.weight = emb_module.weight
                logger.info("Force-tied LM head weight to input embeddings.")
    except Exception as e:
        logger.warning(f"Force-tie of LM head failed: {e}")

    # Disable cache during training
    if hasattr(model.config, "use_cache") and training_args.do_train:
        model.config.use_cache = False

    # Freeze tokenizers
    if model_args.freeze_acoustic_tokenizer and hasattr(model.model, "acoustic_tokenizer"):
        for p in model.model.acoustic_tokenizer.parameters():
            p.requires_grad = False
    if model_args.freeze_semantic_tokenizer and hasattr(model.model, "semantic_tokenizer"):
        for p in model.model.semantic_tokenizer.parameters():
            p.requires_grad = False

    # LoRA wrap LLM
    lora_cfg = build_lora_config(model_args)
    tm_lower = [s.strip().lower() for s in model_args.lora_target_modules.split(",") if s.strip()]
    skip_lm_lora = (len(tm_lower) == 0) or all(
        t in ("none", "off", "disable", "disabled") for t in tm_lower
    )
    if not skip_lm_lora:
        model.model.language_model = get_peft_model(model.model.language_model, lora_cfg)
    else:
        logger.info("Skipping LLM LoRA wrapping.")

    try:
        model.tie_weights()
    except Exception:
        pass

    # Freeze all then enable trainable subsets
    for _, p in model.named_parameters():
        p.requires_grad = False

    try:
        for n, p in model.model.language_model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
    except Exception:
        logger.warning("Could not re-enable LoRA params on language_model.")

    # Train full diffusion head (optional)
    if getattr(model_args, "train_diffusion_head", False) and hasattr(model.model, "prediction_head"):
        for p in model.model.prediction_head.parameters():
            p.requires_grad = True

    # Connectors
    if getattr(model_args, "train_connectors", False):
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = True
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = True
    else:
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = False
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = False

    # Freeze embedding + head
    try:
        emb = model.get_input_embeddings()
        if hasattr(emb, "weight"):
            emb.weight.requires_grad_(False)
        head = model.get_output_embeddings()
        if head is not None and hasattr(head, "weight"):
            head.weight.requires_grad_(False)
    except Exception:
        pass

    # Diagnostics
    def _sum_params(named_iter):
        return sum(p.numel() for _, p in named_iter if p.requires_grad)

    try:
        lm_lora = _sum_params(model.model.language_model.named_parameters()) \
            if hasattr(model.model, "language_model") else 0
        pred_head_train = _sum_params(model.model.prediction_head.named_parameters()) \
            if hasattr(model.model, "prediction_head") else 0
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Trainable -> LLM-LoRA: {lm_lora:,} | diff_head: {pred_head_train:,} | "
            f"TOTAL: {total_trainable:,}"
        )
    except Exception:
        pass

    # Load datasets (plain JSONL — no HuggingFace datasets dependency)
    if data_args.train_jsonl is not None:
        def _load_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [_json.loads(line) for line in f if line.strip()]

        raw = {"train": _load_jsonl(data_args.train_jsonl)}
        if data_args.validation_jsonl is not None:
            raw["validation"] = _load_jsonl(data_args.validation_jsonl)
    else:
        raise ValueError("--train_jsonl is required for local training.")

    train_ds = raw["train"]
    eval_ds = None
    if training_args.do_eval and "validation" in raw:
        eval_ds = raw["validation"]

    train_dataset = VibeVoiceDataset(
        train_ds,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        voice_prompts_column=data_args.voice_prompts_column_name,
    )
    eval_dataset = None
    if eval_ds is not None:
        eval_dataset = VibeVoiceDataset(
            eval_ds,
            text_column=data_args.text_column_name,
            audio_column=data_args.audio_column_name,
            voice_prompts_column=data_args.voice_prompts_column_name,
        )

    # Build collator
    speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
    semantic_dim = getattr(model.config, "semantic_vae_dim", None)
    if semantic_dim is None:
        try:
            semantic_dim = int(getattr(model.config.semantic_tokenizer_config, "vae_dim", 128))
        except Exception:
            semantic_dim = 128

    compute_semantics_flag = (
        hasattr(processor, "semantic_tokenizer") and processor.semantic_tokenizer is not None
    )

    data_collator = VibeVoiceCollator(
        processor=processor,
        max_length=data_args.max_length,
        speech_compress_ratio=speech_compress_ratio,
        semantic_vae_dim=semantic_dim,
        compute_semantics=compute_semantics_flag,
        voice_prompt_drop_rate=data_args.voice_prompt_drop_rate,
    )

    # ========================================================================
    # Custom Trainer
    # ========================================================================

    class VibeVoiceTrainer(Trainer):
        """Custom trainer with mixed CE + diffusion loss."""

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            acoustic_input_mask = inputs.get("acoustic_input_mask")

            # Ensure semantic tensors have correct dtype
            sem = inputs.get("speech_semantic_tensors", None)
            try:
                target_dtype = next(model.model.semantic_connector.parameters()).dtype
            except Exception:
                target_dtype = model.get_input_embeddings().weight.dtype

            if sem is None:
                sm = inputs.get("speech_masks")
                if sm is not None:
                    zeros = torch.zeros(
                        sm.size(0), sm.size(1),
                        getattr(model.config, "semantic_vae_dim", 128),
                        dtype=target_dtype,
                        device=sm.device,
                    )
                    inputs["speech_semantic_tensors"] = zeros
            else:
                if isinstance(sem, torch.Tensor):
                    inputs["speech_semantic_tensors"] = sem.to(dtype=target_dtype)

            # Call model forward directly (handles speech features, diffusion loss, etc.)
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=attention_mask,
                speech_tensors=inputs.get("speech_tensors"),
                speech_masks=inputs.get("speech_masks"),
                speech_semantic_tensors=inputs.get("speech_semantic_tensors"),
                acoustic_input_mask=acoustic_input_mask,
                acoustic_loss_mask=inputs.get("acoustic_loss_mask"),
                speeches_loss_input=inputs.get("speeches_loss_input"),
                ddpm_batch_mul=training_args.ddpm_batch_mul,
            )

            # CE Loss
            logits = outputs.logits
            ce_labels = mask_for_ce(labels, attention_mask, acoustic_input_mask, pad_id=-100)
            shift_logits = logits[:, :-1, :].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), ce_labels.view(-1))

            # Diffusion loss
            diffusion_loss = (
                outputs.diffusion_loss
                if outputs.diffusion_loss is not None
                else torch.tensor(0.0, device=ce_loss.device)
            )
            total = training_args.ce_loss_weight * ce_loss + \
                    training_args.diffusion_loss_weight * diffusion_loss

            # Logs
            try:
                prefix = "train" if model.training else "eval"
                self.log({
                    f"{prefix}/ce_loss": ce_loss.detach().item(),
                    f"{prefix}/diffusion_loss": (
                        diffusion_loss.detach().item()
                        if isinstance(diffusion_loss, torch.Tensor)
                        else float(diffusion_loss)
                    ),
                })
            except Exception:
                pass

            return (total, outputs) if return_outputs else total

        def _save(self, output_dir=None, state_dict=None):
            """Save LoRA adapters, diffusion head, and connectors."""
            try:
                target_dir = output_dir or self.args.output_dir
                lora_out = os.path.join(target_dir, "lora")
                os.makedirs(lora_out, exist_ok=True)

                # LLM PEFT adapters
                language_model = getattr(self.model.model, "language_model", None)
                if hasattr(language_model, "save_pretrained"):
                    language_model.save_pretrained(lora_out)

                # Full diffusion head state_dict
                pred_head = getattr(self.model.model, "prediction_head", None)
                if pred_head is not None and hasattr(pred_head, "state_dict"):
                    sd = pred_head.state_dict()
                    torch.save(sd, os.path.join(lora_out, "diffusion_head_full.bin"))

                # Connectors
                ac = getattr(self.model.model, "acoustic_connector", None)
                if ac is not None:
                    ac_dir = os.path.join(lora_out, "acoustic_connector")
                    os.makedirs(ac_dir, exist_ok=True)
                    torch.save(ac.state_dict(), os.path.join(ac_dir, "pytorch_model.bin"))

                se = getattr(self.model.model, "semantic_connector", None)
                if se is not None:
                    se_dir = os.path.join(lora_out, "semantic_connector")
                    os.makedirs(se_dir, exist_ok=True)
                    torch.save(se.state_dict(), os.path.join(se_dir, "pytorch_model.bin"))

            except Exception as e:
                logger.warning(f"Failed to save LoRA assets: {e}")

    # ========================================================================
    # Build and run trainer
    # ========================================================================

    # Auto-calculate EMA decay based on total training steps.
    # High decay (0.999) needs ~3000 steps to let training dominate.
    # For short runs we lower the decay so trained weights are actually saved.
    ema_decay_val = getattr(training_args, "ema_decay", -1.0)
    if ema_decay_val == 0:
        logger.info("EMA disabled (ema_decay=0). Diffusion head will use raw trained weights.")
        ema_cb = None
    elif ema_decay_val > 0:
        logger.info(f"Using user-specified EMA decay: {ema_decay_val}")
        ema_cb = EmaCallback(attr_path="model.prediction_head", decay=ema_decay_val, device="cpu")
    else:
        # Auto-calculate: estimate total optimizer steps
        n_samples = len(train_dataset)
        eff_batch = max(1, training_args.per_device_train_batch_size * int(training_args.gradient_accumulation_steps))
        steps_per_epoch = max(1, n_samples // eff_batch)
        total_steps = steps_per_epoch * int(training_args.num_train_epochs)

        # Target: EMA should retain at most ~40% of original weights
        # decay^total_steps = 0.4  =>  decay = 0.4^(1/total_steps)
        import math
        if total_steps < 50:
            auto_decay = 0.98
        elif total_steps < 200:
            auto_decay = round(0.4 ** (1.0 / total_steps), 4)
        elif total_steps < 1000:
            auto_decay = round(0.4 ** (1.0 / total_steps), 5)
        else:
            auto_decay = 0.999  # Standard value for long training
        logger.info(f"Auto EMA decay: {auto_decay} (estimated {total_steps} total steps, {n_samples} samples, eff_batch={eff_batch})")
        ema_cb = EmaCallback(attr_path="model.prediction_head", decay=auto_decay, device="cpu")

    callbacks = []
    if ema_cb is not None:
        callbacks.append(ema_cb)

    # Save checkpoints every N epochs with checkpoint-epoch-{N} naming
    save_interval = getattr(training_args, "save_interval", 0)
    epoch_save_cb = None
    if save_interval > 0:
        class EpochIntervalCallback(TrainerCallback):
            """Manually save every N-th epoch with epoch-based naming."""
            _trainer_ref = None

            def on_epoch_end(self, args, state, control, **kwargs):
                current_epoch = int(round(state.epoch))
                if current_epoch % save_interval == 0 and self._trainer_ref is not None:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{current_epoch}")
                    logger.info(f"Saving checkpoint at epoch {current_epoch} -> {ckpt_dir}")
                    self._trainer_ref.save_model(ckpt_dir)
        epoch_save_cb = EpochIntervalCallback()
        callbacks.append(epoch_save_cb)

    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Inject trainer reference for epoch checkpoint saving
    if epoch_save_cb is not None:
        epoch_save_cb._trainer_ref = trainer

    if getattr(training_args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.warning("Failed to enable gradient checkpointing.")

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        # Final save
        lora_out = os.path.join(training_args.output_dir, "lora")
        os.makedirs(lora_out, exist_ok=True)

        # LLM PEFT
        lm = getattr(model.model, "language_model", None)
        if hasattr(lm, "save_pretrained"):
            lm.save_pretrained(lora_out)

        # Diffusion head
        ph = getattr(model.model, "prediction_head", None)
        try:
            if ph is not None and hasattr(ph, "state_dict"):
                sd = ph.state_dict()
                torch.save(sd, os.path.join(lora_out, "diffusion_head_full.bin"))
        except Exception as e:
            logger.warning(f"Failed to save diffusion head: {e}")

        # Connectors
        try:
            ac = getattr(model.model, "acoustic_connector", None)
            if ac is not None:
                ac_dir = os.path.join(lora_out, "acoustic_connector")
                os.makedirs(ac_dir, exist_ok=True)
                torch.save(ac.state_dict(), os.path.join(ac_dir, "pytorch_model.bin"))
        except Exception as e:
            logger.warning(f"Failed to save acoustic_connector: {e}")

        try:
            se = getattr(model.model, "semantic_connector", None)
            if se is not None:
                se_dir = os.path.join(lora_out, "semantic_connector")
                os.makedirs(se_dir, exist_ok=True)
                torch.save(se.state_dict(), os.path.join(se_dir, "pytorch_model.bin"))
        except Exception as e:
            logger.warning(f"Failed to save semantic_connector: {e}")

    if training_args.do_eval and eval_dataset is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
