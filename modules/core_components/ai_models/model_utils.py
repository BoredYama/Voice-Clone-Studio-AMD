"""
Model utilities for AI model management.

Shared utilities for model loading, device management, and VRAM optimization.
"""

import torch
from pathlib import Path

# --- Training process management ---
_active_training_process = None
_training_stop_requested = False


def stop_training():
    """Request the active training subprocess to stop.

    Terminates the subprocess and sets a flag so the training loop
    breaks cleanly on the next iteration.
    """
    global _active_training_process, _training_stop_requested
    _training_stop_requested = True
    if _active_training_process is not None:
        try:
            _active_training_process.terminate()
        except Exception:
            pass


def is_training_active():
    """Return True if a training subprocess is currently running."""
    if _active_training_process is None:
        return False
    return _active_training_process.poll() is None


def get_device(gpu_index=0):
    """Get the best available device (CUDA > MPS > CPU).

    Args:
        gpu_index: CUDA GPU index to use (default 0). Ignored for MPS/CPU.
    """
    if torch.cuda.is_available():
        gpu_index = int(gpu_index) if gpu_index is not None else 0
        if gpu_index >= torch.cuda.device_count():
            gpu_index = 0
        return f"cuda:{gpu_index}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_available_gpus():
    """Return list of available CUDA GPU names for UI dropdowns.

    Returns:
        List of tuples: [(index, name), ...] or empty list if no CUDA.
    """
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        gpus.append((i, name))
    return gpus


def get_dtype(device=None):
    """Get appropriate dtype based on device.

    CUDA uses bfloat16 for best quality.
    MPS uses float32 (float16 causes torch.multinomial failures during
    sampling due to NaN/subnormal values from softmax precision loss;
    unified memory makes float32 low-cost on Apple Silicon).
    CPU uses float32.
    """
    if device is None:
        device = get_device()

    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def get_attention_implementation(user_preference="auto"):
    """
    Get list of attention implementations to try, in order.

    Args:
        user_preference: User's attention preference from config:
            - "auto": Try best options in order
            - "flash_attention_2": Use Flash Attention 2
            - "sdpa": Use Scaled Dot-Product Attention
            - "eager": Use eager attention

    Returns:
        List of attention mechanism strings to try
    """
    if user_preference == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    elif user_preference == "sdpa":
        return ["sdpa", "flash_attention_2", "eager"]
    elif user_preference == "eager":
        return ["eager"]
    else:  # "auto"
        return ["flash_attention_2", "sdpa", "eager"]


# Brand mapping: model name prefixes to brand folder names.
# Used for organized storage under models/<brand>/<model_folder>/
BRAND_MAP = {
    "Qwen3": "qwen3",
    "VibeVoice": "vibevoice",
    "LuxTTS": "luxtts",
    "chatterbox": "chatterbox",
}


def get_model_brand(model_id):
    """Derive brand folder name from a model ID.

    Checks the model name (part after '/') against BRAND_MAP prefixes.
    Falls back to the HuggingFace org/author name lowercased.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Brand folder name string (e.g., "qwen3")
    """
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    for prefix, brand in BRAND_MAP.items():
        if model_name.startswith(prefix):
            return brand

    # Fallback: use the org/author name lowercased
    if "/" in model_id:
        return model_id.split("/")[0].lower()
    return ""


def _has_model_files(path):
    """Check if a directory contains recognized model files."""
    return path.exists() and (
        list(path.glob("*.safetensors"))
        or list(path.glob("*.onnx"))
        or list(path.glob("*.pt"))
    )


def check_model_available_locally(model_name):
    """
    Check if model is available in local models directory.

    Searches in brand subfolder first (e.g., models/qwen3/ModelName/),
    then falls back to flat layout (models/ModelName/) for backward
    compatibility.

    Args:
        model_name: Model name/path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Path to local model or None if not found
    """
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    folder_name = model_name.split("/")[-1]

    # 1. Try brand subfolder: models/<brand>/<folder_name>/
    brand = get_model_brand(model_name)
    if brand:
        brand_path = models_dir / brand / folder_name
        if _has_model_files(brand_path):
            return brand_path

    # 2. Fallback: flat layout models/<folder_name>/ (backward compat)
    flat_path = models_dir / folder_name
    if _has_model_files(flat_path):
        return flat_path

    return None


def download_model_from_huggingface(model_id, models_dir=None, local_folder_name=None, progress=None):
    """Download model from HuggingFace using git clone (not cache).

    Uses git-lfs to download directly to models/ folder without using HF cache.
    Users can also manually clone with:
    git clone https://huggingface.co/{model_id} models/{folder_name}

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        models_dir: Path to models directory (defaults to project models folder)
        local_folder_name: Custom local folder name (default: extract from model_id)
        progress: Optional Gradio progress callback

    Returns:
        Tuple: (success: bool, message: str, local_path: str or None)
    """
    import subprocess
    import threading

    try:
        # Validate inputs
        if not model_id or "/" not in model_id:
            return False, f"Invalid model ID: {model_id}. Use format 'Author/ModelName'", None

        # Determine local folder name
        if not local_folder_name:
            local_folder_name = model_id.split("/")[-1]

        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
        else:
            models_dir = Path(models_dir)

        # Organize into brand subfolder (e.g., models/qwen3/Qwen3-TTS-...)
        brand = get_model_brand(model_id)
        if brand:
            brand_dir = models_dir / brand
            brand_dir.mkdir(parents=True, exist_ok=True)
            local_path = brand_dir / local_folder_name
        else:
            models_dir.mkdir(exist_ok=True)
            local_path = models_dir / local_folder_name

        # Check if already downloaded (look for model files)
        # Check brand subfolder first, then flat layout for backward compat
        if _has_model_files(local_path):
            return True, f"Model already exists at {local_path}", str(local_path)

        # Also check flat layout (backward compat: models/<folder_name>/)
        flat_path = models_dir / local_folder_name
        if flat_path != local_path and _has_model_files(flat_path):
            return True, f"Model already exists at {flat_path}", str(flat_path)

        # Check if git-lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            error_msg = (
                "git-lfs is not installed or not in PATH. Install from: https://git-lfs.com\n"
                "Or manually download from HuggingFace and place in: " + str(local_path.relative_to(local_path.parents[2]) if len(local_path.parts) > 2 else local_path)
            )
            print(error_msg, flush=True)
            return False, error_msg, None

        # Clone repository with git-lfs
        hf_url = f"https://huggingface.co/{model_id}"

        try:
            print(f"\nStarting download: {model_id}", flush=True)
            print(f"URL: {hf_url}", flush=True)
            print(f"Destination: {local_path}\n", flush=True)

            # Track download state
            download_complete = {"done": False, "returncode": None}

            def run_download():
                """Run git clone without capturing output so it shows in console."""
                try:
                    result = subprocess.run(
                        ["git", "clone", hf_url, str(local_path)],
                        timeout=3600
                    )
                    download_complete["returncode"] = result.returncode
                except Exception as e:
                    print(f"Download error: {e}", flush=True)
                    download_complete["returncode"] = -1
                finally:
                    download_complete["done"] = True

            # Start download thread
            download_thread = threading.Thread(target=run_download, daemon=True)
            download_thread.start()

            # Wait for download to complete (progress shown in console)
            download_thread.join()

            if download_complete["returncode"] != 0:
                return False, "Download failed. Check console for details.", None

            # Verify model files exist
            if not (list(local_path.glob("*.safetensors")) or list(local_path.glob("*.onnx")) or list(local_path.glob("*.pt"))):
                return False, "Model files not found - download may be incomplete.", None

            print(f"\nSuccessfully downloaded to {local_path}\n", flush=True)
            return True, f"Successfully downloaded to {local_path}", str(local_path)

        except subprocess.TimeoutExpired:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, "Download timed out after 1 hour. Check your internet connection and try again.", None
        except Exception as e:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, f"Download error: {str(e)}", None

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None


def empty_device_cache():
    """Empty GPU cache (CUDA or MPS) if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Keep old name as alias for compatibility
empty_cuda_cache = empty_device_cache


# ============================================================================
# Pre-model-load hooks
# External processes (e.g., llama.cpp server) register shutdown callbacks here
# so they get stopped before any AI model loads to free VRAM.
# ============================================================================

_pre_load_hooks = []


def register_pre_load_hook(hook):
    """Register a callback to run before any AI model is loaded.

    Used by external processes (e.g., llama.cpp) that need to be shut
    down to free VRAM before loading GPU-resident models.
    """
    if hook not in _pre_load_hooks:
        _pre_load_hooks.append(hook)


def run_pre_load_hooks():
    """Run all registered pre-load hooks (e.g., stop llama.cpp server)."""
    for hook in _pre_load_hooks:
        try:
            hook()
        except Exception:
            pass


def set_seed(seed):
    """Set random seed across all available devices for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't expose detailed memory stats, just note it's active
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: MPS device active (detailed stats not available)")


def get_trained_models(models_dir=None):
    """
    Find trained model checkpoints in the models directory.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                for checkpoint in folder.glob("checkpoint-*"):
                    if checkpoint.is_dir():
                        # Qwen checkpoints contain model.safetensors
                        if not (checkpoint / "model.safetensors").exists():
                            continue
                        # Extract epoch number for sorting
                        epoch_num = 0
                        parts = checkpoint.name.split("-")
                        for i, part in enumerate(parts):
                            if part == "epoch" and i + 1 < len(parts):
                                try:
                                    epoch_num = int(parts[i + 1])
                                except ValueError:
                                    pass
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name,
                            '_epoch': epoch_num
                        })
    # Sort by speaker name ascending, then epoch descending (highest first)
    models.sort(key=lambda m: (m['speaker_name'].lower(), -m['_epoch']))
    for m in models:
        del m['_epoch']
    return models


def get_trained_model_names(models_dir=None):
    """Get list of existing trained model folder names.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of folder name strings
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    if not models_dir.exists():
        return []

    return [folder.name for folder in models_dir.iterdir() if folder.is_dir()]


def get_trained_vibevoice_models(models_dir=None):
    """Find trained VibeVoice LoRA checkpoints in the models directory.

    VibeVoice LoRA checkpoints have a lora/ subdirectory containing
    adapter_config.json. This searches both the top-level model folder
    (final save) and any checkpoint-epoch-* subdirectories (interval saves).

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    def _has_adapter_files(directory):
        """Check if a directory contains LoRA adapter files (directly or in lora/ subdir)."""
        for candidate in [directory / "lora", directory]:
            if (candidate / "adapter_model.safetensors").exists():
                return True
            if (candidate / "adapter_model.bin").exists():
                return True
        return False

    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                # Check top-level (supports both folder/lora/files and folder/files layouts)
                if _has_adapter_files(folder):
                    models.append({
                        'display_name': folder.name,
                        'path': str(folder),
                        'speaker_name': folder.name,
                        '_epoch': 999999,
                    })

                # Check checkpoint-epoch-* subdirs (interval saves)
                for checkpoint in folder.glob("checkpoint-epoch-*"):
                    if checkpoint.is_dir() and _has_adapter_files(checkpoint):
                        epoch_num = 0
                        parts = checkpoint.name.split("-")
                        for i, part in enumerate(parts):
                            if part == "epoch" and i + 1 < len(parts):
                                try:
                                    epoch_num = int(parts[i + 1])
                                except ValueError:
                                    pass
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name,
                            '_epoch': epoch_num,
                        })

    # Sort by speaker name ascending, then epoch descending (final model on top)
    models.sort(key=lambda m: (m['speaker_name'].lower(), -m['_epoch']))
    for m in models:
        del m['_epoch']
    return models


def train_model(folder, speaker_name, ref_audio_filename, batch_size,
                learning_rate, num_epochs, save_interval,
                user_config, datasets_dir, project_root,
                play_completion_beep=None, progress=None):
    """Complete training workflow: validate, prepare data, and train model.

    Args:
        folder: Dataset subfolder name
        speaker_name: Name for the trained model/speaker
        ref_audio_filename: Reference audio file from dataset
        batch_size: Training batch size
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        save_interval: Save checkpoint every N epochs
        user_config: User configuration dict
        datasets_dir: Path to datasets directory
        project_root: Path to project root
        play_completion_beep: Optional callback for audio notification
        progress: Optional Gradio progress callback
    """
    import subprocess
    import sys
    import json
    import os
    from modules.core_components.audio_utils import check_audio_format

    if progress is None:
        def progress(*a, **kw):
            pass

    # ============== STEP 1: Validation ==============
    progress(0.0, desc="Validating dataset...")

    if not folder or folder == "(No folders)" or folder == "(Select Dataset)":
        return "Error: Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "Error: Please enter a speaker name"

    if not ref_audio_filename:
        return "Error: Please select a reference audio file"

    if save_interval is None:
        save_interval = 5

    # Create output directory
    trained_models_folder = user_config.get("trained_models_folder", "trained_models")
    output_dir = project_root / trained_models_folder / speaker_name.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = datasets_dir / folder
    if not base_dir.exists():
        return f"Error: Folder not found: {folder}"

    ref_audio_path = base_dir / ref_audio_filename
    if not ref_audio_path.exists():
        return f"Error: Reference audio not found: {ref_audio_filename}"

    # Only get audio files
    audio_files = [f for f in (list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                   if not f.name.endswith('.txt') and not f.name.endswith('.jsonl')]
    if not audio_files:
        return "Error: No audio files found in folder"

    issues = []
    valid_files = []
    converted_count = 0
    total = len(audio_files)

    status_log = []
    status_log.append("=" * 60)
    status_log.append("STEP 1/3: DATASET VALIDATION")
    status_log.append("=" * 60)

    for i, audio_path in enumerate(audio_files):
        progress(0.0, desc=f"Validating {audio_path.name}...")

        txt_path = audio_path.with_suffix(".txt")

        if not txt_path.exists():
            issues.append(f"[X] {audio_path.name}: Missing transcript")
            continue

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                issues.append(f"[X] {audio_path.name}: Empty transcript")
                continue
        except Exception:
            issues.append(f"[X] {audio_path.name}: Cannot read transcript")
            continue

        is_correct, info = check_audio_format(str(audio_path))
        if not is_correct:
            if not info:
                issues.append(f"[X] {audio_path.name}: Cannot read audio file")
                continue

            progress(0.0, desc=f"Converting {audio_path.name}...")
            temp_output = audio_path.parent / f"temp_{audio_path.name}"
            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le', str(temp_output)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and temp_output.exists():
                    audio_path.unlink()
                    temp_output.rename(audio_path)
                    converted_count += 1
                else:
                    issues.append(f"[X] {audio_path.name}: Conversion failed - {result.stderr[:100]}")
                    continue
            except FileNotFoundError:
                issues.append(f"[X] {audio_path.name}: ffmpeg not found")
                continue
            except Exception as e:
                issues.append(f"[X] {audio_path.name}: Conversion error - {str(e)[:100]}")
                continue

        valid_files.append(audio_path.name)

    if not valid_files:
        return "Error: No valid training samples found\n" + "\n".join(issues[:10])

    status_log.append(f"Found {len(valid_files)} valid training samples")
    if converted_count > 0:
        status_log.append(f"Auto-converted {converted_count} files to 24kHz 16-bit mono")
    if issues:
        status_log.append(f"{len(issues)} files skipped:")
        for issue in issues[:5]:
            status_log.append(f"   {issue}")
        if len(issues) > 5:
            status_log.append(f"   ... and {len(issues) - 5} more")

    # Ensure reference audio is correct format
    progress(0.0, desc="Preparing reference audio...")
    is_correct, info = check_audio_format(str(ref_audio_path))
    if not is_correct:
        temp_output = ref_audio_path.parent / f"temp_{ref_audio_path.name}"
        cmd = [
            'ffmpeg', '-y', '-i', str(ref_audio_path),
            '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
            '-acodec', 'pcm_s16le', str(temp_output)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and temp_output.exists():
            ref_audio_path.unlink()
            temp_output.rename(ref_audio_path)
        else:
            return f"Error: Failed to convert reference audio: {result.stderr[:200]}"

    # Generate train_raw.jsonl
    progress(0.0, desc="Preparing training data...")
    train_raw_path = base_dir / "train_raw.jsonl"
    jsonl_entries = []

    for filename in valid_files:
        audio_path_entry = base_dir / filename
        txt_path = audio_path_entry.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip()

        entry = {
            "audio": str(audio_path_entry.absolute()),
            "text": transcript,
            "ref_audio": str(ref_audio_path.absolute())
        }
        jsonl_entries.append(entry)

    try:
        with open(train_raw_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        status_log.append(f"Generated train_raw.jsonl with {len(jsonl_entries)} entries")
    except Exception as e:
        return f"Error: Failed to write train_raw.jsonl: {str(e)}"

    # ============== STEP 2: Prepare Data (extract audio codes) ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 2/3: EXTRACTING AUDIO CODES")
    status_log.append("=" * 60)
    progress(0.0, desc="Extracting audio codes...")

    train_with_codes_path = base_dir / "train_with_codes.jsonl"
    modules_dir = project_root / "modules"
    prepare_script = modules_dir / "qwen_finetune" / "prepare_data.py"

    if not prepare_script.exists():
        status_log.append("[X] Qwen3-TTS finetuning scripts not found!")
        status_log.append("   Please ensure Qwen3-TTS repository is cloned.")
        return "\n".join(status_log)

    prepare_cmd = [
        sys.executable,
        "-u",
        str(prepare_script.absolute()),
        "--device", get_device(),
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "--input_jsonl", str(train_raw_path),
        "--output_jsonl", str(train_with_codes_path)
    ]

    status_log.append(f"Running: {' '.join(prepare_cmd)}")
    status_log.append("")

    try:
        global _active_training_process, _training_stop_requested
        _training_stop_requested = False

        result = subprocess.Popen(
            prepare_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(base_dir)
        )
        _active_training_process = result

        for line in result.stdout:
            if _training_stop_requested:
                break
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

        if _training_stop_requested:
            try:
                result.kill()
                result.wait(timeout=5)
            except Exception:
                pass
            _active_training_process = None
            status_log.append("")
            status_log.append("Training stopped by user.")
            return "\n".join(status_log)

        result.wait()

        if result.returncode != 0:
            status_log.append(f"[X] prepare_data.py failed with exit code {result.returncode}")
            return "\n".join(status_log)

        if not train_with_codes_path.exists():
            status_log.append("[X] train_with_codes.jsonl was not generated")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("[OK] Audio codes extracted successfully")

    except Exception as e:
        status_log.append(f"[X] Error running prepare_data.py: {str(e)}")
        return "\n".join(status_log)

    # ============== STEP 3: Fine-tune ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 3/3: TRAINING MODEL")
    status_log.append("=" * 60)
    progress(0.0, desc="Starting training...")

    sft_script = modules_dir / "qwen_finetune" / "sft_12hz.py"

    if not sft_script.exists():
        status_log.append("[X] sft_12hz.py not found!")
        return "\n".join(status_log)

    base_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    status_log.append(f"Locating base model: {base_model_id}")
    try:
        from huggingface_hub import snapshot_download
        offline_mode = user_config.get("offline_mode", False)
        base_model_path = snapshot_download(
            repo_id=base_model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.npz"],
            local_files_only=offline_mode
        )
        status_log.append(f"[OK] Using cached model at: {base_model_path}")
    except Exception as e:
        status_log.append(f"[X] Failed to locate/download base model: {str(e)}")
        return "\n".join(status_log)

    attn_impl = user_config.get("attention_mechanism", "auto")

    sft_cmd = [
        sys.executable,
        "-u",
        str(sft_script.absolute()),
        "--init_model_path", base_model_path,
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_with_codes_path),
        "--batch_size", str(int(batch_size)),
        "--lr", str(learning_rate),
        "--num_epochs", str(int(num_epochs)),
        "--save_interval", str(int(save_interval)),
        "--speaker_name", speaker_name.strip().lower(),
        "--attn_implementation", attn_impl
    ]

    status_log.append("")
    status_log.append("Training configuration:")
    status_log.append(f"  Base model: {base_model_id}")
    status_log.append(f"  Attention implementation: {attn_impl}")
    status_log.append(f"  Batch size: {int(batch_size)}")
    status_log.append(f"  Learning rate: {learning_rate}")
    status_log.append(f"  Epochs: {int(num_epochs)}")
    status_log.append(f"  Save interval: Every {int(save_interval)} epoch(s)" if save_interval > 0 else "  Save interval: Every epoch")
    status_log.append(f"  Speaker name: {speaker_name.strip()}")
    status_log.append(f"  Output: {output_dir}")
    status_log.append("")
    status_log.append("Starting training...")
    status_log.append(f"Running: {' '.join([str(arg) for arg in sft_cmd])}")
    status_log.append("")

    try:
        env = os.environ.copy()
        env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'

        result = subprocess.Popen(
            sft_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        _active_training_process = result

        # Track steps per epoch to compute accurate progress
        max_step_seen = 0
        total_epochs = int(num_epochs)

        for line in result.stdout:
            if _training_stop_requested:
                break
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

                if "Epoch" in line and "Step" in line:
                    try:
                        epoch_num = int(line.split("Epoch")[1].split("|")[0].strip())
                        step_num = int(line.split("Step")[1].split("|")[0].strip())
                        if step_num > max_step_seen:
                            max_step_seen = step_num
                        # Calculate progress: 0.0 to 1.0 based on epoch/step
                        if max_step_seen > 0:
                            epoch_progress = (epoch_num * (max_step_seen + 1) + step_num) / (total_epochs * (max_step_seen + 1))
                        else:
                            epoch_progress = epoch_num / total_epochs
                        progress_val = epoch_progress
                        # Extract loss for the description
                        loss_str = ""
                        if "Loss:" in line:
                            loss_str = " | Loss: " + line.split("Loss:")[1].strip()
                        progress(progress_val, desc=f"Training: Epoch {epoch_num + 1}/{total_epochs} | Step {step_num}{loss_str}")
                    except Exception:
                        pass

        if _training_stop_requested:
            try:
                result.kill()
                result.wait(timeout=5)
            except Exception:
                pass
            _active_training_process = None
            status_log.append("")
            status_log.append("Training stopped by user.")
            return "\n".join(status_log)

        result.wait()
        _active_training_process = None

        if result.returncode != 0:
            status_log.append("")
            status_log.append(f"[X] Training failed with exit code {result.returncode}")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("=" * 60)
        status_log.append("TRAINING COMPLETED SUCCESSFULLY!")
        status_log.append("=" * 60)
        status_log.append(f"Model saved to: {output_dir}")
        status_log.append(f"Speaker name: {speaker_name.strip()}")
        status_log.append("")
        status_log.append("To use your trained model:")
        status_log.append("  1. Go to Voice Presets tab")
        status_log.append("  2. Select 'Trained Models' radio button")
        status_log.append(f"  3. Click refresh and select '{speaker_name.strip()}'")

        progress(1.0, desc="Training complete!")
        if play_completion_beep:
            play_completion_beep()

    except Exception as e:
        status_log.append(f"[X] Error during training: {str(e)}")
        return "\n".join(status_log)

    return "\n".join(status_log)


def train_vibevoice_model(folder, speaker_name, batch_size, learning_rate,
                          num_epochs, save_interval, ddpm_batch_mul,
                          diffusion_loss_weight, ce_loss_weight,
                          voice_prompt_drop_rate, train_diffusion_head,
                          gradient_accumulation_steps, warmup_steps,
                          ema_decay,
                          user_config, datasets_dir, project_root,
                          play_completion_beep=None, progress=None):
    """Complete VibeVoice LoRA training workflow.

    Uses the vendored training script via subprocess, same pattern as Qwen3 training.

    Args:
        folder: Dataset subfolder name
        speaker_name: Name for the trained model/speaker
        batch_size: Training batch size
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        ddpm_batch_mul: Diffusion batch multiplier
        diffusion_loss_weight: Weight for diffusion loss
        ce_loss_weight: Weight for cross-entropy loss
        voice_prompt_drop_rate: Probability to drop voice prompts during training
        train_diffusion_head: Whether to train the diffusion head
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Warmup steps for learning rate scheduler
        user_config: User configuration dict
        datasets_dir: Path to datasets directory
        project_root: Path to project root
        play_completion_beep: Optional callback for audio notification
        progress: Optional Gradio progress callback
    """
    import subprocess
    import sys
    import json
    import os
    from modules.core_components.audio_utils import check_audio_format

    if progress is None:
        def progress(*a, **kw):
            pass

    # ============== STEP 1: Validation ==============
    progress(0.0, desc="Validating dataset...")

    if not folder or folder == "(No folders)" or folder == "(Select Dataset)":
        return "Error: Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "Error: Please enter a speaker name"

    # Create output directory
    trained_models_folder = user_config.get("trained_models_folder", "trained_models")
    output_dir = project_root / trained_models_folder / speaker_name.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = datasets_dir / folder
    if not base_dir.exists():
        return f"Error: Folder not found: {folder}"

    # Collect audio files
    audio_files = [f for f in (list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                   if not f.name.endswith('.txt') and not f.name.endswith('.jsonl')]
    if not audio_files:
        return "Error: No audio files found in folder"

    issues = []
    valid_files = []
    converted_count = 0

    status_log = []
    status_log.append("=" * 60)
    status_log.append("STEP 1/2: DATASET VALIDATION")
    status_log.append("=" * 60)

    for audio_path in audio_files:
        progress(0.0, desc=f"Validating {audio_path.name}...")

        txt_path = audio_path.with_suffix(".txt")
        if not txt_path.exists():
            issues.append(f"[X] {audio_path.name}: Missing transcript")
            continue

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                issues.append(f"[X] {audio_path.name}: Empty transcript")
                continue
        except Exception:
            issues.append(f"[X] {audio_path.name}: Cannot read transcript")
            continue

        is_correct, info = check_audio_format(str(audio_path))
        if not is_correct:
            if not info:
                issues.append(f"[X] {audio_path.name}: Cannot read audio file")
                continue

            progress(0.0, desc=f"Converting {audio_path.name}...")
            temp_output = audio_path.parent / f"temp_{audio_path.name}"
            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le', str(temp_output)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and temp_output.exists():
                    audio_path.unlink()
                    temp_output.rename(audio_path)
                    converted_count += 1
                else:
                    issues.append(f"[X] {audio_path.name}: Conversion failed - {result.stderr[:100]}")
                    continue
            except FileNotFoundError:
                issues.append(f"[X] {audio_path.name}: ffmpeg not found")
                continue
            except Exception as e:
                issues.append(f"[X] {audio_path.name}: Conversion error - {str(e)[:100]}")
                continue

        valid_files.append(audio_path.name)

    if not valid_files:
        return "Error: No valid training samples found\n" + "\n".join(issues[:10])

    status_log.append(f"Found {len(valid_files)} valid training samples")
    if converted_count > 0:
        status_log.append(f"Auto-converted {converted_count} files to 24kHz 16-bit mono")
    if issues:
        status_log.append(f"{len(issues)} files skipped:")
        for issue in issues[:5]:
            status_log.append(f"   {issue}")
        if len(issues) > 5:
            status_log.append(f"   ... and {len(issues) - 5} more")

    # Generate train.jsonl for VibeVoice
    progress(0.0, desc="Preparing training data...")
    train_jsonl_path = base_dir / "train_vibevoice.jsonl"
    jsonl_entries = []

    for filename in valid_files:
        audio_path_entry = base_dir / filename
        txt_path = audio_path_entry.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip()

        entry = {
            "audio": str(audio_path_entry.absolute()),
            "text": transcript,
        }
        jsonl_entries.append(entry)

    try:
        with open(train_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        status_log.append(f"Generated train_vibevoice.jsonl with {len(jsonl_entries)} entries")
    except Exception as e:
        return f"Error: Failed to write train_vibevoice.jsonl: {str(e)}"

    # ============== STEP 2: Train VibeVoice LoRA ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 2/2: TRAINING VIBEVOICE LORA")
    status_log.append("=" * 60)
    progress(0.0, desc="Starting VibeVoice training...")

    train_script = project_root / "modules" / "vibevoice_tts" / "finetune" / "train_vibevoice.py"

    if not train_script.exists():
        status_log.append("[X] VibeVoice training script not found!")
        return "\n".join(status_log)

    # Try FranckyB/VibeVoice-1.5B first (same weights, often already cached
    # from voice cloning), then fall back to vibevoice/VibeVoice-1.5B.
    base_model_candidates = ["FranckyB/VibeVoice-1.5B", "vibevoice/VibeVoice-1.5B"]
    base_model_path = None

    # Check local models/ directory first
    for candidate in base_model_candidates:
        local = check_model_available_locally(candidate)
        if local:
            base_model_path = str(local)
            status_log.append(f"[OK] Using local model: {base_model_path}")
            break

    # Fall back to HF cache / download
    if not base_model_path:
        from huggingface_hub import snapshot_download
        offline_mode = user_config.get("offline_mode", False)
        for candidate in base_model_candidates:
            try:
                base_model_path = snapshot_download(
                    repo_id=candidate,
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.npz", "*.model"],
                    local_files_only=offline_mode
                )
                status_log.append(f"[OK] Using cached model ({candidate}): {base_model_path}")
                break
            except Exception:
                continue

    if not base_model_path:
        status_log.append("[X] Failed to locate VibeVoice-1.5B base model. "
                          "Download it via Model Management or run voice cloning first.")
        return "\n".join(status_log)

    use_bf16 = torch.cuda.is_available()

    train_cmd = [
        sys.executable,
        "-u",
        str(train_script.absolute()),
        "--model_name_or_path", base_model_path,
        "--train_jsonl", str(train_jsonl_path),
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", str(int(batch_size)),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(int(num_epochs)),
        "--ddpm_batch_mul", str(int(ddpm_batch_mul)),
        "--diffusion_loss_weight", str(diffusion_loss_weight),
        "--ce_loss_weight", str(ce_loss_weight),
        "--voice_prompt_drop_rate", str(voice_prompt_drop_rate),
        "--train_diffusion_head", str(train_diffusion_head),
        "--gradient_accumulation_steps", str(int(gradient_accumulation_steps)),
        "--warmup_steps", str(int(warmup_steps)),
        "--ema_decay", str(float(ema_decay)),
        "--logging_steps", "10",
        "--do_train",
        "--gradient_clipping",
        "--max_grad_norm", "0.8",
        "--lr_scheduler_type", "cosine",
    ]

    # Always train connectors when training diffusion head — they bridge
    # the LoRA-modified LLM hidden states to the diffusion head's input space
    if train_diffusion_head:
        train_cmd.extend(["--train_connectors", "True"])

    save_interval = int(save_interval) if save_interval else 0
    if save_interval > 0:
        train_cmd.extend(["--save_interval", str(save_interval)])

    if use_bf16:
        train_cmd.append("--bf16")

    status_log.append("")
    status_log.append("Training configuration:")
    status_log.append(f"  Base model: {base_model_path}")
    status_log.append(f"  Batch size: {int(batch_size)}")
    status_log.append(f"  Learning rate: {learning_rate}")
    status_log.append(f"  Epochs: {int(num_epochs)}")
    status_log.append(f"  DDPM batch multiplier: {int(ddpm_batch_mul)}")
    status_log.append(f"  Diffusion loss weight: {diffusion_loss_weight}")
    status_log.append(f"  CE loss weight: {ce_loss_weight}")
    status_log.append(f"  Voice prompt drop rate: {voice_prompt_drop_rate}")
    status_log.append(f"  Train diffusion head: {train_diffusion_head}")
    status_log.append(f"  Gradient accumulation: {int(gradient_accumulation_steps)}")
    status_log.append(f"  Warmup steps: {int(warmup_steps)}")
    status_log.append(f"  EMA decay: {float(ema_decay)}" if float(ema_decay) > 0 else "  EMA decay: disabled")
    status_log.append(f"  Save interval: {'every ' + str(save_interval) + ' epoch(s)' if save_interval > 0 else 'final only'}")
    status_log.append(f"  Speaker name: {speaker_name.strip()}")
    status_log.append(f"  Output: {output_dir}")
    status_log.append("")
    status_log.append("Starting training...")
    status_log.append("")

    try:
        global _active_training_process, _training_stop_requested
        _training_stop_requested = False

        env = os.environ.copy()
        env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        # Ensure project root is on PYTHONPATH so "from modules.*" imports work
        env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

        result = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        _active_training_process = result

        total_epochs = int(num_epochs)

        # Skip verbose HF Trainer config dumps and noisy warnings
        _vv_noise = (
            'TrainingArguments(', 'CustomTrainingArguments(',
            'IntervalStrategy.', 'SchedulerType.', 'SaveStrategy.',
            'OptimizerNames.', 'HubStrategy.',
            '<HUB_TOKEN>', '<PUSH_TO_HUB_TOKEN>',
            'is deprecated', 'tokenizer class you load',
            'It may result in unexpected', 'The class this function',
            'Loading checkpoint shards:',
            'accelerator_config=', 'fsdp_config=',
        )
        # Lines that are just config key=value pairs from the dump
        _vv_config_prefixes = (
            '_n_gpu=', 'adafactor=', 'adam_', 'auto_find', 'average_tokens',
            'batch_eval', 'bf16', 'ce_loss_weight=', 'data_seed',
            'dataloader_', 'ddp_', 'ddpm_batch_mul=', 'debug=', 'deepspeed=',
            'diffusion_loss_weight=', 'disable_tqdm', 'do_eval=', 'do_predict=',
            'do_train=', 'eval_', 'fp16', 'fsdp', 'full_determinism',
            'gradient_', 'greater_is_better', 'group_by_length',
            'half_precision', 'hub_', 'ignore_data', 'include_',
            'jit_mode', 'label_', 'learning_rate=', 'length_column',
            'liger_', 'load_best', 'local_rank', 'log_', 'logging_',
            'lr_scheduler', 'max_grad', 'max_steps', 'metric_for',
            'mp_parameters', 'neftune', 'no_cuda', 'num_train_epochs=',
            'optim', 'output_dir=', 'overwrite_', 'parallelism',
            'past_index', 'per_device_', 'prediction_loss', 'project=',
            'push_to_hub', 'ray_scope', 'remove_unused', 'report_to=',
            'restore_callback', 'resume_from', 'run_name', 'save_',
            'seed=', 'skip_memory', 'tf32', 'torch_compile', 'torch_empty',
            'torchdynamo', 'tpu_', 'trackio', 'use_cpu', 'use_legacy',
            'use_liger', 'use_mps', 'warmup_', 'weight_decay', ')',
        )

        for line in result.stdout:
            if _training_stop_requested:
                break
            line = line.strip()
            if line:
                # Skip verbose config dump and noisy warnings
                if any(p in line for p in _vv_noise):
                    continue
                if line.startswith(_vv_config_prefixes):
                    continue

                # Skip tqdm progress bars (e.g. "  23%|##3       | 7/30 [00:27...")
                if '%|' in line and '[' in line and '/' in line:
                    continue

                # Parse loss dicts: show clean summary instead of raw dicts
                if ("'train/ce_loss'" in line or "'train/diffusion_loss'" in line
                        or "'loss'" in line) and "'epoch'" in line:
                    try:
                        import re
                        epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                        if epoch_match:
                            current_epoch = float(epoch_match.group(1))
                            progress_val = current_epoch / total_epochs

                            # Extract losses
                            diff_match = re.search(r"'(?:train/)?diffusion_loss':\s*([\d.]+)", line)
                            ce_match = re.search(r"'(?:train/)?ce_loss':\s*([\d.]+)", line)
                            loss_match = re.search(r"'loss':\s*([\d.]+)", line)

                            # Build clean summary
                            parts = [f"Epoch {current_epoch:.1f}/{total_epochs}"]
                            if diff_match:
                                parts.append(f"Diff Loss: {float(diff_match.group(1)):.4f}")
                            if ce_match:
                                parts.append(f"CE Loss: {float(ce_match.group(1)):.2f}")
                            if loss_match and not diff_match:
                                parts.append(f"Loss: {float(loss_match.group(1)):.4f}")
                            summary = " | ".join(parts)

                            # Only log when epoch changes (avoid duplicate per-step lines)
                            epoch_key = f"{current_epoch:.1f}"
                            if not hasattr(result, '_last_epoch') or result._last_epoch != epoch_key:
                                result._last_epoch = epoch_key
                                status_log.append(f"  {summary}")

                            # Update progress bar
                            loss_desc = ""
                            if diff_match:
                                loss_desc = f" | Loss: {float(diff_match.group(1)):.4f}"
                            elif loss_match:
                                loss_desc = f" | Loss: {float(loss_match.group(1)):.4f}"
                            progress(
                                min(progress_val, 0.99),
                                desc=f"Training: Epoch {current_epoch:.1f}/{total_epochs}{loss_desc}"
                            )
                    except Exception:
                        pass
                    continue

                # Skip other raw loss dict lines without epoch info
                if line.startswith('{') and ('loss' in line or 'train/' in line):
                    continue

                status_log.append(f"  {line}")

        if _training_stop_requested:
            try:
                result.kill()
                result.wait(timeout=5)
            except Exception:
                pass
            _active_training_process = None
            status_log.append("")
            status_log.append("Training stopped by user.")
            return "\n".join(status_log)

        result.wait()
        _active_training_process = None

        if result.returncode != 0:
            status_log.append("")
            status_log.append(f"[X] Training failed with exit code {result.returncode}")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("=" * 60)
        status_log.append("TRAINING COMPLETED SUCCESSFULLY!")
        status_log.append("=" * 60)
        status_log.append(f"LoRA model saved to: {output_dir}")
        status_log.append(f"Speaker name: {speaker_name.strip()}")
        status_log.append("")
        status_log.append("To use your trained model:")
        status_log.append("  1. Go to Voice Presets tab")
        status_log.append("  2. Select 'VibeVoice Trained' radio button")
        status_log.append(f"  3. Click refresh and select '{speaker_name.strip()}'")

        progress(1.0, desc="Training complete!")
        if play_completion_beep:
            play_completion_beep()

    except Exception as e:
        status_log.append(f"[X] Error during training: {str(e)}")
        return "\n".join(status_log)

    return "\n".join(status_log)
