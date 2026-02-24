"""
Audio utility functions for sample preparation.

Robust audio/video handling with proper error handling and fallbacks.
"""

import os
import re
import platform
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime


def check_audio_format(audio_path):
    """Check if audio is 24kHz, 16-bit, mono.

    Returns:
        Tuple of (is_correct, info_or_none)
    """
    try:
        info = sf.info(audio_path)
        is_correct = (info.samplerate == 24000 and
                      info.channels == 1 and
                      info.subtype == 'PCM_16')
        return is_correct, info
    except Exception:
        return False, None


def is_video_file(filepath):
    """Check if file is a video based on extension."""
    if not filepath:
        return False
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return Path(filepath).suffix.lower() in video_extensions


def is_audio_file(filepath):
    """Check if file is an audio file based on extension."""
    if not filepath:
        return False
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
    return Path(filepath).suffix.lower() in audio_extensions


def extract_audio_from_video(video_path, temp_dir):
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        temp_dir: Directory to save extracted audio

    Returns:
        str: Path to extracted audio file, or None if failed
    """
    try:
        import subprocess
        import shutil

        video_input = Path(video_path)

        # Use a deterministic output name based on the video filename
        # so re-dragging the same video reuses the cached extraction
        stem = re.sub(r'[^\w\s.-]', '', video_input.stem).strip() or 'video'
        audio_output = Path(temp_dir) / f"{stem}.wav"

        # If we already have a cached extraction, reuse it
        if audio_output.exists():
            return str(audio_output), "Reused cached audio from video"

        # Copy input video to project temp if it's outside the project
        # (Gradio uploads go to system temp which can have permission issues for ffmpeg)
        project_root = Path(temp_dir).parent
        try:
            video_input.resolve().relative_to(project_root.resolve())
            local_video = video_input  # Already in project directory
        except ValueError:
            # Video is outside project (e.g. Gradio temp) — copy it locally
            local_name = f"input_video_{stem}{video_input.suffix}"
            local_video = Path(temp_dir) / local_name
            try:
                shutil.copy2(str(video_input), str(local_video))
            except Exception:
                local_video = video_input  # Fall back to original path

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',  # Suppress banner/config output
            '-nostdin',
            '-i', str(local_video),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '24000',  # 24kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up the local video copy if we made one
        if local_video != video_input and local_video.exists():
            try:
                local_video.unlink()
            except Exception:
                pass

        if result.returncode == 0 and audio_output.exists():
            return str(audio_output), "Extracted audio from video"
        else:
            err_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
            print(f"ffmpeg error: {err_msg}")
            return None, "⚠ Failed to extract audio from video"

    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg to extract audio from video.")
        return None, "⚠ ffmpeg not found"
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, "⚠ Error extracting audio"


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        audio_data, sr = sf.read(audio_path)
        return len(audio_data) / sr
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0


def format_time(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def normalize_audio(audio_file, temp_dir):
    """
    Normalize audio levels.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save normalized audio

    Returns:
        tuple: Path to normalized audio file, or original path if failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        return None, "⚠ Audio file not found"

    try:
        data, sr = sf.read(audio_file)

        # Normalize to -1 to 1 range with conservative headroom
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized = data / max_val * 0.85  # Leave 15% headroom to prevent clipping in TTS
        else:
            normalized = data

        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"normalized_{timestamp}.wav"
        temp_path = Path(temp_dir) / filename

        try:
            sf.write(str(temp_path), normalized, sr)
        except (PermissionError, OSError) as e:
            # Fallback to system temp
            try:
                print(f"[WARN] Could not write to {temp_path} ({e}). Falling back to system temp.")
                temp_path = Path(tempfile.gettempdir()) / filename
                sf.write(str(temp_path), normalized, sr)
            except Exception as fbe:
                print(f"Fallback save failed: {fbe}")
                raise RuntimeError(
                    f"Failed to save normalized audio to both {temp_dir} and system temp. "
                    f"Primary error: {e}; fallback error: {fbe}"
                ) from fbe

        # Force file flush on Windows to prevent connection reset errors
        if platform.system() == "Windows":
            time.sleep(0.1)  # Small delay to ensure file is fully written

        return str(temp_path), "Normalized audio"

    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return audio_file, "⚠ Error normalizing audio"


def convert_to_mono(audio_file, temp_dir):
    """
    Convert stereo audio to mono.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save mono audio

    Returns:
        tuple: Path to mono audio file, or original path if already mono or failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        return None, "⚠ Audio file not found"

    try:
        data, sr = sf.read(audio_file)

        # Check if stereo, if mono return original
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono = np.mean(data, axis=1)

            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"mono_{timestamp}.wav"
            temp_path = Path(temp_dir) / filename

            try:
                sf.write(str(temp_path), mono, sr)
            except (PermissionError, OSError) as e:
                # Fallback to system temp
                print(f"[WARN] Could not write to {temp_path} ({e}). Falling back to system temp.")
                temp_path = Path(tempfile.gettempdir()) / filename
                sf.write(str(temp_path), mono, sr)

            # Force file flush on Windows
            if platform.system() == "Windows":
                time.sleep(0.1)

            return str(temp_path), "Converted to mono"
        else:
            return audio_file, "Already mono"

    except Exception as e:
        print(f"Error converting to mono: {e}")
        return audio_file, "⚠ Error converting to mono"


def clean_audio(audio_file, temp_dir, get_deepfilter_model_func, progress_callback=None):
    """
    Clean audio using DeepFilterNet.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save cleaned audio
        get_deepfilter_model_func: Function that returns (model, state, params) tuple
        progress_callback: Optional progress callback function

    Returns:
        tuple: Path to cleaned audio file, or original path if failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at path: {audio_file}")
        return None, "⚠ Audio file not found"

    try:
        if progress_callback:
            progress_callback(0.1, desc="Loading Audio Cleaner...")

        df_model, df_state, df_params = get_deepfilter_model_func()

        # Get sample rate from params or use default
        target_sr = df_params.sr if df_params is not None and hasattr(df_params, 'sr') else 48000

        if progress_callback:
            progress_callback(0.3, desc="Processing audio...")

        # Import DeepFilterNet functions
        from df.enhance import enhance
        from df.io import load_audio as df_load_audio, save_audio
        import torch

        # Load audio using DeepFilterNet's loader
        audio, _ = df_load_audio(audio_file, sr=target_sr)

        # Ensure tensor is contiguous (video-extracted audio can be non-contiguous)
        if hasattr(audio, 'is_contiguous') and not audio.is_contiguous():
            audio = audio.contiguous()

        # Run enhancement with cuDNN disabled to avoid CUDNN_STATUS_NOT_SUPPORTED
        # errors from non-contiguous intermediate tensors inside DeepFilterNet
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            enhanced_audio = enhance(df_model, df_state=df_state, audio=audio)
        finally:
            torch.backends.cudnn.enabled = cudnn_was_enabled

        # Save output
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"cleaned_{timestamp}.wav"
        output_path = Path(temp_dir) / output_filename

        # Robust save with fallback for permission/system errors
        try:
            save_audio(str(output_path), enhanced_audio, target_sr)
        except (PermissionError, OSError, RuntimeError) as e:
            msg = str(e)
            if "Permission denied" in msg or "System error" in msg:
                print(f"[WARN] Could not write to {output_path} ({msg}). Falling back to system temp.")
                output_path = Path(tempfile.gettempdir()) / output_filename
                save_audio(str(output_path), enhanced_audio, target_sr)
            else:
                raise e

        if progress_callback:
            progress_callback(1.0, desc="Done!")

        return str(output_path), "Cleaned with DeepFilterNet"

    except Exception as e:
        print(f"Error cleaning audio: {e}")
        return audio_file, "⚠ Error cleaning audio"


def save_audio_as_sample(audio_file, transcription, sample_name, samples_dir):
    """
    Save audio and transcription as a new sample.

    Args:
        audio_file: Path to audio file
        transcription: Transcription text
        sample_name: Name for the sample
        samples_dir: Directory to save sample

    Returns:
        tuple: (status_message, success_bool)
    """
    import re
    import json

    if not audio_file:
        return "[ERROR] No audio file to save.", False

    if not transcription or transcription.startswith("[ERROR]"):
        return "[ERROR] Please provide a transcription first.", False

    if not sample_name or not sample_name.strip():
        return "[ERROR] Please enter a sample name.", False

    # Clean sample name
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
    clean_name = clean_name.replace(" ", "_")

    if not clean_name:
        return "[ERROR] Invalid sample name.", False

    try:
        # Read audio file
        audio_data, sr = sf.read(audio_file)

        # Clean transcription: remove ALL text in square brackets [...]
        # This removes [Speaker X], [human sounds], [lyrics], etc.
        cleaned_transcription = re.sub(r'\[.*?\]\s*', '', transcription)
        cleaned_transcription = cleaned_transcription.strip()

        # Save wav file
        wav_path = Path(samples_dir) / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)

        # Save .json metadata
        meta = {
            "Type": "Sample",
            "Text": cleaned_transcription if cleaned_transcription else ""
        }
        json_path = Path(samples_dir) / f"{clean_name}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return f"Sample saved as '{clean_name}'", True

    except Exception as e:
        return f"[ERROR] Error saving sample: {str(e)}", False


def save_audio_to_temp(audio_data, sr, temp_dir, filename_stem):
    """Save audio data as WAV to the temp directory.

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate
        temp_dir: Path to temp directory
        filename_stem: filename without extension (e.g. 'bambie_20260210_001059')

    Returns:
        Path to the saved temp WAV file
    """
    temp_path = Path(temp_dir) / f"{filename_stem}.wav"
    sf.write(str(temp_path), audio_data, sr)
    return temp_path


def convert_audio_format(src_path, dst_path, output_format):
    """Convert a WAV file to the specified format.

    Args:
        src_path: Path to source WAV file
        dst_path: Path to destination file (extension should match format)
        output_format: 'wav', 'flac', or 'mp3'

    Returns:
        Path to the converted file
    """
    import shutil
    import subprocess

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if output_format == "wav":
        shutil.copy2(str(src_path), str(dst_path))
    elif output_format == "flac":
        # soundfile supports FLAC natively
        data, sr = sf.read(str(src_path))
        sf.write(str(dst_path), data, sr, format='FLAC')
    elif output_format == "mp3":
        # Use ffmpeg for MP3 encoding at 320kbps
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_path), "-b:a", "320k", "-q:a", "0", str(dst_path)],
                capture_output=True, check=True
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install ffmpeg to export MP3 files.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg error: {e.stderr.decode(errors='replace')}")
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    return dst_path


def save_result_to_output(temp_wav_path, output_dir, output_format, metadata_text=None):
    """Copy a temp WAV to output directory in the chosen format, with embedded metadata.

    Args:
        temp_wav_path: Path to the WAV file in temp
        output_dir: Path to output directory
        output_format: 'wav', 'flac', or 'mp3'
        metadata_text: Optional string to embed in the audio file

    Returns:
        Path to the saved output file
    """
    temp_wav_path = Path(temp_wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = temp_wav_path.stem
    ext = output_format if output_format != "wav" else "wav"
    output_path = output_dir / f"{stem}.{ext}"

    convert_audio_format(temp_wav_path, output_path, output_format)

    # Embed metadata directly in the audio file
    if metadata_text:
        embed_metadata(output_path, metadata_text)

    return output_path


def embed_metadata(audio_path, metadata_text):
    """Embed metadata text into an audio file (WAV, FLAC, or MP3).

    Uses mutagen to write metadata as a comment/description tag.
    The metadata is stored in:
    - WAV: ID3 COMM (comment) tag
    - FLAC: Vorbis DESCRIPTION comment
    - MP3: ID3 COMM (comment) tag

    Args:
        audio_path: Path to the audio file
        metadata_text: String to embed
    """
    audio_path = Path(audio_path)
    ext = audio_path.suffix.lower()

    try:
        if ext == ".wav":
            from mutagen.wave import WAVE
            from mutagen.id3 import COMM

            audio = WAVE(str(audio_path))
            if audio.tags is None:
                audio.add_tags()
            audio.tags.add(COMM(encoding=3, lang="eng", desc="metadata", text=metadata_text))
            audio.save()

        elif ext == ".flac":
            from mutagen.flac import FLAC

            audio = FLAC(str(audio_path))
            audio["DESCRIPTION"] = metadata_text
            audio.save()

        elif ext == ".mp3":
            from mutagen.mp3 import MP3
            from mutagen.id3 import COMM, ID3NoHeaderError

            try:
                audio = MP3(str(audio_path))
            except ID3NoHeaderError:
                from mutagen.id3 import ID3
                audio = MP3(str(audio_path))
                audio.add_tags()

            if audio.tags is None:
                audio.add_tags()
            audio.tags.add(COMM(encoding=3, lang="eng", desc="metadata", text=metadata_text))
            audio.save()

    except Exception as e:
        # Fallback: write companion .txt if embedding fails
        print(f"[WARN] Could not embed metadata in {audio_path.name}: {e}")
        meta_path = audio_path.with_suffix(".txt")
        meta_path.write_text(metadata_text, encoding="utf-8")


def read_embedded_metadata(audio_path):
    """Read embedded metadata from an audio file (WAV, FLAC, or MP3).

    Falls back to reading a companion .txt file if no embedded metadata found.

    Args:
        audio_path: Path to the audio file

    Returns:
        Metadata string, or None if not found
    """
    audio_path = Path(audio_path)
    ext = audio_path.suffix.lower()
    embedded = None

    try:
        if ext == ".wav":
            from mutagen.wave import WAVE
            audio = WAVE(str(audio_path))
            if audio.tags:
                for key in audio.tags:
                    if key.startswith("COMM"):
                        embedded = str(audio.tags[key])
                        break

        elif ext == ".flac":
            from mutagen.flac import FLAC
            audio = FLAC(str(audio_path))
            desc = audio.get("DESCRIPTION")
            if desc:
                embedded = desc[0] if isinstance(desc, list) else str(desc)

        elif ext == ".mp3":
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_path))
            if audio.tags:
                for key in audio.tags:
                    if key.startswith("COMM"):
                        embedded = str(audio.tags[key])
                        break

    except Exception:
        pass

    if embedded:
        return embedded

    # Fallback: read companion .txt file (backward compatibility)
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return None


# ============================================================
# SMART NAMING UTILITIES
# ============================================================

def make_stem_from_text(text, sample_name=None, max_words=8):
    """Create a filename stem from the sample name and first N words of text.

    Produces '<sample>_<first_words>' so the same text generated with
    different voice samples won't collide.

    Args:
        text: Source text (prompt / sentence)
        sample_name: Voice sample name to prefix (optional)
        max_words: How many words to keep (default 8)

    Returns:
        A safe filename stem like 'deep_voice_a_beautiful_night_in_barcelona'
    """
    words = text.strip().split()[:max_words]
    raw = "_".join(words)
    # Keep only alphanumeric, hyphens, underscores
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)
    safe = re.sub(r'_+', '_', safe).strip('_').lower()
    text_part = safe or "clip"

    if sample_name:
        prefix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sample_name)
        prefix = re.sub(r'_+', '_', prefix).strip('_').lower()
        if prefix:
            return f"{prefix}_{text_part}"

    return text_part


def resolve_output_stem(base_stem, output_dir, clip_count=1):
    """Find a non-colliding stem for saving files to output_dir.

    For clip_count == 1 (single file):
        Returns base_stem if it doesn't exist, else base_stem_1, base_stem_2, ...

    For clip_count > 1 (split clips numbered _01, _02, ...):
        Returns base_stem if base_stem_01 doesn't exist,
        else base_stem_1 if base_stem_1_01 doesn't exist, etc.

    Args:
        base_stem: Base filename stem (no extension)
        output_dir: Path to output directory
        clip_count: Number of clips to save (1 = single file)

    Returns:
        A stem string guaranteed not to collide with existing files.
    """
    output_dir = Path(output_dir)
    exts = {'.wav', '.flac', '.mp3'}
    existing = set()
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                existing.add(f.stem.lower())

    low = base_stem.lower()

    if clip_count <= 1:
        # Single-file collision check
        if low not in existing:
            return base_stem
        n = 1
        while f"{low}_{n}" in existing:
            n += 1
        return f"{base_stem}_{n}"
    else:
        # Multi-clip collision check: test whether _01 already exists
        if f"{low}_01" not in existing:
            return base_stem
        n = 1
        while f"{low}_{n}_01" in existing:
            n += 1
        return f"{base_stem}_{n}"
