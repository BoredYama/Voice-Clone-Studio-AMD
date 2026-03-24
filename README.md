## WIP

# Voice Clone Studio (AMD ROCm Fork)

**⚠️ LINUX ONLY ⚠️**
This is a fork of Voice Clone Studio optimized for **AMD GPUs** on **Linux** using **ROCm 7.1**. Windows and macOS are **NOT** supported in this version.

A multi-model, modular Gradio-based web UI for voice cloning, voice design, multi-speaker conversation, voice conversion, voice training and sound effects. Basically, One app, many engines, to tinker with all of them without juggling separate repos or setups.  Powered by [VibeVoice](https://github.com/microsoft/VibeVoice), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [LuxTTS](https://github.com/ysharma3501/LuxTTS), [Chatterbox](https://github.com/resemble-ai/chatterbox) and [MMAudio](https://github.com/hkchengrex/MMAudio). Supports [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [VibeVoice-ASR](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) and [Whisper](https://github.com/openai/whisper) for automatic transcription. As well as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Ollama](https://ollama.com/) for Prompt Generation and a Prompt Saving, based on [ComfyUI Prompt-Manager](https://github.com/FranckyB/ComfyUI-Prompt-Manager)

<img src="https://img.shields.io/badge/VibeVoice-TTS-green" alt="VibeVoice TTS"> <img src="https://img.shields.io/badge/VibeVoice-ASR-green" alt="VibeVoice ASR"> <img src="https://img.shields.io/badge/Qwen3-TTS-blue" alt="Qwen3-TTS"> <img src="https://img.shields.io/badge/Qwen3-ASR-blue" alt="Qwen3-ASR"> <img src="https://img.shields.io/badge/LuxTTS-TTS-orange" alt="LuxTTS"> <img src="https://img.shields.io/badge/Chatterbox-TTS-red" alt="Chatterbox-TTS"> <img src="https://img.shields.io/badge/Whisper-yellow" alt="Whisper"> <img src="https://img.shields.io/badge/MMAudio-SFX-purple" alt="MMAudio">

<a href="docs/preview.png"><img src="docs/preview.png" alt="Voice Clone Studio Preview" width="600"></a>

## Architecture

Voice Clone Studio is fully modular. The main file dynamically loads self-contained tools as tabs. Each tool can be enabled or disabled from Settings without touching any code. It supports multipe engine for voice cloning, as well as Model finetuning. More features are also planned.

## Features

### Voice Clone

Clone voices from your own audio samples. Provide a short reference audio clip with its transcript, and generate new speech in that voice.

- **Multiple engines** - Qwen3-TTS (0.6B/1.7B) or VibeVoice (1.5B/Large/Large-4bit)
- **Voice prompt caching** - First generation processes the sample, subsequent ones are instant
- **Seed control** - Reproducible results with saved seeds
- **Emotion presets** - 40+ emotion presets with adjustable intensity
- **Split by Paragraph** - Generate a separate audio clip for each paragraph, with automatic naming and a combined preview
- **Prompt Hub** - Access saved prompts directly from the tool without switching tabs
- **Metadata tracking** - Each output saves generation info (sample, seed, text)

### Conversation

Create multi-speaker dialogues using either Qwen's premium voices or your own custom voice samples using VibeVoice:

**Choose Your Engine:**

- **Qwen** - Fast generation with 9 preset voices, optimized for their native languages
- **VibeVoice** - High-quality custom voices, up to 90 minutes continuous, perfect for podcasts/audiobooks
- **LuxTTS** -

**Unified Script Format:**
Write scripts using `[N]:` format - works seamlessly with both engines:

```
[1]: Hey, how's it going?
[2]: I'm doing great, thanks for asking!
[3]: Mind if I join this conversation?
```

**Qwen Mode:**

- Mix any of the 9 premium speakers
- Adjustable pause duration between lines
- Fast generation with cached prompts

**Speaker Mapping:**

- [1] = Vivian, [2] = Serena, [3] = Uncle_Fu, [4] = Dylan, [5] = Eric
- [6] = Ryan, [7] = Aiden, [8] = Ono_Anna, [9] = Sohee

**VibeVoice Mode:**

- **Up to 90 minutes** of continuous speech
- **Up to 4 distinct speakers** using your own voice samples
- Cross-lingual support
- May spontaneously add background music/sounds for realism
- Numbers beyond 4 wrap around (5→1, 6→2, 7→3, 8→4, etc.)

Perfect for:

- Podcasts
- Audiobooks
- Long-form conversations
- Multi-speaker narratives

**Models:**

- **Small** - Faster generation (Qwen: 0.6B, VibeVoice: 1.5B)
- **Large** - Best quality (Qwen: 1.7B, VibeVoice: Large model)

### Voice Changer

Change the voice in any audio using Chatterbox speech-to-speech voice conversion (Resemble AI, MIT license):

- **Speech-to-Speech** - Upload or record audio, select a target voice sample, and re-speak the content in the target voice
- **Microphone support** - Record directly from your microphone for real-time voice conversion
- **Any voice sample** - Use the same voice samples from Voice Clone as conversion targets
- **English optimized** - Best results with English speech; multilingual support available with the Multilingual model
- **Multiple models** - TTS (English), Multilingual (23 languages)

### Voice Presets

Generate with premium pre-built voices, trained models, and streaming speakers:

**VibeVoice Trained:**
- Generate with your own VibeVoice LoRA-trained voices
- Optional voice sample conditioning with adjustable LoRA strength
- Advanced params: CFG scale, DDPM steps, temperature, top_k, top_p, repetition penalty

**Qwen Speakers:**
- 9 premium Qwen3-TTS speakers with style instructions (emotion, tone, speed)
- Each speaker works best in native language but supports all

**VibeVoice Speakers:**
- 7 built-in VibeVoice Streaming 0.5B preset voices (Carter, Davis, Emma, Frank, Grace, Mike, Samuel)
- Lightweight 0.5B model with fast generation
- Auto-downloads voice prompt files from GitHub, caches locally

**Qwen Trained:**
- Generate with Qwen3-TTS finetuned models
- ICL (In-Context Learning) mode for enhanced voice cloning

### Voice Design

Create voices from natural language descriptions - no audio needed, using Qwen3-TTS Voice Design Model:

- Describe age, gender, emotion, accent, speaking style
- Generate unique voices matching your description

### Train Custom Voices

Fine-tune your own custom voice models with your training data:

- **Dual Engine** - Train with Qwen3-TTS or VibeVoice LoRA finetuning
- **Dataset Management** - Organize training samples in the `datasets/` folder
- **Audio Preparation** - Auto-converts to 24kHz 16-bit mono format
- **Training Pipeline** - Complete 3-step workflow (validation → extract codes → train)
- **Epoch Selection** - Compare different training checkpoints
- **Live Progress** - Real-time training logs and loss monitoring
- **Stop Training** - Terminate training mid-run with clean status
- **Voice Presets Integration** - Use trained models alongside premium speakers

**VibeVoice Training Features:**
- LoRA finetuning via subprocess with full parameter UI
- Configurable: batch size, learning rate, epochs, save interval, DDPM batch multiplier, diffusion/CE loss weights, voice prompt drop, gradient accumulation, warmup steps
- Train diffusion head toggle, EMA on/off with auto-decay calculation
- Verbose output filtering — clean per-epoch summaries instead of raw logs

**Requirements:**

- CUDA GPU required
- Multiple audio samples with transcripts
- Training time: ~10-30 minutes depending on dataset size

**Workflow:**

1. Prepare audio files (WAV/MP3) and organize in `datasets/YourSpeakerName/` folder
2. Use **Batch Transcribe** to automatically transcribe all files at once
3. Review and edit individual transcripts as needed
4. Configure training parameters (model size, epochs, learning rate)
5. Monitor training progress in real-time
6. Use trained model in Voice Presets tab

### Prep Audio

Unified audio preparation workspace for both voice samples and training datasets:

- **Trim** - Use waveform selection to cut audio
- **Normalize** - Balance audio levels
- **Convert to Mono** - Ensure single-channel audio
- **Denoise** - Clean audio with DeepFilterNet
- **Extract from Video** - Automatically extract audio tracks from video files
- **Auto-Split Audio** - Split long recordings into sentence-level clips using Qwen3-ASR-detected boundaries
- **Transcribe** - Qwen3 ASR, Whisper, or VibeVoice ASR automatic transcription
- **Batch Transcribe** - Process entire folders of audio files at once
- **Save as Sample** - One-click sample creation
- **Dataset Management** - Create, delete, and organize dataset folders directly from the UI

### Sound Effects

Generate sound effects and ambient audio using MMAudio (CVPR 2025, MIT license):

- **Text-to-Audio** - Describe any sound and generate high-quality 44.1kHz audio
- **Video-to-Audio** - Drop in a video clip and generate synchronized sound effects
- **Multiple Models** - Medium (2.4GB) and Large v2 (3.9GB) built-in, plus custom model support
- **Custom Models** - Load your own `.pth` or `.safetensors` checkpoints with automatic architecture detection
- **Video Preview** - Source/Result toggle to compare original video against the audio-muxed result
- **Fine Controls** - Adjustable duration, guidance strength, and negative prompts

### Prompt Manager

Save, browse, and generate text prompts for your TTS sessions. Includes a built-in LLM generator powered by [llama.cpp](https://github.com/ggml-org/llama.cpp) or [Ollama](https://ollama.com/):

- **Saved Prompts** - Store and organize prompts in a local `prompts.json` file, browse with the file lister
- **Prompt Hub** - Every generation tool has a built-in Prompt Loader for one-click access to saved prompts without switching tabs
- **LLM Generation** - Generate prompts locally using Qwen3 language models via llama.cpp or Ollama (no cloud API needed)
- **Ollama Support** - Use any model from your local Ollama installation as an alternative to llama.cpp
- **System Prompt Presets** - Built-in presets for TTS/Voice and Sound Design/SFX workflows, or write your own
- **Model Auto-Download** - Download Qwen3-4B (~4.8GB) or Qwen3-8B (~8.5GB) GGUF models directly from HuggingFace
- **Custom Models** - Drop any `.gguf` file into `models/llama/` to use your own models
- **Automatic Server Management** - llama.cpp server starts/stops automatically, cleaned up on exit or Clear VRAM

Inspired by [ComfyUI-Prompt-Manager](https://github.com/FranckyB/ComfyUI-Prompt-Manager) by FranckyB.

### Output History

View, play back, and manage your previously generated audio files. Multi-select for batch deletion, double-click to play.

### Settings

Centralized application configuration:

- **Model loading** - Attention mechanism, offline mode, low CPU memory usage
- **CUDA Graphs Acceleration** - 5-10x faster Qwen3 inference via [Faster-Qwen3-TTS](https://github.com/andimarafioti/faster-qwen3-tts) (CUDA only, toggle in Settings)
- **Multi-GPU Assignment** - Assign TTS, ASR, and Llama.cpp to different GPUs on multi-GPU systems
- **LLM Backend** - Choose between llama.cpp and Ollama for prompt generation
- **Folder paths** - Configurable directories for samples, output, datasets, models
- **Model downloads** - Download models directly to local storage
- **Visible Tools** - Enable or disable any tool tab (restart to apply)
- **Help Guide** - Built-in documentation for all tools

---

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda) - Required for environment management (Python 3.12 will be installed; note that DeepFilterNet recommends 3.11 but we enforce 3.12)
- **Linux:** CUDA-compatible GPU (NVIDIA) or ROCm-compatible GPU (AMD, 8GB+ VRAM recommended)
- **SOX** (Sound eXchange) - Required for audio processing
- **FFMPEG** - Multimedia framework required for audio format conversion
- **llama.cpp** (optional) - Required only for the Prompt Manager's LLM generation feature. See [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Ollama** (optional) - Alternative LLM backend for prompt generation. See [Ollama](https://ollama.com/)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional, CUDA only)

**Note:** `openai-whisper` is skipped due to compatibility issues on Linux. Use VibeVoice ASR or Qwen3 ASR for transcription instead.

### Setup

1. Clone the repository:

```bash
git clone https://github.com/BoredYama/Voice-Clone-Studio-AMD.git
cd Voice-Clone-Studio
```

2. Make the setup script executable and run it:

```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

This will automatically:

- Check for Conda (Miniconda/Anaconda)
- Create a Conda environment `voice-clone-studio` with Python 3.12
- Ask for GPU type (NVIDIA/AMD)
- Install PyTorch with CUDA (NVIDIA) or ROCm 7.1 (AMD) support
- Install dependencies and specific ONNX Runtime versions
- Warn about Whisper compatibility if needed

3. Activate the environment:

```bash
conda activate voice-clone-studio
```

8. (Optional) Install FlashAttention 2 for faster generation (CUDA only):
   **Note:** The application automatically detects and uses the best available attention mechanism. Configure in Settings tab: `flash_attention_2` (CUDA only) → `sdpa` (CUDA/MPS) → `eager` (all devices)

## Troubleshooting

For troubleshooting solutions, see [docs/troubleshooting.md](docs/troubleshooting.md).

### Running Tests (Docker)

To verify the installation and features (like the DeepFilterNet denoiser), runs the integration tests inside the container:

```powershell
# Run the Denoiser Integration Test
docker-compose exec voice-clone-studio python tests/integration_test_denoiser.py
```

## Usage

### Launch the UI

```bash
conda activate voice-clone-studio
python voice_clone_studio.py
```

Or use the launcher script (ensure it activates conda first):

```bash
./launch.sh
```

The UI will open at `http://127.0.0.1:7860`

### Prepare Voice Samples

1. Go to the **Prep Audio** tab
2. Upload or record audio (3-10 seconds of clear speech)
3. Trim, normalize, and denoise as needed
4. Transcribe or manually enter the text
5. Save as a sample with a name

### Clone a Voice

1. Go to the **Voice Clone** tab
2. Select your sample from the dropdown
3. Enter the text you want to speak
4. Click Generate

### Design a Voice

1. Go to the **Voice Design** tab
2. Enter the text to speak
3. Describe the voice (e.g., "Young female, warm and friendly, slight British accent")
4. Click Generate

## Project Structure

```
Voice-Clone-Studio/
├── voice_clone_studio.py          # Main orchestrator (~230 lines)
├── config.json                    # User preferences & enabled tools
├── requirements.txt               # Python dependencies
├── launch.bat / launch.sh         # Launcher scripts
├── setup-windows.bat / setup-linux.sh / setup-mac.sh  # Platform setup scripts
├── wheel/                         # Pre-built custom Gradio components
│   └── gradio_filelister-0.4.0-py3-none-any.whl
├── samples/                       # Voice samples (.wav + .json)
├── output/                        # Generated audio outputs
├── datasets/                      # Training datasets
├── models/                        # Downloaded & trained models
├── docs/                          # Documentation
│   ├── updates.md                 # Version history
│   ├── troubleshooting.md         # Troubleshooting guide
│   └── MODEL_MANAGEMENT_README.md # AI model manager docs
└── modules/
    ├── core_components/           # Core app code
    │   ├── tools/                 # All UI tools (tabs)
    │   │   ├── voice_clone.py
    │   │   ├── voice_presets.py
    │   │   ├── conversation.py
    │   │   ├── voice_design.py
    │   │   ├── voice_changer.py
    │   │   ├── sound_effects.py
    │   │   ├── prep_audio.py
    │   │   ├── output_history.py
    │   │   ├── train_model.py
    │   │   └── settings.py
    │   ├── ai_models/             # TTS & ASR model managers
    │   ├── ui_components/         # Modals, theme
    │   ├── gradio_filelister/     # Custom file browser component
    │   ├── constants.py           # Central constants
    │   ├── emotion_manager.py     # Emotion presets
    │   ├── audio_utils.py         # Audio processing
    │   └── help_page.py           # Help content
    ├── deepfilternet/             # Audio denoising
    ├── qwen_finetune/             # Training scripts
    ├── chatterbox/                # Chatterbox voice conversion
    ├── vibevoice_tts/             # VibeVoice TTS
    └── vibevoice_asr/             # VibeVoice ASR
```

## Models Used

Each tab lets you choose between model sizes:

| Model                     | Sizes             | Use Case                                               |
| ------------------------- | ----------------- | ------------------------------------------------------ |
| **Qwen3-TTS Base**        | Small, Large      | Voice cloning from samples                             |
| **Qwen3-TTS CustomVoice** | Small, Large      | Premium speakers with style control                    |
| **Qwen3-TTS VoiceDesign** | 1.7B only         | Voice design from descriptions                         |
| **LuxTTS**                | Large             | Voice cloning with speaker encoder                     |
| **VibeVoice-TTS**         | Small, Large      | Voice cloning & Long-form multi-speaker (up to 90 min) |
| **Chatterbox**            | TTS, Multilingual | Speech-to-speech voice conversion                      |
| **VibeVoice-ASR**         | Large             | Audio transcription                                    |
| **Whisper**               | Medium            | Audio transcription                                    |
| **MMAudio**               | Medium, Large v2  | Sound effects generation (text & video to audio)       |

- **Small** = Faster, less VRAM (Qwen: 0.6B ~4GB, VibeVoice: 1.5B)
- **Large** = Better quality, more expressive (Qwen: 1.7B ~8GB, VibeVoice: Large model)
- **4 Bit Quantized** version of the Large model is also included for VibeVoice.

Models are automatically downloaded on first use via HuggingFace.

## Tips

- **Reference Audio**: Use clear, noise-free recordings (3-10 seconds)
- **Transcripts**: Should exactly match what's spoken in the audio
- **Caching**: Voice prompts are cached - first generation is slow, subsequent ones are fast
- **Seeds**: Use the same seed to reproduce identical outputs

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

This project is based on and uses code from:

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** - Apache 2.0 License (Alibaba)
- **[VibeVoice](https://github.com/microsoft/VibeVoice)** - MIT License
- **[LuxTTS](https://github.com/ysharma3501/LuxTTS)** - Apache 2.0 License
- **[Gradio](https://gradio.app/)** - Apache 2.0 License
- **[MMAudio](https://github.com/hkchengrex/MMAudio)** - MIT License
- **[Chatterbox](https://github.com/resemble-ai/chatterbox)** - MIT License (Resemble AI)
- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** - MIT License

## Updates

For detailed version history and release notes, see [docs/updates.md](docs/updates.md).
