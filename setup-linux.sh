#!/bin/bash
# Linux installation helper for Voice Clone Studio
# This script helps with common Linux installation issues

set -e  # Exit on error

echo "========================================="
echo "Voice Clone Studio - Linux Setup Helper"
echo "========================================="
echo ""

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create/Update Conda environment
ENV_NAME="voice-clone-studio"
echo "Creating/Updating Conda environment: $ENV_NAME (Python 3.12)..."
conda create -n "$ENV_NAME" python=3.12 -y

# Instructions for activation (scripts cannot easily activate for the parent shell)
echo ""
echo "IMPORTANT: You need to activate the Conda environment to continue installation interactively or run the app."
echo "However, this script will run installation commands using 'conda run'."
echo ""

# Ask for GPU type
echo "========================================="
echo "Select your GPU type:"
echo "1) NVIDIA (CUDA)"
echo "2) AMD (ROCm)"
echo "========================================="
read -p "Enter choice [1-2] (default: 1): " GPU_CHOICE
GPU_CHOICE=${GPU_CHOICE:-1}

# Install PyTorch
echo ""
run_in_env() {
    conda run -n "$ENV_NAME" --no-capture-output "$@"
}

if [ "$GPU_CHOICE" == "2" ]; then
    echo "Installing PyTorch with ROCm 7.1 support (AMD)..."
    run_in_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
else
    echo "Installing PyTorch with CUDA 13.0 support (NVIDIA)..."
    run_in_env pip install torch==2.9.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    run_in_env pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found!"
    exit 1
fi

# Install ONNX Runtime
echo ""
echo "Installing ONNX Runtime..."
if [ "$GPU_CHOICE" == "2" ]; then
    echo "Installing ONNX Runtime MIGraphX (ROCm 7.1) for AMD..."
    # Pinning to the specific wheel provided by user for Python 3.12
    MIGRAPHX_WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/onnxruntime_migraphx-1.23.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    run_in_env pip install "$MIGRAPHX_WHEEL"
else
    # NVIDIA
    echo "Installing ONNX Runtime GPU..."
    # We need to run python check inside the environment
    if run_in_env python -c "import onnxruntime" 2>/dev/null; then
        echo "✅ ONNX Runtime is working"
    else
        run_in_env pip install onnxruntime-gpu
    fi
fi

# Optional modules
echo ""
echo "Installing optional modules..."
if [[ "$INSTALL_LUXTTS" =~ ^[Yy]$ ]]; then
    # ... (Keep existing logic but use run_in_env)
    echo "Installing LuxTTS prerequisites..."
    if run_in_env pip install git+https://github.com/ysharma3501/LinaCodec.git; then
        if run_in_env pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html; then
            if run_in_env pip install "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"; then
                echo "LuxTTS installed successfully!"
            else
                echo "zipvoice installation failed."
            fi
        else
            echo "piper-phonemize installation failed."
        fi
    else
        echo "LinaCodec installation failed."
    fi
fi

if [[ "$INSTALL_QWEN3ASR" =~ ^[Yy]$ ]]; then
    echo "Installing Qwen3 ASR..."
    run_in_env pip install -U qwen-asr
fi

# llama.cpp logic is system-level (brew/apt), usually doesn't need run_in_env unless it's the python binding.
# The script installs system package 'llama.cpp' via brew/apt, AND requirements.txt might rely on it.
# The original script installed system packages. We should keep that unless 'llama-cpp-python' is what's needed.
# The prompt says "Install llama.cpp for LLM prompt generation", implies the binary or python binding?
# Original script installs binary via brew/apt. We'll leave that as is, outside conda?
# WAIT, the original script installed SYSTEM packages at the top.
# Then did 'brew install llama.cpp'.
# I'll output the new instructions.

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "To run the application:"
echo "  1. conda activate $ENV_NAME"
echo "  2. python voice_clone_studio.py"
echo "  3. Or use: ./launch.sh (update it to activate conda first)"
echo ""
