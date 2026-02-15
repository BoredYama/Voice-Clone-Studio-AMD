#!/bin/bash
# Linux launcher for Voice Clone Studio

echo "========================================"
echo "Voice Clone Studio"
echo "========================================"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    # Use conda run to execute in the environment
    echo "Using Conda environment: voice-clone-studio"
    conda run -n voice-clone-studio --no-capture-output python voice_clone_studio.py
else
    # Fallback or error
    echo "Conda not found. Please activate your environment manually or install Conda."
    echo "Trying to run python directly..."
    python voice_clone_studio.py
fi
