@echo off
echo ========================================
echo Voice Clone Studio - Setup Script
echo ========================================
echo.
echo Select CUDA version for PyTorch (press number or wait 10 seconds for default):
echo   1. CUDA 13.0 (latest, for newest GPUs - DEFAULT)
echo   2. CUDA 12.8 (for newer GPUs)
echo   3. CUDA 12.1 (for older GPUs, GTX 10-series and newer)
echo.
choice /C 123 /T 10 /D 1 /M "Enter choice"
set CUDA_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install LuxTTS voice cloning engine?
echo LuxTTS provides fast, high-quality voice cloning at 48kHz.
echo Requires ~1GB disk space for model files.
echo ========================================
echo.
echo   1. Yes - Install LuxTTS support
echo   2. No  - Skip (DEFAULT)
echo.
choice /C 12 /T 15 /D 2 /M "Install LuxTTS?"
set LUXTTS_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install Qwen3 ASR speech recognition?
echo Qwen3 ASR provides high-quality multilingual speech recognition.
echo As well as automatic splitting of clips under 5 minutes.
echo Supports 52 languages with Small (0.6B) and Large (1.7B) models.
echo Note: This will update transformers to 4.57.6+
echo ========================================
echo.
echo   1. Yes - Install Qwen3 ASR support
echo   2. No  - Skip (DEFAULT)
echo.
choice /C 12 /T 15 /D 2 /M "Install Qwen3 ASR?"
set QWEN3ASR_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install Whisper speech recognition?
echo OpenAI Whisper is an alternative transcription engine.
echo Supports automatic splitting of clips with no length limit.
echo ========================================
echo.
echo   1. Yes - Install Whisper
echo   2. No  - Skip (DEFAULT)
echo.
choice /C 12 /T 15 /D 2 /M "Install Whisper?"
set WHISPER_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install llama.cpp for LLM prompt generation?
echo llama.cpp powers the Prompt Manager's local LLM feature.
echo Lets you generate TTS and SFX prompts using Qwen3 models.
echo ========================================
echo.
echo   1. Yes - Install llama.cpp
echo   2. No  - Skip (DEFAULT)
echo.
choice /C 12 /T 15 /D 2 /M "Install llama.cpp?"
set LLAMA_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install Flash Attention 2 for faster inference?
echo Flash Attention 2 provides fast attention for supported models.
echo Requires CUDA GPU. Cannot be used with multilingual Chatterbox.
echo ========================================
echo.
echo   1. Yes - Install Flash Attention 2
echo   2. No  - Skip (DEFAULT)
echo.
choice /C 12 /T 15 /D 2 /M "Install Flash Attention 2?"
set FLASH_CHOICE=%errorlevel%
echo.
echo All questions answered - installing now...
echo.

REM Check Python version - find a compatible version (3.10-3.12)
echo [1/7] Checking Python installation...
set PYTHON_CMD=

REM First try the py launcher to find a compatible version
where py >nul 2>&1
if %errorlevel% equ 0 (
    REM Try 3.12 first, then 3.11, then 3.10
    for %%V in (3.12 3.11 3.10) do (
        if not defined PYTHON_CMD (
            py -%%V --version >nul 2>&1
            if not errorlevel 1 (
                set PYTHON_CMD=py -%%V
            )
        )
    )
)

REM If py launcher didn't find one, try bare python
if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python not found! Please install Python 3.10-3.12.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Validate the version we found
for /f "tokens=2" %%a in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%a
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if not "%PYMAJOR%"=="3" (
    echo ERROR: Python 3.10-3.12 is required. Detected: %PYVER%
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
if %PYMINOR% LSS 10 (
    echo ERROR: Python 3.10-3.12 is required. Detected: Python %PYVER%
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
if %PYMINOR% GTR 12 (
    echo ERROR: Python 3.13+ is not supported due to dependency conflicts.
    echo Python 3.10, 3.11, or 3.12 was not found on your system.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Using: %PYTHON_CMD% (Python %PYVER%)
echo.

REM Install media processing tools
echo [2/7] Installing media processing tools...
echo Installing SOX...
winget install -e --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo WARNING: SOX installation may have failed.
    echo You can also install manually from: https://sourceforge.net/projects/sox/files/sox/
    echo Or using Chocolatey: choco install sox
)
echo.
echo Installing ffmpeg...
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg installation may have failed.
    echo You can also install manually from: https://ffmpeg.org/download.html
    echo Or using Chocolatey: choco install ffmpeg
)
echo.

REM Create virtual environment
echo [3/7] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    %PYTHON_CMD% -m venv venv
    if not exist venv (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [4/7] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Update pip
echo Updating pip...
python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to update pip!
    pause
    exit /b 1
)

REM Install PyTorch
echo [5/7] Installing PyTorch...
setlocal enabledelayedexpansion

if "%CUDA_CHOICE%"=="1" set CUDA_VER=cu130
if "%CUDA_CHOICE%"=="2" set CUDA_VER=cu128
if "%CUDA_CHOICE%"=="3" set CUDA_VER=cu121

if defined CUDA_VER (
    echo Installing PyTorch with !CUDA_VER!...
    echo This may take several minutes...
    pip install torch==2.9.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/!CUDA_VER!
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install PyTorch!
        pause
        exit /b 1
    )
) else (
    echo Invalid CUDA choice! Please run setup again.
    pause
    exit /b 1
)
endlocal
echo.

REM Install requirements
echo [6/7] Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements!
    pause
    exit /b 1
)

REM Faster Qwen3 TTS (CUDA graph acceleration, lightweight - auto-install)
echo Installing Faster Qwen3 TTS (CUDA graph acceleration)...
pip install faster-qwen3-tts
if %errorlevel% neq 0 (
    echo WARNING: Faster Qwen3 TTS installation failed. Standard engine will be used.
)
echo.

REM Optional modules
echo [7/7] Optional modules...
if not "%LUXTTS_CHOICE%"=="1" goto :skip_luxtts

echo.
echo Installing LuxTTS prerequisites...
echo [Step 1/3] Installing LinaCodec...
pip install git+https://github.com/ysharma3501/LinaCodec.git
if %errorlevel% neq 0 (
    echo WARNING: LinaCodec installation failed. LuxTTS will not be available.
    goto :skip_luxtts
)
echo [Step 2/3] Installing piper-phonemize...
pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
if %errorlevel% neq 0 (
    echo WARNING: piper-phonemize installation failed. LuxTTS will not be available.
    goto :skip_luxtts
)
echo [Step 3/3] Installing zipvoice (LuxTTS)...
pip install "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"
if %errorlevel% neq 0 (
    echo WARNING: zipvoice installation failed. LuxTTS will not be available.
    goto :skip_luxtts
)
echo LuxTTS installed successfully!
goto :luxtts_done

:skip_luxtts
echo Skipping LuxTTS installation.
:luxtts_done
echo.

REM Qwen3 ASR (installed last as it updates transformers)
if not "%QWEN3ASR_CHOICE%"=="1" goto :skip_qwen3asr

echo.
echo Installing Qwen3 ASR...
pip install -U qwen-asr
if %errorlevel% neq 0 (
    echo WARNING: Qwen3 ASR installation failed.
    goto :skip_qwen3asr
)
echo Qwen3 ASR installed successfully!
goto :qwen3asr_done

:skip_qwen3asr
echo Skipping Qwen3 ASR installation.
:qwen3asr_done
echo.

REM Whisper
if not "%WHISPER_CHOICE%"=="1" goto :skip_whisper

echo.
echo Installing Whisper...
pip install openai-whisper
if %errorlevel% neq 0 (
    echo WARNING: Whisper installation failed.
    goto :skip_whisper
)
echo Whisper installed successfully!
goto :whisper_done

:skip_whisper
echo Skipping Whisper installation.
:whisper_done
echo.

REM llama.cpp
if not "%LLAMA_CHOICE%"=="1" goto :skip_llama

echo.
echo Installing llama.cpp...
winget install llama.cpp
if %errorlevel% neq 0 (
    echo WARNING: llama.cpp installation may have failed.
    echo You can install manually from: https://github.com/ggml-org/llama.cpp
    goto :skip_llama
)
echo llama.cpp installed successfully!
goto :llama_done

:skip_llama
echo Skipping llama.cpp installation.
:llama_done
echo.

REM Flash Attention 2
if not "%FLASH_CHOICE%"=="1" goto :skip_flash

echo.
echo Installing Flash Attention 2...
setlocal enabledelayedexpansion

REM Flash Attention needs Python-version-specific wheels (cp310, cp311, cp312)
set FLASH_PY=cp3%PYMINOR%

if "%CUDA_CHOICE%"=="1" (
    set FLASH_WHL=flash_attn-2.8.3+torch2.9.1.cuda13.1-!FLASH_PY!-!FLASH_PY!-win_amd64.whl
) else (
    set FLASH_WHL=flash_attn-2.8.2-!FLASH_PY!-!FLASH_PY!-win_amd64.whl
)
set FLASH_URL=https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/!FLASH_WHL!
echo Downloading: !FLASH_WHL!
pip install "!FLASH_URL!"
if !errorlevel! neq 0 (
    echo WARNING: Flash Attention 2 installation failed.
    echo Wheel may not exist for Python 3.%PYMINOR% with your CUDA version.
    echo You can browse available wheels at: https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main
) else (
    echo Flash Attention 2 installed successfully!
)
endlocal
goto :flash_done

:skip_flash
echo Skipping Flash Attention 2 installation.
:flash_done
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Python version being used:
python --version
echo.
echo To launch Voice Clone Studio:
echo   1. Make sure virtual environment is activated: venv\Scripts\activate
echo   2. Run: python Voice_Clone_Studio.py
echo   3. Or use: launch.bat
echo.
pause
