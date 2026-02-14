"""
Settings Tab

Configure global application settings.

Standalone testing:
    python -m modules.core_components.tools.settings
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.help_page import (
    show_voice_clone_help, show_conversation_help, show_voice_presets_help,
    show_voice_design_help, show_prep_audio_help, show_dataset_help,
    show_train_help, show_tips_help
)

# Tools that can be toggled (everything except Settings)
# Format: (config_key, display_label)
TOGGLEABLE_TOOLS = [
    ("Voice Clone", "Voice Clone"),
    ("Voice Changer", "Voice Changer"),
    ("Voice Presets", "Voice Presets"),
    ("Conversation", "Conversation"),
    ("Voice Design", "Voice Design"),
    ("Prep Samples", "Prep Samples"),
    ("Train Model", "Train Model"),
    ("Sound Effects", "Sound Effects"),
    ("Prompt Manager", "Prompt Manager"),
    ("Output History", "Output History"),
]


class SettingsTool(Tool):
    """Settings tool implementation."""

    config = ToolConfig(
        name="Settings",
        module_name="tool_settings",
        description="Application settings and preferences",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Settings tool UI."""
        components = {}

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        format_help_html = shared_state.get('format_help_html')

        with gr.TabItem("⚙️"):
            with gr.Tabs():
                with gr.TabItem("Settings"):
                    gr.Markdown("# ⚙️ Settings")
                    gr.Markdown("Configure global application settings")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Displayed Tools")

                                # Get current enabled_tools from config
                                tool_settings = _user_config.get("enabled_tools", {})

                                with gr.Column():
                                    for key, label in TOGGLEABLE_TOOLS:
                                        is_enabled = tool_settings.get(key, True)
                                        components[f'tool_toggle_{key}'] = gr.Checkbox(
                                            label=label,
                                            value=is_enabled,
                                            interactive=True
                                        )

                            with gr.Column():
                                gr.Markdown("### Available Voice Clone Engines")

                                # Get TTS_ENGINES from shared_state
                                TTS_ENGINES = shared_state.get('TTS_ENGINES', {})
                                engine_settings = _user_config.get("enabled_engines", {})

                                for engine_key, engine_info in TTS_ENGINES.items():
                                    is_enabled = engine_settings.get(
                                        engine_key, engine_info.get("default_enabled", True)
                                    )
                                    components[f'engine_toggle_{engine_key}'] = gr.Checkbox(
                                        label=engine_info["label"],
                                        value=is_enabled,
                                        interactive=True
                                    )

                                gr.Markdown("### Available Transcription Engines")
                                ASR_ENGINES = shared_state.get('ASR_ENGINES', {})
                                asr_settings = _user_config.get("enabled_asr_engines", {})
                                with gr.Column():
                                    for engine_key, engine_info in ASR_ENGINES.items():
                                        is_enabled = asr_settings.get(
                                            engine_key, engine_info.get("default_enabled", True)
                                        )
                                        components[f'asr_toggle_{engine_key}'] = gr.Checkbox(
                                            label=engine_info["label"],
                                            value=is_enabled,
                                            interactive=True
                                        )

                                    components['bypass_split_limit'] = gr.Checkbox(
                                        label="Allow Qwen3 extended audio splitting (beyond 5 min)",
                                        value=_user_config.get("bypass_split_limit", False),
                                        info="Qwen3 ASR alignment may be less accurate and \ndemand tons of VRAM for very long audio. Enable with caution.",
                                        interactive=True
                                    )
                            with gr.Column():
                                gr.Markdown("### llama.cpp (Prompt Manager)")

                                components['settings_llama_cpp_path'] = gr.Textbox(
                                    label="llama.cpp Location",
                                    value=_user_config.get("llama_cpp_path", ""),
                                    info="Path to the folder containing llama-server. Leave empty to use system PATH.",
                                    placeholder="e.g. C:\\llama.cpp\\build\\bin"
                                )
                                components['reset_llama_cpp_path_btn'] = gr.Button("Reset", size="sm")

                                components['settings_llama_models_path'] = gr.Textbox(
                                    label="Additional LLM Models Folder",
                                    value=_user_config.get("llama_models_path", ""),
                                    info="Extra folder to scan for .gguf models (in addition to models/llama/). Downloads go here if set.",
                                    placeholder="e.g. D:\\models\\gguf"
                                )
                                components['reset_llama_models_path_btn'] = gr.Button("Reset", size="sm")

                        with gr.Row():
                            with gr.Column():

                                gr.Markdown("### Model Optimizations")
                                components['settings_low_cpu_mem'] = gr.Checkbox(
                                    label="Low CPU Memory Usage (Slower loading time)",
                                    value=_user_config.get("low_cpu_mem_usage", False),
                                    info="Reduces CPU RAM usage when loading models."
                                )

                                components['settings_attention_mechanism'] = gr.Dropdown(
                                    label="Choose Attention Mechanism",
                                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                                    value=_user_config.get("attention_mechanism", "auto"),
                                    info="Auto = fastest available."
                                )

                                components['settings_skip_engine_check'] = gr.Checkbox(
                                    label="Skip Engine Check at Startup",
                                    value=_user_config.get("skip_engine_check", False),
                                    info="Assumes all engines are available. Faster launch. (Restart required)"
                                )

                            with gr.Column():

                                gr.Markdown("### Model Downloading")
                                components['settings_offline_mode'] = gr.Checkbox(
                                    label="Offline Mode",
                                    value=_user_config.get("offline_mode", False),
                                    info="When enabled, only uses models found in models folder"
                                )

                                components['model_select'] = gr.Dropdown(
                                    label="Select Model to Download to Models Folder",
                                    info="Needed to work in offline mode.\nFor Whisper models, copy local files to models folder",
                                    choices=[
                                        "--- Qwen3-TTS Base ---",
                                        "Qwen3-TTS-12Hz-0.6B-Base",
                                        "Qwen3-TTS-12Hz-1.7B-Base",
                                        "--- Qwen3-TTS CustomVoice ---",
                                        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
                                        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                        "--- Qwen3-TTS VoiceDesign ---",
                                        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                        "--- VibeVoice TTS ---",
                                        "VibeVoice-1.5B",
                                        "VibeVoice-Large (4-bit)",
                                        "VibeVoice-Large",
                                        "--- VibeVoice ASR ---",
                                        "VibeVoice-ASR",
                                        "--- LuxTTS ---",
                                        "LuxTTS",
                                    ],
                                    value="Qwen3-TTS-12Hz-0.6B-Base"
                                )
                                components['download_btn'] = gr.Button("Download Model", scale=1)

                                # Mapping from display names to HuggingFace model IDs
                                components['MODEL_ID_MAP'] = {
                                    "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                    "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                    "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                                    "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                    "VibeVoice-1.5B": "FranckyB/VibeVoice-1.5B",
                                    "VibeVoice-Large (4-bit)": "FranckyB/VibeVoice-Large-4bit",
                                    "VibeVoice-Large": "FranckyB/VibeVoice-Large",
                                    "VibeVoice-ASR": "microsoft/VibeVoice-ASR",
                                    "LuxTTS": "YatharthS/LuxTTS",
                                }

                            with gr.Column():
                                gr.Markdown("### Output")
                                components['settings_output_format'] = gr.Dropdown(
                                    label="Output Format",
                                    choices=["wav", "flac", "mp3"],
                                    value=_user_config.get("output_format", "wav"),
                                    info="Format for saved audio files. MP3 uses 320kbps."
                                )
                                components['settings_manual_save'] = gr.Checkbox(
                                    label="Review Before Saving",
                                    value=_user_config.get("manual_save", False),
                                    info="Results stay in temp until you click Save.\nLets you keep only the ones you like. (Restart required)"
                                )

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Network")
                                components['settings_listen_on_network'] = gr.Checkbox(
                                    label="Listen on Network",
                                    value=_user_config.get("listen_on_network", False),
                                    info="Allow other devices on your local network to connect. (Restart required)"
                                )
                            with gr.Column():
                                gr.Markdown("### Audio Notifications")
                                components['settings_audio_notifications'] = gr.Checkbox(
                                    label="Enable Audio Notifications",
                                    value=_user_config.get("browser_notifications", True),
                                    info="Play sound when audio generation completes"
                                )
                            with gr.Column():
                                gr.Markdown("")

                        gr.Markdown("Configure where files are stored. Changes apply after clicking **Apply Changes**.")
                        # Default folder paths
                        default_folders = {
                            "samples": "samples",
                            "output": "output",
                            "datasets": "datasets",
                            "temp": "temp",
                            "models": "models"
                        }
                        components['default_folders'] = default_folders

                        # Row 1: Samples, Datasets and Output folders
                        with gr.Row():
                            with gr.Column():
                                components['settings_samples_folder'] = gr.Textbox(
                                    label="Voice Samples Folder",
                                    value=_user_config.get("samples_folder", default_folders["samples"]),
                                    info="Folder for voice sample files (.wav + .json)"
                                )
                                components['reset_samples_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_output_folder'] = gr.Textbox(
                                    label="Output Folder",
                                    value=_user_config.get("output_folder", default_folders["output"]),
                                    info="Folder for generated audio files"
                                )
                                components['reset_output_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_datasets_folder'] = gr.Textbox(
                                    label="Datasets Folder",
                                    value=_user_config.get("datasets_folder", default_folders["datasets"]),
                                    info="Folder for training/finetuning datasets"
                                )
                                components['reset_datasets_btn'] = gr.Button("Reset", size="sm")

                        with gr.Row():
                            with gr.Column():
                                components['settings_models_folder'] = gr.Textbox(
                                    label="Downloaded Models Folder",
                                    value=_user_config.get("models_folder", default_folders["models"]),
                                    info="Folder for downloaded model files (Qwen, VibeVoice)"
                                )
                                components['reset_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_trained_models_folder'] = gr.Textbox(
                                    label="Trained Models Folder",
                                    value=_user_config.get("trained_models_folder", default_folders["models"]),
                                    info="Folder for your custom trained models"
                                )
                                components['reset_trained_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                gr.Markdown("")

                    with gr.Column():
                        components['apply_folders_btn'] = gr.Button("Apply Changes", variant="primary", size="lg")
                        components['settings_status'] = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=10
                        )

                with gr.TabItem("Help Guide"):
                    gr.Markdown("# Voice Clone Studio - Help & Guide")

                    components['help_topic'] = gr.Radio(
                        choices=[
                            "Voice Clone",
                            "Voice Presets",
                            "Conversation",
                            "Voice Design",
                            "Prep Samples",
                            "Prep Dataset",
                            "Train Model",
                            "Tips & Tricks"
                        ],
                        value="Voice Clone",
                        show_label=False,
                        interactive=True,
                        container=False
                    )

                    components['help_content'] = gr.HTML(
                        value=format_help_html(show_voice_clone_help()),
                        container=True,
                        padding=True
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Settings tab events."""

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        save_preference = shared_state.get('save_preference')
        download_model_from_huggingface = shared_state.get('download_model_from_huggingface')
        format_help_html = shared_state.get('format_help_html')

        # Lazy import to avoid circular dependency
        from modules.core_components.tools import save_config

        # Save low CPU memory setting
        components['settings_low_cpu_mem'].change(
            lambda x: save_preference("low_cpu_mem_usage", x),
            inputs=[components['settings_low_cpu_mem']],
            outputs=[]
        )
        components['settings_attention_mechanism'].change(
            lambda x: save_preference("attention_mechanism", x),
            inputs=[components['settings_attention_mechanism']],
            outputs=[]
        )

        # Save offline mode setting
        components['settings_offline_mode'].change(
            lambda x: save_preference("offline_mode", x),
            inputs=[components['settings_offline_mode']],
            outputs=[]
        )

        # Save skip engine check setting
        components['settings_skip_engine_check'].change(
            lambda x: save_preference("skip_engine_check", x),
            inputs=[components['settings_skip_engine_check']],
            outputs=[]
        )

        # Save audio notifications setting
        components['settings_audio_notifications'].change(
            lambda x: save_preference("browser_notifications", x),
            inputs=[components['settings_audio_notifications']],
            outputs=[]
        )

        # Save output format setting
        components['settings_output_format'].change(
            lambda x: save_preference("output_format", x),
            inputs=[components['settings_output_format']],
            outputs=[]
        )

        # Save manual save setting
        components['settings_manual_save'].change(
            lambda x: save_preference("manual_save", x),
            inputs=[components['settings_manual_save']],
            outputs=[]
        )

        # Save listen on network setting
        components['settings_listen_on_network'].change(
            lambda x: save_preference("listen_on_network", x),
            inputs=[components['settings_listen_on_network']],
            outputs=[]
        )

        # Tool toggle handlers
        def toggle_tool(tool_name, enabled):
            """Save tool visibility to config."""
            if "enabled_tools" not in _user_config:
                _user_config["enabled_tools"] = {}
            _user_config["enabled_tools"][tool_name] = enabled
            # Use save_preference to persist (saves full config)
            save_preference("enabled_tools", _user_config["enabled_tools"])
            return "Restart the app to apply changes."

        # Engine toggle handlers
        TTS_ENGINES = shared_state.get('TTS_ENGINES', {})

        def toggle_engine(engine_key, enabled):
            """Save engine visibility to config."""
            if "enabled_engines" not in _user_config:
                _user_config["enabled_engines"] = {}
            _user_config["enabled_engines"][engine_key] = enabled
            save_preference("enabled_engines", _user_config["enabled_engines"])
            return "Restart the app to apply changes."

        for key, label in TOGGLEABLE_TOOLS:
            comp = components[f'tool_toggle_{key}']
            # Use default arg to capture key in closure
            comp.change(
                lambda enabled, k=key: toggle_tool(k, enabled),
                inputs=[comp],
                outputs=[components['settings_status']]
            )

        for engine_key in TTS_ENGINES:
            comp = components[f'engine_toggle_{engine_key}']
            comp.change(
                lambda enabled, k=engine_key: toggle_engine(k, enabled),
                inputs=[comp],
                outputs=[components['settings_status']]
            )

        # ASR engine toggle handlers
        ASR_ENGINES = shared_state.get('ASR_ENGINES', {})

        def toggle_asr_engine(engine_key, enabled):
            """Save ASR engine visibility to config."""
            if "enabled_asr_engines" not in _user_config:
                _user_config["enabled_asr_engines"] = {}
            _user_config["enabled_asr_engines"][engine_key] = enabled
            save_preference("enabled_asr_engines", _user_config["enabled_asr_engines"])
            return "Restart the app to apply changes."

        for engine_key in ASR_ENGINES:
            comp = components[f'asr_toggle_{engine_key}']
            comp.change(
                lambda enabled, k=engine_key: toggle_asr_engine(k, enabled),
                inputs=[comp],
                outputs=[components['settings_status']]
            )

        components['bypass_split_limit'].change(
            lambda x: save_preference("bypass_split_limit", x),
            inputs=[components['bypass_split_limit']],
            outputs=[]
        )

        # Reset button handlers
        def reset_folder(folder_key):
            return components['default_folders'][folder_key]

        components['reset_samples_btn'].click(
            lambda: reset_folder("samples"),
            outputs=[components['settings_samples_folder']]
        )

        components['reset_output_btn'].click(
            lambda: reset_folder("output"),
            outputs=[components['settings_output_folder']]
        )

        components['reset_datasets_btn'].click(
            lambda: reset_folder("datasets"),
            outputs=[components['settings_datasets_folder']]
        )

        components['reset_models_btn'].click(
            lambda: reset_folder("models"),
            outputs=[components['settings_models_folder']]
        )

        components['reset_trained_models_btn'].click(
            lambda: reset_folder("models"),
            outputs=[components['settings_trained_models_folder']]
        )

        components['reset_llama_cpp_path_btn'].click(
            lambda: "",
            outputs=[components['settings_llama_cpp_path']]
        )

        components['reset_llama_models_path_btn'].click(
            lambda: "",
            outputs=[components['settings_llama_models_path']]
        )

        def download_model_clicked(model_display_name):
            if not model_display_name or model_display_name.startswith("---"):
                return "❌ Please select an actual model (not a category header)"
            # Convert display name to full model ID
            model_id = components['MODEL_ID_MAP'].get(model_display_name, model_display_name)

            success, message, path = download_model_from_huggingface(model_id, progress=None)

            status = f"✓ {message}" if success else f"❌ {message}"
            return status

        # Apply folder changes
        def apply_folder_changes(samples, output, datasets, models, trained_models, llama_cpp_path, llama_models_path):
            try:
                # Get project root directory
                base_dir = Path(__file__).parent.parent.parent.parent

                # Update paths
                new_samples = base_dir / samples
                new_output = base_dir / output
                new_datasets = base_dir / datasets
                new_models = base_dir / models
                new_trained_models = base_dir / trained_models

                # Create directories if they don't exist
                new_samples.mkdir(exist_ok=True)
                new_output.mkdir(exist_ok=True)
                new_datasets.mkdir(exist_ok=True)
                new_models.mkdir(exist_ok=True)
                new_trained_models.mkdir(exist_ok=True)

                # Set HuggingFace cache environment variable
                import os
                os.environ['HF_HOME'] = str(new_models)

                # Save to config
                _user_config["samples_folder"] = samples
                _user_config["output_folder"] = output
                _user_config["datasets_folder"] = datasets
                _user_config["models_folder"] = models
                _user_config["trained_models_folder"] = trained_models
                _user_config["llama_cpp_path"] = llama_cpp_path.strip()
                _user_config["llama_models_path"] = llama_models_path.strip()
                save_config(_user_config)

                status_lines = [
                    "Folder paths updated successfully!",
                    f"\nSamples: {new_samples}",
                    f"Output: {new_output}",
                    f"Datasets: {new_datasets}",
                    f"Downloaded Models: {new_models}",
                    f"Trained Models: {new_trained_models}",
                ]
                if llama_cpp_path.strip():
                    status_lines.append(f"llama.cpp: {llama_cpp_path.strip()}")
                if llama_models_path.strip():
                    status_lines.append(f"LLM Models: {llama_models_path.strip()}")
                status_lines.append("\nNote: Restart the app to fully apply changes to all components.")
                return "\n".join(status_lines)

            except Exception as e:
                return f"❌ Error applying changes: {str(e)}"

        components['download_btn'].click(
            fn=download_model_clicked,
            inputs=[components['model_select']],
            outputs=[components['settings_status']]
        )

        components['apply_folders_btn'].click(
            apply_folder_changes,
            inputs=[
                components['settings_samples_folder'], components['settings_output_folder'],
                components['settings_datasets_folder'], components['settings_models_folder'],
                components['settings_trained_models_folder'],
                components['settings_llama_cpp_path'], components['settings_llama_models_path']
            ],
            outputs=[components['settings_status']]
        )

        # Help Guide topic selector
        def show_help(topic):
            help_map = {
                "Voice Clone": show_voice_clone_help,
                "Conversation": show_conversation_help,
                "Voice Presets": show_voice_presets_help,
                "Voice Design": show_voice_design_help,
                "Prep Samples": show_prep_audio_help,
                "Prep Dataset": show_dataset_help,
                "Train Model": show_train_help,
                "Tips & Tricks": show_tips_help
            }
            return format_help_html(help_map[topic]())

        components['help_topic'].change(
            fn=show_help,
            inputs=components['help_topic'],
            outputs=components['help_content']
        )


# Export for tab registry
get_tool_class = lambda: SettingsTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(SettingsTool, port=7870, title="Settings - Standalone")
