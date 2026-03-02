"""
Voice Presets Tab

Use Qwen3-TTS pre-trained models or custom trained models with style control.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import sys
from pathlib import Path
import gradio as gr
import soundfile as sf
from datetime import datetime
from textwrap import dedent
from pathlib import Path
from gradio_filelister import FileLister

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.emotion_manager import process_save_emotion_result, process_delete_emotion_result


class VoicePresetsTool(Tool):
    """Voice Presets tool implementation."""

    config = ToolConfig(
        name="Voice Presets",
        module_name="tool_voice_presets",
        description="Generate with trained models or Qwen3's Style-Controlled Premium Speakers",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Presets tool UI."""
        components = {}

        # Get helper functions and config
        get_trained_models = shared_state['get_trained_models']
        get_trained_vibevoice_models = shared_state.get('get_trained_vibevoice_models', lambda: [])
        create_qwen_advanced_params = shared_state['create_qwen_advanced_params']
        create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
        _user_config = shared_state['_user_config']
        _active_emotions = shared_state['_active_emotions']
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        MODEL_SIZES_CUSTOM = shared_state['MODEL_SIZES_CUSTOM']
        VIBEVOICE_STREAMING_VOICES = shared_state.get('VIBEVOICE_STREAMING_VOICES', [])
        LANGUAGES = shared_state['LANGUAGES']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        format_help_html = shared_state['format_help_html']
        get_sample_choices = shared_state['get_sample_choices']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']

        ALL_VOICE_TYPES = ["Qwen Trained", "VibeVoice Trained", "Qwen Speakers", "VibeVoice Speakers"]

        with gr.TabItem("Voice Presets", id="tab_voice_presets") as voice_presets_tab:
            components['voice_presets_tab'] = voice_presets_tab
            gr.Markdown("Generate with Qwen3 or VibeVoice trained models and speakers")

            initial_voice_type = _user_config.get("voice_type", "Qwen Trained")
            if initial_voice_type not in ALL_VOICE_TYPES:
                initial_voice_type = "Qwen Trained"
            is_premium = (initial_voice_type.strip() == "Qwen Speakers")
            is_vv_trained = (initial_voice_type.strip() == "VibeVoice Trained")
            is_vv_streaming = (initial_voice_type.strip() == "VibeVoice Speakers")

            components['voice_type_radio'] = gr.Radio(
                choices=ALL_VOICE_TYPES,
                value=initial_voice_type,
                show_label=False,
                container=False,
                label="Voice Source"
            )

            with gr.Row():
                # Left - Speaker selection
                with gr.Column(scale=1):
                    gr.Markdown("### Select Voice Type")
                    # Qwen Speakers dropdown
                    components['speaker_section'] = gr.Column(visible=is_premium)
                    with components['speaker_section']:
                        speaker_choices = CUSTOM_VOICE_SPEAKERS
                        components['custom_speaker_dropdown'] = gr.Dropdown(
                            choices=speaker_choices,
                            label="Speaker",
                            info="Choose a Qwen speaker voice"
                        )

                        components['custom_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_CUSTOM,
                            value=_user_config.get("custom_voice_size", "Large"),
                            label="Model",
                            info="Small = faster, Large = better quality",
                            scale=1
                        )

                        premium_speaker_guide = dedent("""\
                            **Qwen Speakers:**

                            | Speaker | Voice | Language |
                            |---------|-------|----------|
                            | Vivian | Bright young female    | 🇨🇳 Chinese |
                            | Serena | Warm gentle female    | 🇨🇳 Chinese |
                            | Uncle_Fu | Seasoned mellow male    | 🇨🇳 Chinese |
                            | Dylan | Youthful Beijing male    | 🇨🇳 Chinese |
                            | Eric | Lively Chengdu male    | 🇨🇳 Chinese |
                            | Ryan | Dynamic male | 🇺🇸 English    |
                            | Aiden | Sunny American male    | 🇺🇸 English |
                            | Ono_Anna | Playful female    | 🇯🇵 Japanese |
                            | Sohee | Warm female    | 🇰🇷 Korean |

                            *Each speaker works best in native language.*
                            """)

                        gr.HTML(
                            value=format_help_html(premium_speaker_guide, height="auto"),
                            container=True,
                            padding=True
                        )
                    # Trained models dropdown (Qwen3)
                    is_qwen_trained = not is_premium and not is_vv_trained and not is_vv_streaming
                    components['trained_section'] = gr.Column(visible=is_qwen_trained)
                    with components['trained_section']:
                        def get_initial_model_list():
                            """Get initial list of trained models for dropdown initialization."""
                            models = get_trained_models()
                            if not models:
                                return ["(No trained models found)"]
                            return ["(Select Model)"] + [m['display_name'] for m in models]

                        def refresh_trained_models():
                            """Refresh model list."""
                            models = get_trained_models()
                            if not models:
                                return gr.update(choices=["(No trained models found)"], value="(No trained models found)")
                            choices = ["(Select Model)"] + [m['display_name'] for m in models]
                            return gr.update(choices=choices, value="(Select Model)")

                        initial_choices = get_initial_model_list()
                        initial_value = initial_choices[0]

                        components['trained_model_dropdown'] = gr.Dropdown(
                            choices=initial_choices,
                            value=initial_value,
                            label="Trained Model",
                            info="Select your custom trained voice"
                        )

                        components['refresh_trained_btn'] = gr.Button("Refresh", size="sm", visible=False)

                        # ICL (In-Context Learning) mode for enhanced voice cloning
                        components['icl_enabled'] = gr.Checkbox(
                            label="Enable ICL (Experimental)",
                            value=False,
                            info="Use a voice sample for more expressive results"
                        )

                        components['icl_sample_section'] = gr.Column(visible=False)
                        with components['icl_sample_section']:
                            components['icl_dataset_dropdown'] = gr.Dropdown(
                                choices=["(Select Dataset)"] + get_dataset_folders(),
                                value="(Select Dataset)",
                                label="Dataset",
                                info="Select the dataset used for training",
                                interactive=True
                            )
                            components['icl_refresh_datasets'] = gr.Button("Refresh Datasets", size="sm")
                            components['icl_voice_lister'] = FileLister(
                                value=[],
                                height=150,
                                show_footer=False,
                                interactive=True,
                            )

                            components['icl_audio_preview'] = gr.Audio(
                                label="Preview",
                                type="filepath",
                                interactive=False,
                                elem_id="icl-audio-preview"
                            )

                        trained_models_tip = dedent("""\
                        **Trained Models:**

                        Custom voices you've trained in the Train Model tab.

                        **ICL Mode (Enhanced Voice Clone):**
                        Select a sample from your training dataset for more expressive results. The model uses both its training and the reference audio for better voice similarity.

                        *Tip: Later epochs are usually better trained*
                        """)
                        gr.HTML(
                            value=format_help_html(trained_models_tip, height="auto"),
                            container=True,
                            padding=True,
                        )

                    # VibeVoice Trained models (LoRA checkpoints)
                    components['vv_trained_section'] = gr.Column(visible=is_vv_trained)
                    with components['vv_trained_section']:
                        def get_initial_vv_model_list():
                            """Get initial list of trained VibeVoice models."""
                            models = get_trained_vibevoice_models()
                            if not models:
                                return ["(No trained VibeVoice models found)"]
                            return ["(Select Model)"] + [m['display_name'] for m in models]

                        def refresh_vv_trained_models():
                            """Refresh VibeVoice trained model list."""
                            models = get_trained_vibevoice_models()
                            if not models:
                                return gr.update(choices=["(No trained VibeVoice models found)"], value="(No trained VibeVoice models found)")
                            choices = ["(Select Model)"] + [m['display_name'] for m in models]
                            return gr.update(choices=choices, value="(Select Model)")

                        vv_initial_choices = get_initial_vv_model_list()
                        vv_initial_value = vv_initial_choices[0]

                        components['vv_trained_model_dropdown'] = gr.Dropdown(
                            choices=vv_initial_choices,
                            value=vv_initial_value,
                            label="VibeVoice Trained Model",
                            info="Select your VibeVoice LoRA-trained voice"
                        )
                        components['vv_refresh_trained_btn'] = gr.Button("Refresh", size="sm", visible=False)

                        vv_trained_tip = dedent("""\
                        **VibeVoice Trained Models:**

                        Voices trained with VibeVoice LoRA in the Train Model tab. The voice identity is baked into the LoRA weights, so no reference audio is needed.

                        *Tip: Higher epochs = better trained, but watch for overfit*
                        """)
                        gr.HTML(
                            value=format_help_html(vv_trained_tip, height="auto"),
                            container=True,
                            padding=True,
                        )

                    # VibeVoice Streaming (baked-in voices)
                    components['vv_streaming_section'] = gr.Column(visible=is_vv_streaming)
                    with components['vv_streaming_section']:
                        components['vv_streaming_voice_dropdown'] = gr.Dropdown(
                            choices=VIBEVOICE_STREAMING_VOICES,
                            value=VIBEVOICE_STREAMING_VOICES[0] if VIBEVOICE_STREAMING_VOICES else None,
                            label="Voice",
                            info="Pre-built VibeVoice 0.5B speaker voice"
                        )

                        vv_streaming_guide = dedent("""\
                            **VibeVoice Speakers:**

                            | Voice | Type | Language |
                            |-------|------|----------|
                            | Carter | Male | English |
                            | Davis | Male | English |
                            | Emma | Female | English |
                            | Frank | Male | English |
                            | Grace | Female | English |
                            | Mike | Male | English |
                            | Samuel | Male | Indian English |

                            *Lightweight 0.5B model with fast generation.*
                            """)
                        gr.HTML(
                            value=format_help_html(vv_streaming_guide, height="auto"),
                            container=True,
                            padding=True
                        )

                # Right - Generation
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['custom_text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want spoken...",
                        lines=6
                    )

                    components['custom_instruct_input'] = gr.Textbox(
                        label="Style Instructions (Optional)",
                        placeholder="e.g., 'Speak with excitement' or 'Very sad and slow' or '用愤怒的语气说'",
                        lines=2,
                        info="Control emotion, tone, speed, etc.",
                        visible=is_premium
                    )

                    import modules.core_components.prompt_hub as _prompt_hub
                    components.update(_prompt_hub.create_prompt_loader("vp", "Saved Prompts"))

                    with gr.Row():
                        components['custom_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['custom_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # --- Qwen Speakers Advanced Parameters (no emotions) ---
                    components['qwen_custom_advanced'] = gr.Column(visible=is_premium)
                    with components['qwen_custom_advanced']:
                        custom_params = create_qwen_advanced_params(
                            emotions_dict=_active_emotions,
                            include_emotion=False,
                            visible=True,
                            shared_state=shared_state
                        )
                    components['custom_do_sample'] = custom_params['do_sample']
                    components['custom_temperature'] = custom_params['temperature']
                    components['custom_top_k'] = custom_params['top_k']
                    components['custom_top_p'] = custom_params['top_p']
                    components['custom_repetition_penalty'] = custom_params['repetition_penalty']
                    components['custom_max_new_tokens'] = custom_params['max_new_tokens']
                    components['custom_params'] = custom_params

                    # --- Trained Models Advanced Parameters (with emotions) ---
                    components['qwen_trained_advanced'] = gr.Column(visible=is_qwen_trained)
                    with components['qwen_trained_advanced']:
                        trained_params = create_qwen_advanced_params(
                            emotions_dict=_active_emotions,
                            include_emotion=True,
                            initial_emotion="(None)",
                            initial_intensity=1.0,
                            visible=True,
                            emotion_visible=True,
                            shared_state=shared_state
                        )
                    components['trained_emotion_row'] = trained_params.get('emotion_row')
                    components['trained_emotion_buttons_row'] = trained_params.get('emotion_buttons_row')
                    components['trained_emotion_preset'] = trained_params['emotion_preset']
                    components['trained_emotion_intensity'] = trained_params['emotion_intensity']
                    components['trained_save_emotion_btn'] = trained_params.get('save_emotion_btn')
                    components['trained_delete_emotion_btn'] = trained_params.get('delete_emotion_btn')
                    components['trained_emotion_save_name'] = trained_params.get('emotion_save_name')
                    components['trained_do_sample'] = trained_params['do_sample']
                    components['trained_temperature'] = trained_params['temperature']
                    components['trained_top_k'] = trained_params['top_k']
                    components['trained_top_p'] = trained_params['top_p']
                    components['trained_repetition_penalty'] = trained_params['repetition_penalty']
                    components['trained_max_new_tokens'] = trained_params['max_new_tokens']
                    components['trained_params'] = trained_params

                    # --- VibeVoice Trained Parameters ---
                    components['vv_trained_advanced'] = gr.Column(visible=is_vv_trained)
                    with components['vv_trained_advanced']:
                        vv_trained_params = create_vibevoice_advanced_params(visible=True)
                    components['vv_trained_cfg_scale'] = vv_trained_params['cfg_scale']
                    components['vv_trained_num_steps'] = vv_trained_params['num_steps']
                    components['vv_trained_do_sample'] = vv_trained_params['do_sample']
                    components['vv_trained_repetition_penalty'] = vv_trained_params['repetition_penalty']
                    components['vv_trained_temperature'] = vv_trained_params['temperature']
                    components['vv_trained_top_k'] = vv_trained_params['top_k']
                    components['vv_trained_top_p'] = vv_trained_params['top_p']
                    components['vv_trained_params'] = vv_trained_params

                    # --- VibeVoice Streaming Parameters ---
                    components['vv_streaming_advanced'] = gr.Column(visible=is_vv_streaming)
                    with components['vv_streaming_advanced']:
                        with gr.Accordion("Streaming Parameters", open=False):
                            with gr.Row():
                                components['vv_streaming_cfg_scale'] = gr.Slider(
                                    minimum=1.0, maximum=5.0, value=1.5, step=0.1,
                                    label="CFG Scale",
                                    info="Classifier-free guidance strength"
                                )
                                components['vv_streaming_ddpm_steps'] = gr.Slider(
                                    minimum=5, maximum=50, value=20, step=1,
                                    label="DDPM Steps",
                                    info="Denoising diffusion steps"
                                )

                    components['custom_generate_btn'] = gr.Button("Generate Audio", variant="primary", size="lg")

                    components['custom_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )

                    # Save button — visible only when "Review Before Saving" is enabled
                    manual_save = _user_config.get("manual_save", False)
                    components['save_result_btn'] = gr.Button(
                        "Save to Output", variant="primary", size="lg",
                        visible=manual_save, interactive=False
                    )
                    # Hidden state for metadata text
                    components['_result_metadata'] = gr.Textbox(visible=False)

                    components['preset_status'] = gr.Textbox(label="Status", max_lines=5, interactive=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Presets tab events."""

        # Get helper functions and directories
        get_trained_models = shared_state['get_trained_models']
        get_trained_vibevoice_models = shared_state.get('get_trained_vibevoice_models', lambda: [])
        get_sample_choices = shared_state['get_sample_choices']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        TEMP_DIR = shared_state['TEMP_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        user_config = shared_state.get('_user_config', {})
        save_audio_to_temp = shared_state['save_audio_to_temp']
        save_result_to_output = shared_state['save_result_to_output']

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        # Wire param persistence (auto-save on change)
        wire_param_persistence = shared_state['wire_param_persistence']
        param_map = {
            'qwen_custom': [
                ('custom_do_sample', 'do_sample'),
                ('custom_temperature', 'temperature'),
                ('custom_top_k', 'top_k'),
                ('custom_top_p', 'top_p'),
                ('custom_repetition_penalty', 'repetition_penalty'),
                ('custom_max_new_tokens', 'max_new_tokens'),
            ],
            'qwen_trained': [
                ('trained_emotion_preset', 'emotion_preset'),
                ('trained_emotion_intensity', 'emotion_intensity'),
                ('trained_do_sample', 'do_sample'),
                ('trained_temperature', 'temperature'),
                ('trained_top_k', 'top_k'),
                ('trained_top_p', 'top_p'),
                ('trained_repetition_penalty', 'repetition_penalty'),
                ('trained_max_new_tokens', 'max_new_tokens'),
            ],
            'vv_trained': [
                ('vv_trained_cfg_scale', 'cfg_scale'),
                ('vv_trained_num_steps', 'num_steps'),
                ('vv_trained_do_sample', 'do_sample'),
                ('vv_trained_repetition_penalty', 'repetition_penalty'),
                ('vv_trained_temperature', 'temperature'),
                ('vv_trained_top_k', 'top_k'),
                ('vv_trained_top_p', 'top_p'),
            ],
            'vv_streaming': [
                ('vv_streaming_cfg_scale', 'cfg_scale'),
                ('vv_streaming_ddpm_steps', 'ddpm_steps'),
            ],
        }
        wire_param_persistence(components, user_config, param_map)

        # Create restore handler for applying saved params on tab select
        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, user_config, param_map)

        custom_params = components['custom_params']

        def generate_custom_voice_handler(text_to_generate, language, speaker, instruct, seed, model_size="1.7B",
                                          do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                                          repetition_penalty=1.05, max_new_tokens=2048, progress=gr.Progress()):
            """Generate audio using the CustomVoice model with premium speakers."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", "", gr.update()

            if not speaker:
                return None, "❌ Please select a speaker.", "", gr.update()

            try:
                progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")

                audio_data, sr = tts_manager.generate_custom_voice(
                    text=text_to_generate,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    model_size=model_size,
                    seed=int(seed) if seed is not None else -1,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )

                progress(0.8, desc="Saving audio...")
                from modules.core_components.audio_utils import make_stem_from_text, resolve_output_stem
                stem = make_stem_from_text(text_to_generate, sample_name=speaker)
                filename_stem = resolve_output_stem(stem, OUTPUT_DIR, clip_count=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Custom Voice
                    Model: CustomVoice {model_size}
                    Speaker: {speaker}
                    Language: {language}
                    Seed: {seed}
                    Instruct: {instruct.strip() if instruct else ''}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                # Always save WAV to temp first
                temp_path = save_audio_to_temp(audio_data, sr, TEMP_DIR, filename_stem)

                instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
                manual_save_mode = user_config.get("manual_save", False)
                if manual_save_mode:
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(temp_path), f"Speaker: {speaker}{instruct_msg}\nSeed: {seed} | {model_size}\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
                else:
                    output_format = user_config.get("output_format", "wav")
                    output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(output_path), f"Audio saved to: {output_path.name}\nSpeaker: {speaker}{instruct_msg}\nSeed: {seed} | {model_size}", "", gr.update()

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def generate_with_trained_model_handler(text_to_generate, language, speaker_name, checkpoint_path, instruct, seed,
                                                do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                                                repetition_penalty=1.05, max_new_tokens=2048,
                                                icl_enabled=False, icl_dataset=None, icl_sample_name=None,
                                                progress=gr.Progress()):
            """Generate audio using a trained custom voice model checkpoint."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", "", gr.update()

            # Resolve ICL voice sample from dataset if enabled
            voice_sample_path = None
            ref_text = None
            if icl_enabled:
                if not icl_dataset or icl_dataset in ["(Select Dataset)", ""]:
                    return None, "❌ Please select a dataset for ICL mode.", "", gr.update()
                if not icl_sample_name or icl_sample_name.strip() == "":
                    return None, "❌ Please select a voice sample for ICL mode.", "", gr.update()

                audio_path = DATASETS_DIR / icl_dataset / icl_sample_name
                if not audio_path.exists():
                    return None, f"❌ Audio file not found: {icl_sample_name}", "", gr.update()

                voice_sample_path = str(audio_path)

                # Look for matching .txt transcript file
                txt_path = audio_path.with_suffix(".txt")
                if txt_path.exists():
                    ref_text = txt_path.read_text(encoding="utf-8").strip()

                if not ref_text or not ref_text.strip():
                    return None, (
                        f"❌ No transcript found for '{icl_sample_name}' in dataset '{icl_dataset}'.\n\n"
                        "Make sure the sample has a matching .txt file with the transcript."
                    ), "", gr.update()

            try:
                mode_desc = "ICL mode" if icl_enabled and voice_sample_path else "speaker embedding"
                progress(0.1, desc=f"Loading trained model ({mode_desc})...")

                # Call tts_manager method
                audio_data, sr = tts_manager.generate_with_trained_model(
                    text=text_to_generate,
                    language=language,
                    speaker_name=speaker_name,
                    checkpoint_path=checkpoint_path,
                    instruct=instruct if not icl_enabled else None,
                    seed=int(seed) if seed is not None else -1,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    user_config=user_config,
                    icl_mode=icl_enabled and voice_sample_path is not None,
                    voice_sample_path=voice_sample_path,
                    ref_text=ref_text,
                )

                progress(0.8, desc="Saving audio...")
                from modules.core_components.audio_utils import make_stem_from_text, resolve_output_stem
                stem = make_stem_from_text(text_to_generate, sample_name=speaker_name)
                filename_stem = resolve_output_stem(stem, OUTPUT_DIR, clip_count=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                icl_active = icl_enabled and voice_sample_path is not None
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Trained Model{' (ICL)' if icl_active else ''}
                    Model: {checkpoint_path}
                    Speaker: {speaker_name}
                    Language: {language}
                    Seed: {seed}
                    ICL Mode: {icl_active}
                    ICL Dataset: {icl_dataset if icl_active else 'N/A'}
                    ICL Sample: {icl_sample_name if icl_active else 'N/A'}
                    Instruct: {instruct.strip() if instruct and not icl_enabled else ''}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                # Always save WAV to temp first
                temp_path = save_audio_to_temp(audio_data, sr, TEMP_DIR, filename_stem)

                instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() and not icl_enabled else ""
                icl_msg = f" | ICL: {icl_dataset}/{icl_sample_name}" if icl_enabled and voice_sample_path else ""
                manual_save_mode = user_config.get("manual_save", False)
                if manual_save_mode:
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(temp_path), f"Speaker: {speaker_name}{instruct_msg}{icl_msg}\nSeed: {seed} | Trained Model\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
                else:
                    output_format = user_config.get("output_format", "wav")
                    output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(output_path), f"Audio saved: {output_path.name}\nSpeaker: {speaker_name}{instruct_msg}{icl_msg}\nSeed: {seed} | Trained Model", "", gr.update()

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def generate_vibevoice_streaming_handler(
                text_to_generate, seed, voice_name,
                cfg_scale=1.5, ddpm_steps=20, progress=gr.Progress()):
            """Generate audio using VibeVoice Speakers (0.5B) with baked-in voices."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", "", gr.update()
            if not voice_name:
                return None, "❌ Please select a speaker voice.", "", gr.update()
            try:
                import random
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                progress(0.1, desc=f"Loading VibeVoice Speakers ({voice_name})...")
                audio_data, sr = tts_manager.generate_vibevoice_streaming(
                    text=text_to_generate,
                    voice_name=voice_name,
                    cfg_scale=float(cfg_scale),
                    ddpm_steps=int(ddpm_steps),
                    seed=seed,
                )
                progress(0.8, desc="Saving audio...")
                from modules.core_components.audio_utils import make_stem_from_text, resolve_output_stem
                stem = make_stem_from_text(text_to_generate, sample_name=voice_name)
                filename_stem = resolve_output_stem(stem, OUTPUT_DIR, clip_count=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: VibeVoice Speakers
                    Model: VibeVoice-Realtime-0.5B
                    Voice: {voice_name}
                    Seed: {seed}
                    CFG Scale: {cfg_scale}
                    DDPM Steps: {ddpm_steps}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                temp_path = save_audio_to_temp(audio_data, sr, TEMP_DIR, filename_stem)
                manual_save_mode = user_config.get("manual_save", False)
                if manual_save_mode:
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(temp_path), f"Voice: {voice_name} | Seed: {seed}\nVibeVoice Speakers 0.5B\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
                else:
                    output_format = user_config.get("output_format", "wav")
                    output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(output_path), f"Audio saved: {output_path.name}\nVoice: {voice_name} | Seed: {seed} | VibeVoice Speakers", "", gr.update()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def generate_vibevoice_trained_handler(
                text_to_generate, language, vv_trained_model, seed,
                do_sample=False, temperature=1.0, top_k=50, top_p=1.0,
                repetition_penalty=1.0, cfg_scale=1.3, num_steps=10,
                progress=gr.Progress()):
            """Generate audio using a trained VibeVoice LoRA checkpoint."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", "", gr.update()
            if not vv_trained_model or vv_trained_model in ["(No trained VibeVoice models found)", "(Select Model)"]:
                return None, "❌ Please select a trained VibeVoice model or train one in the Train Model tab.", "", gr.update()

            # Resolve model path from display name
            models = get_trained_vibevoice_models()
            model_path = None
            speaker_name = None
            for m in models:
                if m['display_name'] == vv_trained_model:
                    model_path = m['path']
                    speaker_name = m['speaker_name']
                    break
            if not model_path:
                return None, f"❌ Model not found: {vv_trained_model}", "", gr.update()

            try:
                import random
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                progress(0.1, desc=f"Loading VibeVoice LoRA ({speaker_name})...")
                audio_data, sr = tts_manager.generate_with_trained_vibevoice(
                    text=text_to_generate,
                    language=language,
                    checkpoint_path=model_path,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    cfg_scale=float(cfg_scale),
                    num_steps=int(num_steps),
                    user_config=user_config,
                )
                progress(0.8, desc="Saving audio...")
                from modules.core_components.audio_utils import make_stem_from_text, resolve_output_stem
                stem = make_stem_from_text(text_to_generate, sample_name=speaker_name)
                filename_stem = resolve_output_stem(stem, OUTPUT_DIR, clip_count=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: VibeVoice Trained (LoRA)
                    Model: {model_path}
                    Speaker: {speaker_name}
                    Language: {language}
                    Seed: {seed}
                    CFG Scale: {cfg_scale}
                    Steps: {num_steps}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                temp_path = save_audio_to_temp(audio_data, sr, TEMP_DIR, filename_stem)
                manual_save_mode = user_config.get("manual_save", False)
                if manual_save_mode:
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(temp_path), f"Speaker: {speaker_name}\nSeed: {seed} | VibeVoice Trained\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
                else:
                    output_format = user_config.get("output_format", "wav")
                    output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(output_path), f"Audio saved: {output_path.name}\nSpeaker: {speaker_name} | Seed: {seed} | VibeVoice Trained", "", gr.update()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def extract_speaker_name(selection):
            """Extract speaker name from dropdown selection."""
            if not selection:
                return None
            return selection.split(" - ")[0].split(" (")[0]

        def toggle_voice_type(voice_type):
            """Toggle between voice type sections."""
            is_premium = voice_type == "Qwen Speakers"
            is_qwen_trained = voice_type == "Qwen Trained"
            is_vv_trained = voice_type == "VibeVoice Trained"
            is_vv_streaming = voice_type == "VibeVoice Speakers"
            return (
                gr.update(visible=is_premium),          # speaker_section
                gr.update(visible=is_qwen_trained),     # trained_section
                gr.update(visible=is_vv_trained),       # vv_trained_section
                gr.update(visible=is_vv_streaming),     # vv_streaming_section
                gr.update(visible=is_premium),          # instruct_input
                gr.update(visible=is_premium),          # qwen_custom_advanced
                gr.update(visible=is_qwen_trained),     # qwen_trained_advanced
                gr.update(visible=is_vv_trained),       # vv_trained_advanced
                gr.update(visible=is_vv_streaming),     # vv_streaming_advanced
            )

        def generate_with_voice_type(text, lang, speaker_sel, instruct, seed, model_size, voice_type, premium_speaker, trained_model,
                                     custom_do_sample, custom_temperature, custom_top_k, custom_top_p, custom_repetition_penalty, custom_max_new_tokens,
                                     trained_do_sample, trained_temperature, trained_top_k, trained_top_p, trained_repetition_penalty, trained_max_new_tokens,
                                     icl_enabled=False, icl_dataset=None, icl_lister_value=None,
                                     vv_trained_model=None,
                                     vv_trained_cfg_scale=3.0, vv_trained_num_steps=20,
                                     vv_trained_do_sample=False, vv_trained_rep_pen=1.1,
                                     vv_trained_temperature=1.0, vv_trained_top_k=50, vv_trained_top_p=1.0,
                                     vv_streaming_voice=None,
                                     vv_streaming_cfg_scale=1.5, vv_streaming_ddpm_steps=20,
                                     progress=gr.Progress()):
            """Generate audio with the selected voice type."""
            icl_sample_name = get_selected_icl_filename(icl_lister_value) if icl_lister_value else None

            if voice_type == "Qwen Speakers":
                speaker = extract_speaker_name(premium_speaker)
                if not speaker:
                    return None, "❌ Please select a premium speaker", "", gr.update()

                return generate_custom_voice_handler(
                    text, lang, speaker, instruct, seed,
                    "1.7B" if model_size == "Large" else "0.6B",
                    custom_do_sample, custom_temperature, custom_top_k, custom_top_p, custom_repetition_penalty, custom_max_new_tokens,
                    progress
                )

            elif voice_type == "VibeVoice Speakers":
                return generate_vibevoice_streaming_handler(
                    text, seed, vv_streaming_voice, vv_streaming_cfg_scale, vv_streaming_ddpm_steps, progress
                )

            elif voice_type == "VibeVoice Trained":
                return generate_vibevoice_trained_handler(
                    text, lang, vv_trained_model, seed,
                    vv_trained_do_sample, vv_trained_temperature, vv_trained_top_k, vv_trained_top_p,
                    vv_trained_rep_pen, vv_trained_cfg_scale, vv_trained_num_steps,
                    progress
                )

            else:
                # Qwen Trained
                if not trained_model or trained_model in ["(No trained models found)", "(Select Model)"]:
                    return None, "❌ Please select a trained model or train one first", "", gr.update()

                models = get_trained_models()
                model_path = None
                speaker_name = None
                for model in models:
                    if model['display_name'] == trained_model:
                        model_path = model['path']
                        speaker_name = model['speaker_name']
                        break

                if not model_path:
                    return None, f"❌ Model not found: {trained_model}", "", gr.update()

                return generate_with_trained_model_handler(
                    text, lang, speaker_name, model_path, instruct, seed,
                    trained_do_sample, trained_temperature, trained_top_k, trained_top_p, trained_repetition_penalty, trained_max_new_tokens,
                    icl_enabled, icl_dataset, icl_sample_name,
                    progress
                )

        # Only wire events for components that exist (not None)
        if components.get('voice_type_radio') is not None:
            components['voice_type_radio'].change(
                toggle_voice_type,
                inputs=[components['voice_type_radio']],
                outputs=[
                    components['speaker_section'], components['trained_section'],
                    components['vv_trained_section'], components['vv_streaming_section'],
                    components['custom_instruct_input'],
                    components['qwen_custom_advanced'],
                    components['qwen_trained_advanced'],
                    components['vv_trained_advanced'],
                    components['vv_streaming_advanced'],
                ]
            ).then(
                lambda x: save_preference("voice_type", x),
                inputs=[components['voice_type_radio']],
                outputs=[]
            )

        # Auto-refresh trained models when tab is selected
        def refresh_all_model_dropdowns():
            """Refresh both Qwen and VibeVoice trained model dropdowns on tab select."""
            qwen_models = get_trained_models()
            if qwen_models:
                qwen_choices = ["(Select Model)"] + [m['display_name'] for m in qwen_models]
            else:
                qwen_choices = ["(No trained models found)"]

            vv_models = get_trained_vibevoice_models()
            if vv_models:
                vv_choices = ["(Select Model)"] + [m['display_name'] for m in vv_models]
            else:
                vv_choices = ["(No trained VibeVoice models found)"]

            return (
                gr.update(choices=qwen_choices),
                gr.update(choices=vv_choices),
            )

        components['voice_presets_tab'].select(
            refresh_all_model_dropdowns,
            outputs=[components['trained_model_dropdown'], components['vv_trained_model_dropdown']]
        )

        # Restore saved params when accordion is opened
        components['custom_params']['accordion'].expand(restore_fn, outputs=restore_outputs)
        components['trained_params']['accordion'].expand(restore_fn, outputs=restore_outputs)
        if 'accordion' in components.get('vv_trained_params', {}):
            components['vv_trained_params']['accordion'].expand(restore_fn, outputs=restore_outputs)

        # ICL toggle: show/hide voice sample section
        components['icl_enabled'].change(
            lambda enabled: gr.update(visible=enabled),
            inputs=[components['icl_enabled']],
            outputs=[components['icl_sample_section']]
        )

        def get_selected_icl_filename(lister_value):
            """Extract selected filename from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                return selected[0]
            return None

        # ICL dataset change: update sample lister
        def update_icl_samples(folder):
            """Update ICL sample lister when dataset changes."""
            files = get_dataset_files(folder)
            return files, None

        components['icl_dataset_dropdown'].change(
            update_icl_samples,
            inputs=[components['icl_dataset_dropdown']],
            outputs=[components['icl_voice_lister'], components['icl_audio_preview']]
        )

        # ICL refresh datasets
        components['icl_refresh_datasets'].click(
            lambda: gr.update(choices=["(Select Dataset)"] + get_dataset_folders(), value="(Select Dataset)"),
            outputs=[components['icl_dataset_dropdown']]
        )

        # ICL sample preview on selection
        def load_icl_audio_preview(lister_value, folder):
            """Load ICL audio preview from FileLister selection."""
            filename = get_selected_icl_filename(lister_value)
            if not folder or not filename or folder in ("(No folders)", "(Select Dataset)"):
                return None
            audio_path = DATASETS_DIR / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['icl_voice_lister'].change(
            load_icl_audio_preview,
            inputs=[components['icl_voice_lister'], components['icl_dataset_dropdown']],
            outputs=[components['icl_audio_preview']]
        )

        # Double-click = play preview
        components['icl_voice_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#icl-audio-preview .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # Apply emotion preset to Trained Model parameters
        if 'update_from_emotion' in components.get('trained_params', {}):
            components['trained_emotion_preset'].change(
                components['trained_params']['update_from_emotion'],
                inputs=[components['trained_emotion_preset'], components['trained_emotion_intensity']],
                outputs=[components['trained_temperature'], components['trained_top_p'], components['trained_repetition_penalty']]
            )

            components['trained_emotion_intensity'].change(
                components['trained_params']['update_from_emotion'],
                inputs=[components['trained_emotion_preset'], components['trained_emotion_intensity']],
                outputs=[components['trained_temperature'], components['trained_top_p'], components['trained_repetition_penalty']]
            )

        # Emotion management buttons
        components['trained_save_emotion_btn'].click(
            fn=None,
            inputs=[components['trained_emotion_preset']],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="trained_emotion_"
            )
        )

        def handle_trained_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Presets emotion save."""
            if not input_value or not input_value.startswith("trained_emotion_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), ""
                emotion_name = "_".join(parts[2:-1])

                # Use shared helper to process save result
                save_result = save_emotion_handler(emotion_name, intensity, temp, rep_pen, top_p)
                return process_save_emotion_result(save_result, shared_state)

            return gr.update(), gr.update()

        components['trained_delete_emotion_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="trained_emotion_"
            )
        )

        def _disable_gen_btn():
            return gr.update(interactive=False)

        def _enable_gen_btn():
            return gr.update(interactive=True)

        components['custom_generate_btn'].click(
            _disable_gen_btn, outputs=[components['custom_generate_btn']]
        ).then(
            restore_fn, outputs=restore_outputs
        ).then(
            generate_with_voice_type,
            inputs=[
                components['custom_text_input'], components['custom_language'], components['custom_speaker_dropdown'],
                components['custom_instruct_input'], components['custom_seed'], components['custom_model_size'],
                components['voice_type_radio'], components['custom_speaker_dropdown'], components['trained_model_dropdown'],
                components['custom_do_sample'], components['custom_temperature'], components['custom_top_k'], components['custom_top_p'],
                components['custom_repetition_penalty'], components['custom_max_new_tokens'],
                components['trained_do_sample'], components['trained_temperature'], components['trained_top_k'], components['trained_top_p'],
                components['trained_repetition_penalty'], components['trained_max_new_tokens'],
                components['icl_enabled'], components['icl_dataset_dropdown'], components['icl_voice_lister'],
                # VibeVoice Trained
                components['vv_trained_model_dropdown'],
                components['vv_trained_cfg_scale'], components['vv_trained_num_steps'],
                components['vv_trained_do_sample'], components['vv_trained_repetition_penalty'],
                components['vv_trained_temperature'], components['vv_trained_top_k'], components['vv_trained_top_p'],
                # VibeVoice Streaming
                components['vv_streaming_voice_dropdown'],
                components['vv_streaming_cfg_scale'], components['vv_streaming_ddpm_steps'],
            ],
            outputs=[components['custom_output_audio'], components['preset_status'], components['_result_metadata'], components['save_result_btn']]
        ).then(
            _enable_gen_btn, outputs=[components['custom_generate_btn']]
        )

        # Save result button handler
        def save_result_handler(audio_path, metadata_text):
            """Save the temp result to output folder in chosen format."""
            if not audio_path:
                return "❌ No audio to save.", gr.update()
            try:
                output_format = user_config.get("output_format", "wav")
                output_path = save_result_to_output(audio_path, OUTPUT_DIR, output_format, metadata_text or None)
                return f"Saved to: {output_path.name}", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Error saving: {str(e)}", gr.update()

        components['save_result_btn'].click(
            save_result_handler,
            inputs=[components['custom_output_audio'], components['_result_metadata']],
            outputs=[components['preset_status'], components['save_result_btn']]
        )

        def delete_trained_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches trained_emotion_."""
            if not confirm_value or not confirm_value.startswith("trained_emotion_"):
                return gr.update(), gr.update()

            # Call the delete handler and discard the clear_trigger (3rd value)
            delete_result = delete_emotion_handler(confirm_value, emotion_name)
            dropdown_update, status_msg, _clear = process_delete_emotion_result(delete_result, shared_state)
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_trained_emotion_wrapper,
            inputs=[confirm_trigger, components['trained_emotion_preset']],
            outputs=[components['trained_emotion_preset'], components['preset_status']]
        )

        input_trigger.change(
            handle_trained_emotion_input,
            inputs=[input_trigger, components['trained_emotion_intensity'], components['trained_temperature'], components['trained_repetition_penalty'], components['trained_top_p']],
            outputs=[components['trained_emotion_preset'], components['preset_status']]
        )

        # Refresh emotion dropdowns when tab is selected
        components['voice_presets_tab'].select(
            lambda: gr.update(choices=shared_state['get_emotion_choices'](shared_state['_active_emotions'])),
            outputs=[components['trained_emotion_preset']]
        )

        # Save preferences when settings change
        components['custom_model_size'].change(
            lambda x: save_preference("custom_voice_size", x),
            inputs=[components['custom_model_size']],
            outputs=[]
        )

        components['custom_language'].change(
            lambda x: save_preference("language", x),
            inputs=[components['custom_language']],
            outputs=[]
        )

        # voice_type preference is saved via .then() on line 873+

        # --- Cross-tab prompt routing ---
        import modules.core_components.prompt_hub as _prompt_hub
        _prompt_hub.wire_prompt_loader(components, "vp", {"voice_presets.text": components['custom_text_input']})

        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        if prompt_apply_trigger is not None:

            def _apply_vp_text(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'voice_presets.text':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_vp_text,
                inputs=[prompt_apply_trigger, components['custom_text_input']],
                outputs=[components['custom_text_input']],
            )

            def _apply_vp_style(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'voice_presets.style':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_vp_style,
                inputs=[prompt_apply_trigger, components['custom_instruct_input']],
                outputs=[components['custom_instruct_input']],
            )

# Export for tab registry
get_tool_class = lambda: VoicePresetsTool

if __name__ == "__main__":
    """Standalone testing of Voice Presets tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoicePresetsTool, port=7863, title="Voice Presets - Standalone")
