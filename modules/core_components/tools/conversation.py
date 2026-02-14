"""
Conversation Tab

Create conversations using VibeVoice, Qwen Base, or Qwen Speakers.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))
import gradio as gr
import soundfile as sf
import torch
import numpy as np
import random
import re
from datetime import datetime
from pathlib import Path
from modules.core_components.ai_models.model_utils import set_seed, get_device
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager


class ConversationTool(Tool):
    """Conversation tool implementation."""

    config = ToolConfig(
        name="Conversation",
        module_name="tool_conversation",
        description="Create multi-speaker conversations",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Conversation tool UI."""
        components = {}

        # Get helper functions and config
        format_help_html = shared_state['format_help_html']
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
        create_qwen_advanced_params = shared_state['create_qwen_advanced_params']
        create_luxtts_advanced_params = shared_state['create_luxtts_advanced_params']
        create_emotion_intensity_slider = shared_state['create_emotion_intensity_slider']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        MODEL_SIZES_CUSTOM = shared_state['MODEL_SIZES_CUSTOM']
        MODEL_SIZES_BASE = shared_state['MODEL_SIZES_BASE']
        MODEL_SIZES_VIBEVOICE = shared_state['MODEL_SIZES_VIBEVOICE']
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        _active_emotions = shared_state.get('_active_emotions', {})
        TTS_ENGINES = shared_state.get('TTS_ENGINES', {})

        # Mapping: conversation radio label -> engine key in TTS_ENGINES
        CONV_ENGINE_MAP = {
            "VibeVoice": "VibeVoice",
            "Qwen Base": "Qwen3",
            "Qwen Speakers": "Qwen3",
            "Chatterbox": "Chatterbox",
            "LuxTTS": "LuxTTS",
        }
        ALL_CONV_CHOICES = ["VibeVoice", "Qwen Base", "Qwen Speakers", "Chatterbox", "LuxTTS"]

        # Filter conversation choices based on enabled engines
        engine_settings = _user_config.get("enabled_engines", {})
        visible_choices = []
        for choice in ALL_CONV_CHOICES:
            engine_key = CONV_ENGINE_MAP.get(choice)
            if engine_key:
                engine_info = TTS_ENGINES.get(engine_key, {})
                if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                    visible_choices.append(choice)
        if not visible_choices:
            visible_choices = ALL_CONV_CHOICES  # Safety fallback

        # Model selector at top
        initial_conv_model = _user_config.get("conv_model_type", "VibeVoice")
        if initial_conv_model not in visible_choices:
            initial_conv_model = visible_choices[0]
        is_vibevoice = initial_conv_model == "VibeVoice"
        is_qwen_base = initial_conv_model == "Qwen Base"
        is_qwen_custom = initial_conv_model == "Qwen Speakers"
        is_luxtts = initial_conv_model == "LuxTTS"
        is_chatterbox = initial_conv_model == "Chatterbox"

        with gr.TabItem("Conversation") as conv_tab:
            components['conv_tab'] = conv_tab
            gr.Markdown("Choose a model and create multi-speaker conversations with your custom voices")
            components['conv_model_type'] = gr.Radio(
                choices=visible_choices,
                value=initial_conv_model,
                show_label=False,
                container=False
            )

            # Get sample choices once for all dropdowns
            conversation_available_samples = get_sample_choices()
            conversation_first_sample = conversation_available_samples[0] if conversation_available_samples else None

            with gr.Row():
                # Left - Script input and model-specific controls
                with gr.Column(scale=2):
                    gr.Markdown("### Conversation Script")

                    components['conversation_script'] = gr.Textbox(
                        label="Script:",
                        placeholder=dedent("""\
                            Use [N]: format for speaker labels (1-4 for VibeVoice, 1-8 for Base/LuxTTS, 1-9 for CustomVoice).
                            Qwen also supports (style) for emotions:

                            [1]: (cheerful) Hey, how's it going?
                            [2]: (excited) I'm doing great, thanks for asking!
                            [1]: That's wonderful to hear.
                            [3]: (curious) Mind if I join this conversation?

                            VibeVoice: Natural long-form generation.
                            Base: Your custom voice clips with advanced pause control, with hacked Style control.
                            CustomVoice: Qwen Preset speakers with style control and Pause Controls.
                            LuxTTS: Voice cloning with custom samples, sequential generation."""),
                        lines=18
                    )

                    # Qwen speaker mapping
                    speaker_guide = dedent("""\
                        **Qwen Speaker Numbers → Preset Voices:**

                        | # | Speaker | Voice | Language |   | # | Speaker | Voice | Language |
                        |---|---------|-------|----------|---|---|---------|-------|----------|
                        | 1 | Vivian | Bright young female | 🇨🇳 Chinese |   | 6 | Ryan | Dynamic male | 🇺🇸 English |
                        | 2 | Serena | Warm gentle female | 🇨🇳 Chinese |   | 7 | Aiden | Sunny American male | 🇺🇸 English |
                        | 3 | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |   | 8 | Ono_Anna | Playful female | 🇯🇵 Japanese |
                        | 4 | Dylan | Youthful Beijing male | 🇨🇳 Chinese |   | 9 | Sohee | Warm female | 🇰🇷 Korean |
                        | 5 | Eric | Lively Chengdu male | 🇨🇳 Chinese |  |  |  |  |  |

                        *Each speaker works best in their native language.*
                        """)

                    components['qwen_speaker_table'] = gr.HTML(
                        value=format_help_html(speaker_guide),
                        container=True,
                        padding=True,
                        visible=is_qwen_custom
                    )

                    # Shared voice sample selectors (used by Qwen Base, LuxTTS, Chatterbox, VibeVoice)
                    uses_samples = is_qwen_base or is_luxtts or is_chatterbox or is_vibevoice
                    components['shared_voices_section'] = gr.Column(visible=uses_samples)
                    with components['shared_voices_section']:
                        gr.Markdown("### Voice Samples (Up to 8 Speakers)")

                        with gr.Row():
                            with gr.Column():
                                components['conv_voice_1'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[1] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_2'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[2] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        with gr.Row():
                            with gr.Column():
                                components['conv_voice_3'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[3] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_4'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[4] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        # Rows 5-8 hidden for VibeVoice (only supports 4 speakers)
                        components['conv_voices_row_5_6'] = gr.Row(visible=not is_vibevoice)
                        with components['conv_voices_row_5_6']:
                            with gr.Column():
                                components['conv_voice_5'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[5] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_6'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[6] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        components['conv_voices_row_7_8'] = gr.Row(visible=not is_vibevoice)
                        with components['conv_voices_row_7_8']:
                            with gr.Column():
                                components['conv_voice_7'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[7] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_8'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[8] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                # Right - Settings and output
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")

                    # Qwen Speakers settings
                    components['qwen_custom_settings'] = gr.Column(visible=is_qwen_custom)
                    with components['qwen_custom_settings']:
                        components['conv_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_CUSTOM,
                            value=_user_config.get("conv_model_size", "Large"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                    # Qwen Base settings
                    components['qwen_base_settings'] = gr.Column(visible=is_qwen_base)
                    with components['qwen_base_settings']:
                        components['conv_base_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_BASE,
                            value=_user_config.get("conv_base_model_size", "Small"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                    # Shared Language and Seed
                    is_qwen_initial = is_qwen_base or is_qwen_custom
                    from modules.core_components.constants import CHATTERBOX_LANGUAGES
                    if is_qwen_initial:
                        initial_lang_choices = LANGUAGES
                        initial_lang_value = _user_config.get("language", "Auto")
                    elif is_chatterbox:
                        initial_lang_choices = CHATTERBOX_LANGUAGES
                        initial_lang_value = "English"
                    else:
                        initial_lang_choices = ["Auto"]
                        initial_lang_value = "Auto"
                    with gr.Column():
                        with gr.Row():
                            components['conv_language'] = gr.Dropdown(
                                scale=5,
                                choices=initial_lang_choices,
                                value=initial_lang_value,
                                label="Language",
                                info="Language for all lines (Auto recommended)"
                            )
                            components['conv_seed'] = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="(-1 for random)"
                            )

                    # Shared Pause Controls
                    components['qwen_pause_controls'] = gr.Accordion("Pause Controls", open=False, visible=(is_qwen_custom or is_qwen_base))
                    with components['qwen_pause_controls']:
                        with gr.Column():
                            components['conv_pause_linebreak'] = gr.Slider(
                                minimum=0.0,
                                maximum=3.0,
                                value=_user_config.get("conv_pause_linebreak", 0.25),
                                step=0.05,
                                label="Pause Between Lines",
                                info="Silence between each speaker turn"
                            )

                            with gr.Row():
                                components['conv_pause_period'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_period", 0.4),
                                    step=0.1,
                                    label="After Period (.)",
                                    info="Pause after periods"
                                )
                                components['conv_pause_comma'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_comma", 0.2),
                                    step=0.1,
                                    label="After Comma (,)",
                                    info="Pause after commas"
                                )

                            with gr.Row():
                                components['conv_pause_question'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_question", 0.6),
                                    step=0.1,
                                    label="After Question (?)",
                                    info="Pause after questions"
                                )
                                components['conv_pause_hyphen'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_hyphen", 0.3),
                                    step=0.1,
                                    label="After Hyphen (-)",
                                    info="Pause after hyphens"
                                )

                    # Chatterbox-specific settings
                    components['cb_settings'] = gr.Column(visible=is_chatterbox)
                    with components['cb_settings']:
                        # Pause between lines
                        components['cb_pause_linebreak'] = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.25,
                            step=0.05,
                            label="Pause Between Lines",
                            info="Silence between each speaker turn"
                        )

                        # Chatterbox Advanced Parameters
                        create_chatterbox_advanced_params = shared_state['create_chatterbox_advanced_params']
                        cb_conv_params = create_chatterbox_advanced_params(
                            visible=True
                        )
                        components['cb_conv_exaggeration'] = cb_conv_params['exaggeration']
                        components['cb_conv_cfg_weight'] = cb_conv_params['cfg_weight']
                        components['cb_conv_temperature'] = cb_conv_params['temperature']
                        components['cb_conv_repetition_penalty'] = cb_conv_params['repetition_penalty']
                        components['cb_conv_top_p'] = cb_conv_params['top_p']
                        components['cb_conv_accordion'] = cb_conv_params.get('accordion')

                    # LuxTTS-specific settings
                    components['luxtts_settings'] = gr.Column(visible=is_luxtts)
                    with components['luxtts_settings']:
                        # Pause between lines
                        components['luxtts_pause_linebreak'] = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.25,
                            step=0.05,
                            label="Pause Between Lines",
                            info="Silence between each speaker turn"
                        )

                        # LuxTTS Advanced Parameters
                        lux_conv_params = create_luxtts_advanced_params(
                            visible=True
                        )
                        components['lux_conv_num_steps'] = lux_conv_params['num_steps']
                        components['lux_conv_t_shift'] = lux_conv_params['t_shift']
                        components['lux_conv_speed'] = lux_conv_params['speed']
                        components['lux_conv_guidance_scale'] = lux_conv_params['guidance_scale']
                        components['lux_conv_rms'] = lux_conv_params['rms']
                        components['lux_conv_ref_duration'] = lux_conv_params['ref_duration']
                        components['lux_conv_return_smooth'] = lux_conv_params['return_smooth']
                        components['lux_conv_accordion'] = lux_conv_params.get('accordion')

                    # VibeVoice-specific settings
                    components['vibevoice_settings'] = gr.Column(visible=is_vibevoice)
                    with components['vibevoice_settings']:
                        components['longform_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_VIBEVOICE,
                            value=_user_config.get("vibevoice_model_size", "Large"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                        # VibeVoice Advanced Parameters
                        vv_conv_params = create_vibevoice_advanced_params(
                            include_sentences_per_chunk=True,
                            visible=True
                        )
                        components['vv_conv_num_steps'] = vv_conv_params['num_steps']
                        components['longform_cfg_scale'] = vv_conv_params['cfg_scale']
                        components['vv_conv_do_sample'] = vv_conv_params['do_sample']
                        components['vv_conv_sentences_per_chunk'] = vv_conv_params['sentences_per_chunk']
                        components['vv_conv_repetition_penalty'] = vv_conv_params['repetition_penalty']
                        components['vv_conv_temperature'] = vv_conv_params['temperature']
                        components['vv_conv_top_k'] = vv_conv_params['top_k']
                        components['vv_conv_top_p'] = vv_conv_params['top_p']
                        components['vv_conv_accordion'] = vv_conv_params.get('accordion')

                    # Qwen Custom Voice Advanced Parameters
                    components['qwen_custom_conv_advanced'] = gr.Column(visible=is_qwen_custom)
                    with components['qwen_custom_conv_advanced']:
                        qwen_custom_conv_params = create_qwen_advanced_params(
                            include_emotion=False,
                            visible=True
                        )
                        components['qwen_custom_conv_do_sample'] = qwen_custom_conv_params['do_sample']
                        components['qwen_custom_conv_temperature'] = qwen_custom_conv_params['temperature']
                        components['qwen_custom_conv_top_k'] = qwen_custom_conv_params['top_k']
                        components['qwen_custom_conv_top_p'] = qwen_custom_conv_params['top_p']
                        components['qwen_custom_conv_repetition_penalty'] = qwen_custom_conv_params['repetition_penalty']
                        components['qwen_custom_conv_max_new_tokens'] = qwen_custom_conv_params['max_new_tokens']
                        components['qwen_custom_conv_accordion'] = qwen_custom_conv_params.get('accordion')

                    # Qwen Base Advanced Parameters
                    components['qwen_base_conv_advanced'] = gr.Column(visible=is_qwen_base)
                    with components['qwen_base_conv_advanced']:
                        # Emotion intensity slider (Base only)
                        components['conv_emotion_intensity_row'] = gr.Row()
                        with components['conv_emotion_intensity_row']:
                            components['conv_emotion_intensity'] = create_emotion_intensity_slider(
                                initial_intensity=1.0,
                                label="Emotion Intensity",
                                visible=True
                            )

                        qwen_base_conv_params = create_qwen_advanced_params(
                            include_emotion=False,
                            visible=True
                        )
                        components['qwen_base_conv_do_sample'] = qwen_base_conv_params['do_sample']
                        components['qwen_base_conv_temperature'] = qwen_base_conv_params['temperature']
                        components['qwen_base_conv_top_k'] = qwen_base_conv_params['top_k']
                        components['qwen_base_conv_top_p'] = qwen_base_conv_params['top_p']
                        components['qwen_base_conv_repetition_penalty'] = qwen_base_conv_params['repetition_penalty']
                        components['qwen_base_conv_max_new_tokens'] = qwen_base_conv_params['max_new_tokens']
                        components['qwen_base_conv_accordion'] = qwen_base_conv_params.get('accordion')

                    # Shared settings
                    components['conv_generate_btn'] = gr.Button("Generate Conversation", variant="primary", size="lg")

                    components['conv_output_audio'] = gr.Audio(
                        label="Generated Conversation",
                        type="filepath"
                    )

                    # Save button — visible only when "Review Before Saving" is enabled
                    manual_save = _user_config.get("manual_save", False)
                    components['conv_save_result_btn'] = gr.Button(
                        "Save to Output", variant="primary", size="lg",
                        visible=manual_save, interactive=False
                    )
                    components['_conv_result_metadata'] = gr.Textbox(visible=False)

                    components['conv_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

                    # Model-specific tips
                    qwen_custom_tips_text = dedent("""\
                    **Qwen Speakers Tips:**
                    - Fast generation with preset voices
                    - Up to 9 different speakers
                    - Tip: Use `[break=1.5]` inline for custom pauses
                    - Each voice optimized for their native language
                    - Style instructions: (cheerful), (sad), (excited), etc.
                    """)

                    qwen_base_tips_text = dedent("""\
                    **Qwen Base Tips:**
                    - Use your own custom voice samples
                    - Up to 8 different speakers
                    - Tip: Use `[break=1.5]` inline for custom pauses
                    - Advanced pause control (periods, commas, questions, hyphens)
                    - Prepare 3-10 second voice samples in samples/ folder
                    """)

                    vibevoice_tips_text = dedent("""\
                    **VibeVoice Tips:**
                    - Up to 90 minutes continuous generation
                    - Up to 4 speakers with custom voices
                    - May spontaneously add background music/sounds
                    - Longer scripts work best with Large model
                    - Natural conversation flow (no manual pause control)
                    """)

                    luxtts_tips_text = dedent("""\
                    **LuxTTS Tips:**
                    - Up to 8 speakers with custom voice samples
                    - Sequential generation: each line generated individually then stitched
                    - Voice prompts are cached per speaker for fast subsequent lines
                    - 48kHz output for high quality audio
                    - No style/emotion control (use Qwen for that)
                    - Adjust ref_duration if voice quality is poor
                    """)

                    components['qwen_custom_tips'] = gr.HTML(
                        value=format_help_html(qwen_custom_tips_text),
                        container=True,
                        padding=True,
                        visible=is_qwen_custom
                    )

                    components['qwen_base_tips'] = gr.HTML(
                        value=format_help_html(qwen_base_tips_text),
                        container=True,
                        padding=True,
                        visible=is_qwen_base
                    )

                    components['vibevoice_tips'] = gr.HTML(
                        value=format_help_html(vibevoice_tips_text),
                        container=True,
                        padding=True,
                        visible=is_vibevoice
                    )

                    components['luxtts_tips'] = gr.HTML(
                        value=format_help_html(luxtts_tips_text),
                        container=True,
                        padding=True,
                        visible=is_luxtts
                    )

                    chatterbox_tips_text = dedent("""\
                    **Chatterbox Tips:**
                    - Up to 8 speakers with custom voice samples
                    - English uses fast default TTS model
                    - Other languages use Multilingual model (23 languages)
                    - Sequential generation: each line generated then stitched
                    - Exaggeration controls emotion intensity (0=neutral, 2=max)
                    - No text transcript needed for voice samples
                    """)

                    components['cb_tips'] = gr.HTML(
                        value=format_help_html(chatterbox_tips_text),
                        container=True,
                        padding=True,
                        visible=is_chatterbox
                    )

            return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Conversation tab events."""

        # Get helper functions
        get_available_samples = shared_state['get_available_samples']
        get_sample_choices = shared_state['get_sample_choices']
        save_preference = shared_state['save_preference']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        TEMP_DIR = shared_state['TEMP_DIR']
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        LANGUAGES = shared_state['LANGUAGES']
        _active_emotions = shared_state.get('_active_emotions', {})
        play_completion_beep = shared_state.get('play_completion_beep')
        get_or_create_voice_prompt = shared_state.get('get_or_create_voice_prompt')
        save_audio_to_temp = shared_state['save_audio_to_temp']
        save_result_to_output = shared_state['save_result_to_output']

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        # Wire param persistence (auto-save on change)
        wire_param_persistence = shared_state['wire_param_persistence']
        _user_config = shared_state['_user_config']
        param_map = {
            'qwen_custom': [
                ('qwen_custom_conv_do_sample', 'do_sample'),
                ('qwen_custom_conv_temperature', 'temperature'),
                ('qwen_custom_conv_top_k', 'top_k'),
                ('qwen_custom_conv_top_p', 'top_p'),
                ('qwen_custom_conv_repetition_penalty', 'repetition_penalty'),
                ('qwen_custom_conv_max_new_tokens', 'max_new_tokens'),
            ],
            'qwen_base': [
                ('conv_emotion_intensity', 'emotion_intensity'),
                ('qwen_base_conv_do_sample', 'do_sample'),
                ('qwen_base_conv_temperature', 'temperature'),
                ('qwen_base_conv_top_k', 'top_k'),
                ('qwen_base_conv_top_p', 'top_p'),
                ('qwen_base_conv_repetition_penalty', 'repetition_penalty'),
                ('qwen_base_conv_max_new_tokens', 'max_new_tokens'),
            ],
            'vibevoice': [
                ('longform_cfg_scale', 'cfg_scale'),
                ('vv_conv_num_steps', 'num_steps'),
                ('vv_conv_do_sample', 'do_sample'),
                ('vv_conv_sentences_per_chunk', 'sentences_per_chunk'),
                ('vv_conv_repetition_penalty', 'repetition_penalty'),
                ('vv_conv_temperature', 'temperature'),
                ('vv_conv_top_k', 'top_k'),
                ('vv_conv_top_p', 'top_p'),
            ],
            'luxtts': [
                ('lux_conv_num_steps', 'num_steps'),
                ('lux_conv_t_shift', 't_shift'),
                ('lux_conv_speed', 'speed'),
                ('lux_conv_return_smooth', 'return_smooth'),
                ('lux_conv_rms', 'rms'),
                ('lux_conv_ref_duration', 'ref_duration'),
                ('lux_conv_guidance_scale', 'guidance_scale'),
            ],
            'chatterbox': [
                ('cb_conv_exaggeration', 'exaggeration'),
                ('cb_conv_cfg_weight', 'cfg_weight'),
                ('cb_conv_temperature', 'temperature'),
                ('cb_conv_repetition_penalty', 'repetition_penalty'),
                ('cb_conv_top_p', 'top_p'),
            ],
        }
        wire_param_persistence(components, _user_config, param_map)

        # Create restore handler for applying saved params on tab select / model switch
        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, _user_config, param_map)

        def finalize_conversation_output(final_audio, sr, filename_stem, metadata_out, status_msg, progress):
            """Save conversation audio to temp, then auto-save or wait for manual save."""
            temp_path = save_audio_to_temp(final_audio, sr, TEMP_DIR, filename_stem)
            manual_save = _user_config.get("manual_save", False)
            if manual_save:
                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(temp_path), status_msg + "\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
            else:
                output_format = _user_config.get("output_format", "wav")
                output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(output_path), status_msg, "", gr.update()

        def prepare_voice_samples_dict(v1, v2=None, v3=None, v4=None, v5=None, v6=None, v7=None, v8=None):
            """Prepare voice samples dictionary for generation."""
            from modules.core_components.tools import strip_sample_extension
            samples = {}
            available_samples = get_available_samples()

            voice_inputs = [("Speaker1", v1), ("Speaker2", v2), ("Speaker3", v3), ("Speaker4", v4),
                            ("Speaker5", v5), ("Speaker6", v6), ("Speaker7", v7), ("Speaker8", v8)]

            for speaker_num, sample_name in voice_inputs:
                if sample_name:
                    bare_name = strip_sample_extension(sample_name)
                    for s in available_samples:
                        if s["name"] == bare_name:
                            samples[speaker_num] = {
                                "wav_path": s["wav_path"],
                                "ref_text": s["ref_text"],
                                "name": s["name"]
                            }
                            break
            return samples

        def validate_samples_have_transcripts(voice_samples_dict):
            """Check that all voice samples have transcripts. Returns error message or None."""
            missing = []
            for speaker_key, data in voice_samples_dict.items():
                ref_text = data.get("ref_text", "").strip()
                if not ref_text:
                    name = data.get("name", speaker_key)
                    missing.append(name)
            if missing:
                names = ", ".join(missing)
                return (
                    f"No transcript found for: {names}.\n\n"
                    "Please transcribe these samples first in the **Prep Audio** tab "
                    "(using Whisper or VibeVoice ASR), then try again."
                )
            return None

        def preprocess_conversation_script(script):
            """Add [1]: to lines without speaker labels."""
            lines = script.strip().split('\n')
            processed_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line already has a speaker label [N]: or [Speaker N]:
                if line.startswith('[') and ']:' in line:
                    processed_lines.append(line)
                else:
                    # Add default [1]: label
                    processed_lines.append(f"[1]: {line}")
            return '\n'.join(processed_lines)

        def extract_style_instructions(text):
            """Extract style instructions from parentheses."""
            import re
            instructions = re.findall(r'\(([^)]+)\)', text)
            clean_text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)
            clean_text = ' '.join(clean_text.split())
            combined_instruct = ', '.join(instructions) if instructions else ''
            return clean_text, combined_instruct

        def generate_conversation_handler(conversation_data, pause_linebreak, pause_period, pause_comma,
                                          pause_question, pause_hyphen, language, seed, model_size,
                                          do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                          progress=gr.Progress()):
            """Generate multi-speaker conversation with Qwen Speakers preset speakers."""
            if not conversation_data or not conversation_data.strip():
                return None, "❌ Please enter conversation lines.", "", gr.update()

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse conversation lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= len(CUSTOM_VOICE_SPEAKERS):
                                speaker = CUSTOM_VOICE_SPEAKERS[speaker_num - 1]
                                if text:
                                    lines.append((speaker, text))

                if not lines:
                    return None, "❌ No valid conversation lines found. Use format: [N]: Text", "", gr.update()

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)

                progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")
                model = tts_manager.get_qwen3_custom_voice(model_size)

                # Generate all lines with pause control
                all_segments = []
                sr = None
                pause_pattern = re.compile(r'\[break=([\d\.]+)\]')

                for i, (speaker, text) in enumerate(lines):
                    progress_val = 0.1 + (0.8 * i / len(lines))
                    clean_text, style_instruct = extract_style_instructions(text)

                    # Insert pause markers
                    if pause_period > 0:
                        clean_text = re.sub(r'\.(?!\d)', f'. [break={pause_period}]', clean_text)
                    if pause_comma > 0:
                        clean_text = re.sub(r',(?!\d)', f', [break={pause_comma}]', clean_text)
                    if pause_question > 0:
                        clean_text = re.sub(r'\?(?!\d)', f'? [break={pause_question}]', clean_text)
                    if pause_hyphen > 0:
                        clean_text = re.sub(r'-(?!\d)', f'- [break={pause_hyphen}]', clean_text)

                    parts = pause_pattern.split(clean_text)

                    if style_instruct:
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} [{style_instruct[:15]}...]")
                    else:
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker})")

                    # Generate segments
                    for j in range(0, len(parts), 2):
                        segment_text = parts[j].strip()
                        if not segment_text:
                            continue
                        segment_text = pause_pattern.sub('', segment_text).strip()
                        if not segment_text:
                            continue

                        kwargs = {
                            "text": segment_text,
                            "language": language if language != "Auto" else "Auto",
                            "speaker": speaker,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                            "repetition_penalty": repetition_penalty,
                            "max_new_tokens": max_new_tokens
                        }
                        if style_instruct:
                            kwargs["instruct"] = style_instruct

                        wavs, sr = model.generate_custom_voice(**kwargs)

                        segment_pause = 0.0
                        if j + 1 < len(parts):
                            try:
                                segment_pause = float(parts[j + 1])
                            except ValueError:
                                pass

                        all_segments.append((wavs[0], segment_pause))

                    # Add linebreak pause
                    if i < len(lines) - 1 and all_segments:
                        last_wav, last_pause = all_segments[-1]
                        all_segments[-1] = (last_wav, last_pause + pause_linebreak)

                # Concatenate
                progress(0.9, desc="Stitching conversation...")
                conversation_audio = []
                for wav, pause_duration in all_segments:
                    conversation_audio.append(wav)
                    if pause_duration > 0:
                        pause_samples = int(sr * pause_duration)
                        conversation_audio.append(np.zeros(pause_samples))

                final_audio = np.concatenate(conversation_audio)

                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_stem = f"conversation_qwen3_{timestamp}"
                speakers_used = list(set(s for s, _ in lines))
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Qwen3-TTS Conversation
                    Model: CustomVoice {model_size}
                    Language: {language}
                    Seed: {seed}
                    Pause Settings:
                      - Linebreak: {pause_linebreak}s
                      - Period: {pause_period}s
                      - Comma: {pause_comma}s
                      - Question: {pause_question}s
                      - Hyphen: {pause_hyphen}s
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {len(all_segments)}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                duration = len(final_audio) / sr
                status_msg = f"Conversation generated.\n{len(lines)} lines | {duration:.1f}s | Seed: {seed} | {model_size}"
                return finalize_conversation_output(final_audio, sr, filename_stem, metadata_out, status_msg, progress)

            except Exception as e:
                return None, f"❌ Error generating conversation: {str(e)}", "", gr.update()

        def generate_conversation_base_handler(conversation_data, voice_samples_dict, pause_linebreak,
                                               pause_period, pause_comma, pause_question, pause_hyphen,
                                               language, seed, model_size, do_sample, temperature, top_k,
                                               top_p, repetition_penalty, max_new_tokens, emotion_intensity,
                                               progress=gr.Progress()):
            """Generate multi-speaker conversation with Qwen Base + custom voice samples."""
            if not conversation_data or not conversation_data.strip():
                return None, "❌ Please enter conversation lines.", "", gr.update()

            if not voice_samples_dict:
                return None, "❌ Please select at least one voice sample.", "", gr.update()

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data["ref_text"], text))

                if not lines:
                    return None, "❌ No valid conversation lines found. Use format: [N]: Text (N=1-8)", "", gr.update()

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)


                progress(0.1, desc=f"Loading Base model ({model_size})...")
                model = tts_manager.get_qwen3_base(model_size)

                # Generate all segments
                all_segments = []
                sr = None
                pause_pattern = re.compile(r'\[break=([\d\.]+)\]')

                for i, (speaker_key, voice_sample_path, ref_text, text) in enumerate(lines):
                    progress_val = 0.1 + (0.8 * i / len(lines))
                    clean_text, detected_emotion = extract_style_instructions(text)

                    # Apply emotion adjustments
                    emotion_key = detected_emotion.lower().replace(" ", "_").replace(",", "").strip() if detected_emotion else None
                    line_temp = temperature
                    line_top_p = top_p
                    line_rep_pen = repetition_penalty

                    if emotion_key and emotion_key in _active_emotions:
                        adjustments = _active_emotions[emotion_key]
                        line_temp = max(0.1, min(2.0, temperature + (adjustments["temp"] * emotion_intensity)))
                        line_top_p = max(0.0, min(1.0, top_p + (adjustments["top_p"] * emotion_intensity)))
                        line_rep_pen = max(1.0, min(2.0, repetition_penalty + (adjustments["penalty"] * emotion_intensity)))
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key}) [{emotion_key}]")
                    else:
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                    text = clean_text

                    # Insert pause markers
                    if pause_period > 0:
                        text = re.sub(r'\.(?!\d)', f'. [break={pause_period}]', text)
                    if pause_comma > 0:
                        text = re.sub(r',(?!\d)', f', [break={pause_comma}]', text)
                    if pause_question > 0:
                        text = re.sub(r'\?(?!\d)', f'? [break={pause_question}]', text)
                    if pause_hyphen > 0:
                        text = re.sub(r'-(?!\d)', f'- [break={pause_hyphen}]', text)

                    parts = pause_pattern.split(text)

                    # Get voice prompt (cached if available)
                    voice_prompt = None
                    if get_or_create_voice_prompt:
                        voice_prompt = get_or_create_voice_prompt(model, speaker_key, voice_sample_path, ref_text, model_size)

                    # Generate segments
                    for j in range(0, len(parts), 2):
                        segment_text = parts[j].strip()
                        if not segment_text:
                            continue
                        segment_text = pause_pattern.sub('', segment_text).strip()
                        if not segment_text:
                            continue

                        wavs, sr = model.generate_voice_clone(
                            text=segment_text,
                            language=language if language != "Auto" else "auto",
                            ref_audio=voice_sample_path,
                            ref_text=ref_text,
                            voice_prompt=voice_prompt,
                            do_sample=do_sample,
                            temperature=line_temp,
                            top_k=top_k,
                            top_p=line_top_p,
                            repetition_penalty=line_rep_pen,
                            max_new_tokens=max_new_tokens
                        )

                        segment_pause = 0.0
                        if j + 1 < len(parts):
                            try:
                                segment_pause = float(parts[j + 1])
                            except ValueError:
                                pass

                        all_segments.append((wavs[0], segment_pause))

                    # Add linebreak pause
                    if i < len(lines) - 1 and all_segments:
                        last_wav, last_pause = all_segments[-1]
                        all_segments[-1] = (last_wav, last_pause + pause_linebreak)

                # Concatenate
                progress(0.9, desc="Stitching conversation...")
                conversation_audio = []
                for wav, pause_duration in all_segments:
                    conversation_audio.append(wav)
                    if pause_duration > 0:
                        pause_samples = int(sr * pause_duration)
                        conversation_audio.append(np.zeros(pause_samples))

                final_audio = np.concatenate(conversation_audio)

                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_stem = f"conversation_qwen_base_{timestamp}"
                speakers_used = list(set(k for k, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Qwen3-TTS Conversation (Base Model + Custom Voices)
                    Model: Base {model_size}
                    Language: {language}
                    Seed: {seed}
                    Pause Settings:
                      - Linebreak: {pause_linebreak}s
                      - Period: {pause_period}s
                      - Comma: {pause_comma}s
                      - Question: {pause_question}s
                      - Hyphen: {pause_hyphen}s
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {len(all_segments)}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                duration = len(final_audio) / sr
                status_msg = f"Conversation generated.\n{len(lines)} lines | {duration:.1f}s | Seed: {seed} | Base {model_size}"
                return finalize_conversation_output(final_audio, sr, filename_stem, metadata_out, status_msg, progress)

            except Exception as e:
                import traceback
                print(f"Error in generate_conversation_base_handler:\n{traceback.format_exc()}")
                return None, f"❌ Error generating conversation: {str(e)}", "", gr.update()

        def generate_vibevoice_longform_handler(script_text, voice_samples_dict, model_size, cfg_scale, seed,
                                                num_steps, do_sample, temperature, top_k, top_p, repetition_penalty,
                                                sentences_per_chunk=0, progress=gr.Progress()):
            """Generate long-form multi-speaker audio with VibeVoice (up to 90 min)."""
            if not script_text or not script_text.strip():
                return None, "❌ Please enter a script.", "", gr.update()

            script_text = preprocess_conversation_script(script_text)

            try:
                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)

                progress(0.1, desc=f"Loading VibeVoice TTS ({model_size})...")
                model = tts_manager.get_vibevoice_tts(model_size)

                # Import processor
                from vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
                import warnings
                import logging

                # Map model size
                if model_size == "Large (4-bit)":
                    model_path = "FranckyB/VibeVoice-Large-4bit"
                else:
                    model_path = f"FranckyB/VibeVoice-{model_size}"

                # Suppress tokenizer warning
                prev_level = logging.getLogger("transformers.tokenization_utils_base").level
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    processor = VibeVoiceProcessor.from_pretrained(model_path, local_files_only=False)

                logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

                # Parse script
                progress(0.3, desc="Processing script...")
                lines = []
                for line in script_text.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if text:
                                wrapped_num = ((speaker_num - 1) % 4) + 1
                                lines.append((f"Speaker{wrapped_num}", text, speaker_num))

                # Build voice samples list
                available_samples = []
                for i in range(1, 5):
                    speaker_key = f"Speaker{i}"
                    if speaker_key in voice_samples_dict and voice_samples_dict[speaker_key]:
                        sample_data = voice_samples_dict[speaker_key]
                        wav_path = sample_data["wav_path"] if isinstance(sample_data, dict) else sample_data
                        available_samples.append((speaker_key, wav_path))

                if not available_samples:
                    return None, "❌ Please provide at least one voice sample (Speaker1).", "", gr.update()

                voice_samples = [sample for _, sample in available_samples]
                speaker_to_sample = {speaker: idx for idx, (speaker, _) in enumerate(available_samples)}

                # Format script for VibeVoice (0-based)
                formatted_lines = []
                for speaker, text, original_num in lines:
                    if speaker in speaker_to_sample:
                        vv_speaker_num = speaker_to_sample[speaker]
                        clean_text, _ = extract_style_instructions(text)
                        formatted_lines.append(f"Speaker {vv_speaker_num}: {clean_text}")

                formatted_script = '\n'.join(formatted_lines)

                # Process inputs
                inputs = processor(
                    text=[formatted_script],
                    voice_samples=[voice_samples],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move to device
                device = get_device()
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(device)

                progress(0.5, desc="Generating audio...")

                # Build generation config
                gen_config = {'do_sample': do_sample}
                if do_sample:
                    gen_config['temperature'] = temperature
                    if top_k > 0:
                        gen_config['top_k'] = int(top_k)
                    if top_p < 1.0:
                        gen_config['top_p'] = top_p
                    if repetition_penalty != 1.0:
                        gen_config['repetition_penalty'] = repetition_penalty

                model.set_ddpm_inference_steps(num_steps=num_steps)
                sr = 24000

                sentences_per_chunk = int(sentences_per_chunk) if sentences_per_chunk else 0

                # Chunked generation: process N lines at a time to prevent
                # quality degradation (screaming/rushing) on long conversations.
                if sentences_per_chunk > 0 and len(formatted_lines) > sentences_per_chunk:
                    import numpy as np
                    chunks = []
                    for i in range(0, len(formatted_lines), sentences_per_chunk):
                        chunks.append(formatted_lines[i:i + sentences_per_chunk])

                    print(f"VibeVoice conversation chunking: {len(chunks)} chunks of ~{sentences_per_chunk} line(s)")
                    audio_segments = []

                    for idx, chunk_lines in enumerate(chunks):
                        chunk_script = '\n'.join(chunk_lines)
                        progress_val = 0.5 + (0.4 * idx / len(chunks))
                        progress(progress_val, desc=f"Generating chunk {idx + 1}/{len(chunks)}...")

                        chunk_inputs = processor(
                            text=[chunk_script],
                            voice_samples=[voice_samples],
                            padding=True,
                            return_tensors="pt",
                            return_attention_mask=True,
                        )

                        for k, v in chunk_inputs.items():
                            if torch.is_tensor(v):
                                chunk_inputs[k] = v.to(device)

                        outputs = model.generate(
                            **chunk_inputs,
                            max_new_tokens=None,
                            cfg_scale=cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config=gen_config,
                            verbose=False,
                        )

                        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                            audio_segments.append(audio_tensor.squeeze().numpy())
                            print(f"  Chunk {idx + 1}/{len(chunks)} done ({len(chunk_lines)} lines)")
                        else:
                            print(f"  Chunk {idx + 1}/{len(chunks)} produced no audio, skipping")

                    if not audio_segments:
                        return None, "❌ VibeVoice failed to generate audio for any chunk", "", gr.update()

                    generated_audio = np.concatenate(audio_segments)

                else:
                    # Standard single-pass generation
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config=gen_config,
                        verbose=False,
                    )

                    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                        audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                        generated_audio = audio_tensor.squeeze().numpy()
                    else:
                        return None, "❌ No audio generated", "", gr.update()

                progress(0.9, desc="Saving audio...")

                if generated_audio is not None:

                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_stem = f"Conversation_vibevoice_{timestamp}"
                    duration = len(generated_audio) / sr
                    chunk_info = f"Lines per Chunk: {sentences_per_chunk}" if sentences_per_chunk > 0 else "Chunking: Off"
                    metadata = dedent(f"""\
                        Generated: {timestamp}
                        Type: VibeVoice Conversation
                        Model: VibeVoice-{model_size}
                        Seed: {seed}
                        CFG Scale: {cfg_scale}
                        {chunk_info}
                        Lines: {len(lines)}
                        Speakers: {len(available_samples)}

                        --- Script ---
                        {script_text.strip()}
                        """)
                    metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                    status_msg = f"Conversation generated.\n{len(lines)} lines | {duration:.1f}s | Seed: {seed}"
                    return finalize_conversation_output(generated_audio, sr, filename_stem, metadata_out, status_msg, progress)

            except Exception as e:
                import traceback
                print(f"Error in generate_vibevoice_longform_handler:\n{traceback.format_exc()}")
                return None, f"❌ Error generating conversation: {str(e)}", "", gr.update()

        def generate_luxtts_conversation_handler(
            conversation_data, voice_samples_dict,
            pause_linebreak, seed,
            num_steps, t_shift, speed, guidance_scale,
            rms, ref_duration, return_smooth,
            progress=gr.Progress()
        ):
            """Generate multi-speaker conversation with LuxTTS voice cloning.

            Each line is generated sequentially with per-speaker voice prompts (cached),
            then stitched together with configurable pauses between speaker turns.
            """
            if not conversation_data or not conversation_data.strip():
                return None, "❌ Please enter conversation lines.", "", gr.update()

            if not voice_samples_dict:
                return None, "❌ Please select at least one voice sample.", "", gr.update()

            # Check all samples have transcripts
            transcript_error = validate_samples_have_transcripts(voice_samples_dict)
            if transcript_error:
                return None, f"❌ {transcript_error}", "", gr.update()

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data.get("name", speaker_key), text, sample_data.get("ref_text", "")))

                if not lines:
                    return None, "❌ No valid conversation lines found. Use format: [N]: Text (N=1-8)", "", gr.update()

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)

                progress(0.05, desc="Loading LuxTTS model...")
                tts_manager.get_luxtts()

                # Generate all segments
                all_segments = []
                sr = 48000  # LuxTTS outputs at 48kHz

                for i, (speaker_key, wav_path, sample_name, text, ref_text) in enumerate(lines):
                    progress_val = 0.1 + (0.8 * i / len(lines))

                    # Strip any (style) markers — LuxTTS doesn't support them
                    clean_text, _ = extract_style_instructions(text)
                    if not clean_text.strip():
                        continue

                    progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                    audio_data, audio_sr, was_cached = tts_manager.generate_voice_clone_luxtts(
                        text=clean_text,
                        voice_sample_path=wav_path,
                        sample_name=sample_name,
                        num_steps=num_steps,
                        t_shift=t_shift,
                        speed=speed,
                        return_smooth=return_smooth,
                        rms=rms,
                        ref_duration=ref_duration,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        ref_text=ref_text or None,
                    )
                    sr = audio_sr  # Should be 48000

                    # Add segment with linebreak pause after (except last line)
                    line_pause = pause_linebreak if i < len(lines) - 1 else 0.0
                    all_segments.append((audio_data, line_pause))

                if not all_segments:
                    return None, "❌ No audio segments generated.", "", gr.update()

                # Concatenate
                progress(0.9, desc="Stitching conversation...")
                conversation_audio = []
                for wav, pause_duration in all_segments:
                    conversation_audio.append(wav)
                    if pause_duration > 0:
                        pause_samples = int(sr * pause_duration)
                        conversation_audio.append(np.zeros(pause_samples))

                # Add short tail silence so the last utterance doesn't clip
                conversation_audio.append(np.zeros(int(sr * 0.15)))

                final_audio = np.concatenate(conversation_audio)

                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_stem = f"conversation_luxtts_{timestamp}"
                speakers_used = list(set(k for k, _, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: LuxTTS Conversation
                    Seed: {seed}
                    Sample Rate: {sr}Hz
                    Pause Between Lines: {pause_linebreak}s
                    Steps: {num_steps} | t_shift: {t_shift} | Speed: {speed}
                    Guidance Scale: {guidance_scale} | RMS: {rms} | Ref Duration: {ref_duration}s
                    Return Smooth: {return_smooth}
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {len(all_segments)}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                duration = len(final_audio) / sr
                status_msg = f"Conversation generated.\n{len(lines)} lines | {duration:.1f}s | Seed: {seed} | LuxTTS"
                return finalize_conversation_output(final_audio, sr, filename_stem, metadata_out, status_msg, progress)

            except Exception as e:
                import traceback
                print(f"Error in generate_luxtts_conversation_handler:\n{traceback.format_exc()}")
                return None, f"Error generating LuxTTS conversation: {str(e)}", "", gr.update()

        def generate_chatterbox_conversation_handler(
            conversation_data, voice_samples_dict,
            pause_linebreak, language, seed,
            exaggeration, cfg_weight, temperature, repetition_penalty, top_p,
            progress=gr.Progress()
        ):
            """Generate multi-speaker conversation with Chatterbox voice cloning.

            Each line is generated sequentially with per-speaker voice samples,
            then stitched together with configurable pauses between speaker turns.
            English uses the fast default model; other languages use Multilingual.
            """
            if not conversation_data or not conversation_data.strip():
                return None, "❌ Please enter conversation lines.", "", gr.update()

            if not voice_samples_dict:
                return None, "❌ Please select at least one voice sample.", "", gr.update()

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                from modules.core_components.constants import CHATTERBOX_LANG_TO_CODE

                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data.get("name", speaker_key), text))

                if not lines:
                    return None, "❌ No valid conversation lines found. Use format: [N]: Text (N=1-8)", "", gr.update()

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)

                # Determine if multilingual
                lang_code = CHATTERBOX_LANG_TO_CODE.get(language, "en")
                use_multilingual = lang_code != "en"
                model_label = "Multilingual" if use_multilingual else "Default"

                if use_multilingual:
                    progress(0.05, desc="Loading Chatterbox Multilingual...")
                    tts_manager.get_chatterbox_multilingual()
                else:
                    progress(0.05, desc="Loading Chatterbox TTS...")
                    tts_manager.get_chatterbox_tts()

                # Generate all segments
                all_segments = []
                sr = 24000

                for i, (speaker_key, wav_path, sample_name, text) in enumerate(lines):
                    progress_val = 0.1 + (0.8 * i / len(lines))

                    # Strip any (style) markers
                    clean_text, _ = extract_style_instructions(text)
                    if not clean_text.strip():
                        continue

                    progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                    if use_multilingual:
                        audio_data, audio_sr = tts_manager.generate_voice_clone_chatterbox_multilingual(
                            text=clean_text,
                            language_code=lang_code,
                            voice_sample_path=wav_path,
                            seed=seed,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            top_p=top_p,
                        )
                    else:
                        audio_data, audio_sr = tts_manager.generate_voice_clone_chatterbox(
                            text=clean_text,
                            voice_sample_path=wav_path,
                            seed=seed,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            top_p=top_p,
                        )

                    sr = audio_sr  # Should be 24000

                    # Add segment with linebreak pause after (except last line)
                    line_pause = pause_linebreak if i < len(lines) - 1 else 0.0
                    all_segments.append((audio_data, line_pause))

                if not all_segments:
                    return None, "❌ No audio segments generated.", "", gr.update()

                # Concatenate
                progress(0.9, desc="Stitching conversation...")
                conversation_audio = []
                for wav, pause_duration in all_segments:
                    conversation_audio.append(wav)
                    if pause_duration > 0:
                        pause_samples = int(sr * pause_duration)
                        conversation_audio.append(np.zeros(pause_samples))

                # Add short tail silence
                conversation_audio.append(np.zeros(int(sr * 0.15)))

                final_audio = np.concatenate(conversation_audio)

                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_stem = f"conversation_chatterbox_{timestamp}"
                speakers_used = list(set(k for k, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Chatterbox Conversation
                    Model: {model_label}
                    Language: {language}
                    Seed: {seed}
                    Sample Rate: {sr}Hz
                    Pause Between Lines: {pause_linebreak}s
                    Exaggeration: {exaggeration} | CFG Weight: {cfg_weight}
                    Temperature: {temperature} | Repetition Penalty: {repetition_penalty} | Top-p: {top_p}
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {len(all_segments)}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                duration = len(final_audio) / sr
                status_msg = f"Conversation generated.\n{len(lines)} lines | {duration:.1f}s | Seed: {seed} | Chatterbox {model_label}"
                return finalize_conversation_output(final_audio, sr, filename_stem, metadata_out, status_msg, progress)

            except Exception as e:
                import traceback
                print(f"Error in generate_chatterbox_conversation_handler:\n{traceback.format_exc()}")
                return None, f"Error generating Chatterbox conversation: {str(e)}", "", gr.update()

        def unified_conversation_generate(
            model_type, script,
            # Shared voice samples (used by all sample-based engines)
            sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8,
            # Qwen Speakers params
            qwen_custom_pause_linebreak, qwen_custom_pause_period, qwen_custom_pause_comma,
            qwen_custom_pause_question, qwen_custom_pause_hyphen, qwen_custom_model_size,
            # Qwen Base params
            qwen_base_pause_linebreak, qwen_base_pause_period, qwen_base_pause_comma, qwen_base_pause_question,
            qwen_base_pause_hyphen, qwen_base_model_size,
            # Shared Qwen
            qwen_lang, qwen_seed, emotion_intensity,
            # Qwen Custom Voice advanced params
            qwen_custom_do_sample, qwen_custom_temperature, qwen_custom_top_k, qwen_custom_top_p,
            qwen_custom_repetition_penalty, qwen_custom_max_new_tokens,
            # Qwen Base advanced params
            qwen_base_do_sample, qwen_base_temperature, qwen_base_top_k, qwen_base_top_p,
            qwen_base_repetition_penalty, qwen_base_max_new_tokens,
            # VibeVoice params
            vv_model_size, vv_cfg,
            # VibeVoice advanced params
            vv_num_steps, vv_do_sample, vv_temperature, vv_top_k, vv_top_p, vv_repetition_penalty,
            vv_sentences_per_chunk,
            # LuxTTS params
            lux_pause_linebreak,
            lux_num_steps, lux_t_shift, lux_speed, lux_guidance_scale,
            lux_rms, lux_ref_duration, lux_return_smooth,
            # Chatterbox params
            cb_pause_linebreak,
            cb_exaggeration, cb_cfg_weight, cb_temperature, cb_repetition_penalty, cb_top_p,
            # Shared
            seed, progress=gr.Progress()
        ):
            """Route to appropriate generation function based on model type."""
            voice_samples = prepare_voice_samples_dict(sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8)

            if model_type == "Qwen Speakers":
                qwen_size = "1.7B" if qwen_custom_model_size == "Large" else "0.6B"
                return generate_conversation_handler(script, qwen_custom_pause_linebreak, qwen_custom_pause_period,
                                                     qwen_custom_pause_comma, qwen_custom_pause_question,
                                                     qwen_custom_pause_hyphen, qwen_lang, qwen_seed, qwen_size,
                                                     qwen_custom_do_sample, qwen_custom_temperature, qwen_custom_top_k, qwen_custom_top_p,
                                                     qwen_custom_repetition_penalty, qwen_custom_max_new_tokens, progress)
            elif model_type == "Qwen Base":
                qwen_size = "1.7B" if qwen_base_model_size == "Large" else "0.6B"
                return generate_conversation_base_handler(script, voice_samples, qwen_base_pause_linebreak,
                                                          qwen_base_pause_period, qwen_base_pause_comma,
                                                          qwen_base_pause_question, qwen_base_pause_hyphen,
                                                          qwen_lang, qwen_seed, qwen_size,
                                                          qwen_base_do_sample, qwen_base_temperature, qwen_base_top_k, qwen_base_top_p,
                                                          qwen_base_repetition_penalty, qwen_base_max_new_tokens,
                                                          emotion_intensity, progress)
            elif model_type == "LuxTTS":
                return generate_luxtts_conversation_handler(script, voice_samples, lux_pause_linebreak,
                                                            seed, lux_num_steps, lux_t_shift, lux_speed,
                                                            lux_guidance_scale, lux_rms, lux_ref_duration,
                                                            lux_return_smooth, progress)
            elif model_type == "Chatterbox":
                return generate_chatterbox_conversation_handler(script, voice_samples, cb_pause_linebreak,
                                                                qwen_lang, seed,
                                                                cb_exaggeration, cb_cfg_weight, cb_temperature,
                                                                cb_repetition_penalty, cb_top_p, progress)
            else:  # VibeVoice
                if vv_model_size == "Small":
                    vv_size = "1.5B"
                elif vv_model_size == "Large (4-bit)":
                    vv_size = "Large (4-bit)"
                else:
                    vv_size = "Large"
                return generate_vibevoice_longform_handler(script, voice_samples, vv_size, vv_cfg, seed,
                                                           vv_num_steps, vv_do_sample, vv_temperature, vv_top_k,
                                                           vv_top_p, vv_repetition_penalty,
                                                           vv_sentences_per_chunk, progress)

        # Event handlers
        components['conv_generate_btn'].click(
            restore_fn, outputs=restore_outputs
        ).then(
            unified_conversation_generate,
            inputs=[
                components['conv_model_type'], components['conversation_script'],
                # Shared voice samples
                components['conv_voice_1'], components['conv_voice_2'], components['conv_voice_3'], components['conv_voice_4'],
                components['conv_voice_5'], components['conv_voice_6'], components['conv_voice_7'], components['conv_voice_8'],
                # Qwen Speakers
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_model_size'],
                # Qwen Base
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_base_model_size'],
                # Shared Qwen
                components['conv_language'], components['conv_seed'], components['conv_emotion_intensity'],
                # Qwen Custom Voice advanced params
                components['qwen_custom_conv_do_sample'], components['qwen_custom_conv_temperature'],
                components['qwen_custom_conv_top_k'], components['qwen_custom_conv_top_p'],
                components['qwen_custom_conv_repetition_penalty'], components['qwen_custom_conv_max_new_tokens'],
                # Qwen Base advanced params
                components['qwen_base_conv_do_sample'], components['qwen_base_conv_temperature'],
                components['qwen_base_conv_top_k'], components['qwen_base_conv_top_p'],
                components['qwen_base_conv_repetition_penalty'], components['qwen_base_conv_max_new_tokens'],
                # VibeVoice
                components['longform_model_size'], components['longform_cfg_scale'],
                # VibeVoice advanced params
                components['vv_conv_num_steps'], components['vv_conv_do_sample'], components['vv_conv_temperature'], components['vv_conv_top_k'],
                components['vv_conv_top_p'], components['vv_conv_repetition_penalty'],
                components['vv_conv_sentences_per_chunk'],
                # LuxTTS
                components['luxtts_pause_linebreak'],
                components['lux_conv_num_steps'], components['lux_conv_t_shift'], components['lux_conv_speed'], components['lux_conv_guidance_scale'],
                components['lux_conv_rms'], components['lux_conv_ref_duration'], components['lux_conv_return_smooth'],
                # Chatterbox
                components['cb_pause_linebreak'],
                components['cb_conv_exaggeration'], components['cb_conv_cfg_weight'], components['cb_conv_temperature'],
                components['cb_conv_repetition_penalty'], components['cb_conv_top_p'],
                # Shared
                components['conv_seed']
            ],
            outputs=[components['conv_output_audio'], components['conv_status'], components['_conv_result_metadata'], components['conv_save_result_btn']]
        )

        # Save result button handler
        def conv_save_result_handler(audio_path, metadata_text):
            """Save the temp result to output folder in chosen format."""
            if not audio_path:
                return "❌ No audio to save.", gr.update()
            try:
                output_format = _user_config.get("output_format", "wav")
                output_path = save_result_to_output(audio_path, OUTPUT_DIR, output_format, metadata_text or None)
                return f"Saved to: {output_path.name}", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Error saving: {str(e)}", gr.update()

        components['conv_save_result_btn'].click(
            conv_save_result_handler,
            inputs=[components['conv_output_audio'], components['_conv_result_metadata']],
            outputs=[components['conv_status'], components['conv_save_result_btn']]
        )

        # Toggle UI based on model selection
        def toggle_conv_ui(model_type):
            from modules.core_components.constants import CHATTERBOX_LANGUAGES
            is_qwen_custom = model_type == "Qwen Speakers"
            is_qwen_base = model_type == "Qwen Base"
            is_vibevoice = model_type == "VibeVoice"
            is_luxtts = model_type == "LuxTTS"
            is_chatterbox = model_type == "Chatterbox"
            is_qwen = is_qwen_custom or is_qwen_base
            uses_samples = is_qwen_base or is_luxtts or is_chatterbox or is_vibevoice

            # Language dropdown: full list for Qwen, Chatterbox languages for Chatterbox, Auto-only for VV/LuxTTS
            if is_qwen:
                lang_update = gr.update(choices=LANGUAGES, value=LANGUAGES[0])
            elif is_chatterbox:
                lang_update = gr.update(choices=CHATTERBOX_LANGUAGES, value="English")
            else:
                lang_update = gr.update(choices=["Auto"], value="Auto")

            return {
                components['qwen_speaker_table']: gr.update(visible=is_qwen_custom),
                components['shared_voices_section']: gr.update(visible=uses_samples),
                components['conv_voices_row_5_6']: gr.update(visible=uses_samples and not is_vibevoice),
                components['conv_voices_row_7_8']: gr.update(visible=uses_samples and not is_vibevoice),
                components['qwen_custom_settings']: gr.update(visible=is_qwen_custom),
                components['qwen_base_settings']: gr.update(visible=is_qwen_base),
                components['qwen_pause_controls']: gr.update(visible=is_qwen),
                components['qwen_custom_conv_advanced']: gr.update(visible=is_qwen_custom),
                components['qwen_base_conv_advanced']: gr.update(visible=is_qwen_base),
                components['luxtts_settings']: gr.update(visible=is_luxtts),
                components['vibevoice_settings']: gr.update(visible=is_vibevoice),
                components['cb_settings']: gr.update(visible=is_chatterbox),
                components['qwen_custom_tips']: gr.update(visible=is_qwen_custom),
                components['qwen_base_tips']: gr.update(visible=is_qwen_base),
                components['vibevoice_tips']: gr.update(visible=is_vibevoice),
                components['luxtts_tips']: gr.update(visible=is_luxtts),
                components['cb_tips']: gr.update(visible=is_chatterbox),
                components['conv_language']: lang_update,
            }

        components['conv_model_type'].change(
            toggle_conv_ui,
            inputs=[components['conv_model_type']],
            outputs=[components['qwen_speaker_table'],
                     components['shared_voices_section'], components['conv_voices_row_5_6'], components['conv_voices_row_7_8'],
                     components['qwen_custom_settings'], components['qwen_base_settings'], components['qwen_pause_controls'],
                     components['qwen_custom_conv_advanced'], components['qwen_base_conv_advanced'],
                     components['luxtts_settings'],
                     components['vibevoice_settings'],
                     components['cb_settings'],
                     components['qwen_custom_tips'], components['qwen_base_tips'], components['vibevoice_tips'],
                     components['luxtts_tips'], components['cb_tips'],
                     components['conv_language']]
        )

        # Refresh voice samples handler
        def refresh_all_voice_samples():
            """Refresh all shared voice sample dropdowns."""
            updated_samples = get_sample_choices()
            update = gr.update(choices=updated_samples)
            return [update] * 8

        # Auto-refresh all voice sample dropdowns when tab is selected
        components['conv_tab'].select(
            refresh_all_voice_samples,
            outputs=[
                components['conv_voice_1'], components['conv_voice_2'],
                components['conv_voice_3'], components['conv_voice_4'],
                components['conv_voice_5'], components['conv_voice_6'],
                components['conv_voice_7'], components['conv_voice_8'],
            ]
        )

        # Restore saved params when accordion is opened
        for acc_key in ['qwen_custom_conv_accordion', 'qwen_base_conv_accordion', 'vv_conv_accordion', 'lux_conv_accordion', 'cb_conv_accordion']:
            acc = components.get(acc_key)
            if acc is not None:
                acc.expand(restore_fn, outputs=restore_outputs)

        # Save preferences
        components['conv_model_type'].change(
            lambda x: save_preference("conv_model_type", x),
            inputs=[components['conv_model_type']],
            outputs=[]
        )

        components['conv_model_size'].change(
            lambda x: save_preference("conv_model_size", x),
            inputs=[components['conv_model_size']],
            outputs=[]
        )

        components['conv_base_model_size'].change(
            lambda x: save_preference("conv_base_model_size", x),
            inputs=[components['conv_base_model_size']],
            outputs=[]
        )

        components['longform_model_size'].change(
            lambda x: save_preference("vibevoice_model_size", x),
            inputs=[components['longform_model_size']],
            outputs=[]
        )

        components['conv_language'].change(
            lambda x: save_preference("language", x),
            inputs=[components['conv_language']],
            outputs=[]
        )

        # Save conversation pause preferences
        components['conv_pause_linebreak'].change(
            lambda x: save_preference("conv_pause_linebreak", x),
            inputs=[components['conv_pause_linebreak']],
            outputs=[]
        )

        components['conv_pause_period'].change(
            lambda x: save_preference("conv_pause_period", x),
            inputs=[components['conv_pause_period']],
            outputs=[]
        )

        components['conv_pause_comma'].change(
            lambda x: save_preference("conv_pause_comma", x),
            inputs=[components['conv_pause_comma']],
            outputs=[]
        )

        components['conv_pause_question'].change(
            lambda x: save_preference("conv_pause_question", x),
            inputs=[components['conv_pause_question']],
            outputs=[]
        )

        components['conv_pause_hyphen'].change(
            lambda x: save_preference("conv_pause_hyphen", x),
            inputs=[components['conv_pause_hyphen']],
            outputs=[]
        )


# Export for tab registry
get_tool_class = lambda: ConversationTool


if __name__ == "__main__":
    """Standalone testing of Conversation tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(ConversationTool, port=7864, title="Conversation - Standalone")
