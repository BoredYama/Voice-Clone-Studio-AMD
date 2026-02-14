"""
Voice Clone Tab

Clone voices from samples using Qwen3-TTS, VibeVoice, LuxTTS, or Chatterbox.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    # Also add modules directory for vibevoice_tts imports
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import soundfile as sf
import torch
import random
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from modules.core_components.ai_models.model_utils import set_seed

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from gradio_filelister import FileLister

class VoiceCloneTool(Tool):
    """Voice Clone tool implementation."""

    config = ToolConfig(
        name="Voice Clone",
        module_name="tool_voice_clone",
        description="Clone voices from voice samples",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Clone tool UI."""
        components = {}

        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_emotion_choices = shared_state['get_emotion_choices']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        LANGUAGES = shared_state['LANGUAGES']
        VOICE_CLONE_OPTIONS = shared_state['VOICE_CLONE_OPTIONS']
        TTS_ENGINES = shared_state.get('TTS_ENGINES', {})
        _user_config = shared_state['_user_config']
        _active_emotions = shared_state['_active_emotions']

        # Filter voice clone options based on enabled engines
        engine_settings = _user_config.get("enabled_engines", {})
        visible_options = []
        for engine_key, engine_info in TTS_ENGINES.items():
            if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                visible_options.extend(engine_info["choices"])
        # Fall back to all options if nothing is enabled (safety)
        if not visible_options:
            visible_options = VOICE_CLONE_OPTIONS

        # Resolve default model based on which engines are enabled
        from modules.core_components.constants import get_default_voice_clone_model
        default_model = get_default_voice_clone_model(_user_config)
        saved_model = _user_config.get("voice_clone_model", default_model)
        if saved_model not in visible_options:
            saved_model = visible_options[0]
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        refresh_samples = shared_state['refresh_samples']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']

        with gr.TabItem("Voice Clone") as voice_clone_tab:
            components['voice_clone_tab'] = voice_clone_tab
            gr.Markdown("Clone Voices from Samples. <small>(Use Prep Audio Samples to add samples)</small>")
            with gr.Row():
                # Left column - Sample selection (1/3 width)
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")

                    components['sample_lister'] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    with gr.Row():
                        pass

                    components['sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False,
                        visible=True,
                        value=None,
                        elem_id="voice-clone-sample-audio"
                    )

                    components['sample_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                # Right column - Generation (2/3 width)
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6
                    )

                    with gr.Row():
                        components['clone_model_dropdown'] = gr.Dropdown(
                            choices=visible_options,
                            value=saved_model,
                            label="Engine & Model",
                            scale=4
                        )
                        components['seed_input'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Determine which model is initially selected to conditionally show/hide parameters
                    is_qwen_initial = "Qwen" in saved_model

                    # Language dropdown (hidden for VibeVoice models)
                    components['language_row'] = gr.Row(visible=is_qwen_initial)
                    with components['language_row']:
                        components['language_dropdown'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                        )

                    # Qwen3 Advanced Parameters (create_qwen_advanced_params includes its own accordion)
                    create_qwen_advanced_params = shared_state['create_qwen_advanced_params']

                    qwen_params = create_qwen_advanced_params(
                        emotions_dict=_active_emotions,
                        include_emotion=True,
                        initial_emotion="(None)",
                        initial_intensity=1.0,
                        visible=is_qwen_initial,
                        emotion_visible=True,
                        shared_state=shared_state
                    )

                    # Store the accordion reference for toggling
                    components['qwen_params_accordion'] = qwen_params.get('accordion')

                    # Store references
                    components['qwen_params'] = qwen_params
                    components['qwen_emotion_preset'] = qwen_params['emotion_preset']
                    components['qwen_emotion_intensity'] = qwen_params['emotion_intensity']
                    components['qwen_save_emotion_btn'] = qwen_params.get('save_emotion_btn')
                    components['qwen_delete_emotion_btn'] = qwen_params.get('delete_emotion_btn')
                    components['qwen_emotion_save_name'] = qwen_params.get('emotion_save_name')
                    components['qwen_do_sample'] = qwen_params['do_sample']
                    components['qwen_temperature'] = qwen_params['temperature']
                    components['qwen_top_k'] = qwen_params['top_k']
                    components['qwen_top_p'] = qwen_params['top_p']
                    components['qwen_repetition_penalty'] = qwen_params['repetition_penalty']
                    components['qwen_max_new_tokens'] = qwen_params['max_new_tokens']

                    # VibeVoice Advanced Parameters
                    is_vv_initial = "VibeVoice" in saved_model
                    create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
                    vv_params = create_vibevoice_advanced_params(
                        include_sentences_per_chunk=True,
                        visible=is_vv_initial
                    )
                    components['vv_params_accordion'] = vv_params['accordion']
                    components['vv_cfg_scale'] = vv_params['cfg_scale']
                    components['vv_num_steps'] = vv_params['num_steps']
                    components['vv_do_sample'] = vv_params['do_sample']
                    components['vv_sentences_per_chunk'] = vv_params['sentences_per_chunk']
                    components['vv_repetition_penalty'] = vv_params['repetition_penalty']
                    components['vv_temperature'] = vv_params['temperature']
                    components['vv_top_k'] = vv_params['top_k']
                    components['vv_top_p'] = vv_params['top_p']

                    # LuxTTS Advanced Parameters
                    is_lux_initial = "LuxTTS" in saved_model
                    create_luxtts_advanced_params = shared_state['create_luxtts_advanced_params']
                    luxtts_params = create_luxtts_advanced_params(
                        visible=is_lux_initial
                    )
                    components['luxtts_params_accordion'] = luxtts_params.get('accordion')
                    components['luxtts_num_steps'] = luxtts_params['num_steps']
                    components['luxtts_t_shift'] = luxtts_params['t_shift']
                    components['luxtts_speed'] = luxtts_params['speed']
                    components['luxtts_return_smooth'] = luxtts_params['return_smooth']
                    components['luxtts_rms'] = luxtts_params['rms']
                    components['luxtts_ref_duration'] = luxtts_params['ref_duration']
                    components['luxtts_guidance_scale'] = luxtts_params['guidance_scale']

                    # Chatterbox Advanced Parameters
                    is_cb_initial = "Chatterbox" in saved_model
                    create_chatterbox_advanced_params = shared_state['create_chatterbox_advanced_params']
                    cb_params = create_chatterbox_advanced_params(
                        visible=is_cb_initial
                    )
                    components['cb_params_accordion'] = cb_params['accordion']
                    components['cb_exaggeration'] = cb_params['exaggeration']
                    components['cb_cfg_weight'] = cb_params['cfg_weight']
                    components['cb_temperature'] = cb_params['temperature']
                    components['cb_repetition_penalty'] = cb_params['repetition_penalty']
                    components['cb_top_p'] = cb_params['top_p']

                    # Chatterbox Multilingual language dropdown (only shown for Chatterbox - Multilingual)
                    is_cb_mtl_initial = "Multilingual" in saved_model
                    components['cb_language_row'] = gr.Row(visible=is_cb_mtl_initial)
                    with components['cb_language_row']:
                        from modules.core_components.constants import CHATTERBOX_LANGUAGES
                        components['cb_language_dropdown'] = gr.Dropdown(
                            choices=CHATTERBOX_LANGUAGES,
                            value="English",
                            label="Language (Chatterbox Multilingual)",
                        )

                    components['generate_btn'] = gr.Button("Generate Audio", variant="primary", size="lg")

                    components['output_audio'] = gr.Audio(
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

                    components['clone_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Clone tab events."""

        # Get helper functions and directories
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        get_or_create_voice_prompt = shared_state['get_or_create_voice_prompt']
        refresh_samples = shared_state['refresh_samples']
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
        save_audio_to_temp = shared_state['save_audio_to_temp']
        save_result_to_output = shared_state['save_result_to_output']

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        # Wire param persistence (auto-save on change)
        wire_param_persistence = shared_state['wire_param_persistence']
        _user_config = shared_state['_user_config']
        param_map = {
            'qwen': [
                ('qwen_emotion_preset', 'emotion_preset'),
                ('qwen_emotion_intensity', 'emotion_intensity'),
                ('qwen_do_sample', 'do_sample'),
                ('qwen_temperature', 'temperature'),
                ('qwen_top_k', 'top_k'),
                ('qwen_top_p', 'top_p'),
                ('qwen_repetition_penalty', 'repetition_penalty'),
                ('qwen_max_new_tokens', 'max_new_tokens'),
            ],
            'vibevoice': [
                ('vv_cfg_scale', 'cfg_scale'),
                ('vv_num_steps', 'num_steps'),
                ('vv_do_sample', 'do_sample'),
                ('vv_sentences_per_chunk', 'sentences_per_chunk'),
                ('vv_repetition_penalty', 'repetition_penalty'),
                ('vv_temperature', 'temperature'),
                ('vv_top_k', 'top_k'),
                ('vv_top_p', 'top_p'),
            ],
            'luxtts': [
                ('luxtts_num_steps', 'num_steps'),
                ('luxtts_t_shift', 't_shift'),
                ('luxtts_speed', 'speed'),
                ('luxtts_return_smooth', 'return_smooth'),
                ('luxtts_rms', 'rms'),
                ('luxtts_ref_duration', 'ref_duration'),
                ('luxtts_guidance_scale', 'guidance_scale'),
            ],
            'chatterbox': [
                ('cb_exaggeration', 'exaggeration'),
                ('cb_cfg_weight', 'cfg_weight'),
                ('cb_temperature', 'temperature'),
                ('cb_repetition_penalty', 'repetition_penalty'),
                ('cb_top_p', 'top_p'),
            ],
        }
        wire_param_persistence(components, _user_config, 'voice_clone', param_map)

        # Create restore handler for applying saved params on tab select / model switch
        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, _user_config, 'voice_clone', param_map)

        def get_selected_sample_name(lister_value):
            """Extract selected sample name from FileLister value (strips .wav extension)."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def generate_audio_handler(sample_name, text_to_generate, language, seed, model_selection="Qwen3 - Small",
                                   qwen_do_sample=True, qwen_temperature=0.9, qwen_top_k=50, qwen_top_p=1.0, qwen_repetition_penalty=1.05,
                                   qwen_max_new_tokens=2048,
                                   vv_do_sample=False, vv_temperature=1.0, vv_top_k=50, vv_top_p=1.0, vv_repetition_penalty=1.1,
                                   vv_cfg_scale=3.0, vv_num_steps=20, vv_sentences_per_chunk=0,
                                   lux_num_steps=4, lux_t_shift=0.5, lux_speed=1.0, lux_return_smooth=False,
                                   lux_rms=0.01, lux_ref_duration=30, lux_guidance_scale=3.0,
                                   cb_exaggeration=0.5, cb_cfg_weight=0.5, cb_temperature=0.8,
                                   cb_repetition_penalty=1.2, cb_top_p=1.0, cb_language="English",
                                   progress=gr.Progress()):
            """Generate audio using voice cloning - supports Qwen, VibeVoice, and LuxTTS engines."""
            if not sample_name:
                return None, "Please select a voice sample first."

            if not text_to_generate or not text_to_generate.strip():
                return None, "Please enter text to generate."

            # Parse model selection to determine engine and size
            if "LuxTTS" in model_selection:
                engine = "luxtts"
                model_size = "Default"
            elif "VibeVoice" in model_selection:
                engine = "vibevoice"
                if "Small" in model_selection:
                    model_size = "1.5B"
                elif "4-bit" in model_selection:
                    model_size = "Large (4-bit)"
                else:  # Large
                    model_size = "Large"
            elif "Chatterbox" in model_selection:
                engine = "chatterbox"
                if "Multilingual" in model_selection:
                    model_size = "Multilingual"
                else:
                    model_size = "Default"
            else:  # Qwen3
                engine = "qwen"
                if "Small" in model_selection:
                    model_size = "0.6B"
                else:  # Large
                    model_size = "1.7B"

            # Find the selected sample
            samples = get_available_samples()
            sample = None
            for s in samples:
                if s["name"] == sample_name:
                    sample = s
                    break

            if not sample:
                return None, f"❌ Sample '{sample_name}' not found."

            # Check that sample has a transcript (required for all engines)
            sample_ref_text = sample.get("ref_text") or sample.get("meta", {}).get("Text", "")
            if not sample_ref_text.strip():
                return None, (
                    f"❌ No transcript found for sample '{sample_name}'.\n\n"
                    "Please transcribe this sample first in the **Prep Audio** tab "
                    "(using Whisper or VibeVoice ASR), then try again."
                )

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                set_seed(seed)
                seed_msg = f"Seed: {seed}"

                if engine == "qwen":
                    # Qwen engine - uses cached prompts
                    progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")

                    # Get or create the voice prompt (with caching)
                    model = tts_manager.get_qwen3_base(model_size)
                    prompt_items, was_cached = get_or_create_voice_prompt(
                        model=model,
                        sample_name=sample_name,
                        wav_path=sample["wav_path"],
                        ref_text=sample["ref_text"],
                        model_size=model_size,
                        progress_callback=progress
                    )

                    cache_status = "cached" if was_cached else "newly processed"
                    progress(0.6, desc=f"Generating audio ({cache_status} prompt)...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_qwen(
                        text=text_to_generate,
                        language=language,
                        prompt_items=prompt_items,
                        seed=seed,
                        do_sample=qwen_do_sample,
                        temperature=qwen_temperature,
                        top_k=qwen_top_k,
                        top_p=qwen_top_p,
                        repetition_penalty=qwen_repetition_penalty,
                        max_new_tokens=qwen_max_new_tokens,
                        model_size=model_size
                    )
                    wavs = [audio_data]

                    engine_display = f"Qwen3-{model_size}"

                elif engine == "vibevoice":
                    progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_vibevoice(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        seed=seed,
                        do_sample=vv_do_sample,
                        temperature=vv_temperature,
                        top_k=vv_top_k,
                        top_p=vv_top_p,
                        repetition_penalty=vv_repetition_penalty,
                        cfg_scale=vv_cfg_scale,
                        num_steps=vv_num_steps,
                        sentences_per_chunk=int(vv_sentences_per_chunk),
                        model_size=model_size,
                        user_config=shared_state.get('_user_config', {})
                    )
                    wavs = [audio_data]

                    engine_display = f"VibeVoice-{model_size}"
                    cache_status = "no caching (VibeVoice)"

                elif engine == "luxtts":
                    progress(0.05, desc="Loading LuxTTS model...")

                    # Generate using manager method (with prompt caching)
                    audio_data, sr, was_cached = tts_manager.generate_voice_clone_luxtts(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        sample_name=sample_name,
                        num_steps=int(lux_num_steps),
                        t_shift=float(lux_t_shift),
                        speed=float(lux_speed),
                        return_smooth=bool(lux_return_smooth),
                        rms=float(lux_rms),
                        ref_duration=int(lux_ref_duration),
                        guidance_scale=float(lux_guidance_scale),
                        seed=seed,
                        ref_text=sample.get("ref_text") or sample.get("meta", {}).get("Text"),
                        progress_callback=progress,
                    )
                    wavs = [audio_data]

                    engine_display = "LuxTTS"
                    cache_status = "cached" if was_cached else "newly processed"

                elif engine == "chatterbox":
                    if model_size == "Multilingual":
                        progress(0.1, desc="Loading Chatterbox Multilingual model...")
                        from modules.core_components.constants import CHATTERBOX_LANG_TO_CODE
                        lang_code = CHATTERBOX_LANG_TO_CODE.get(cb_language, "en")
                        audio_data, sr = tts_manager.generate_voice_clone_chatterbox_multilingual(
                            text=text_to_generate,
                            language_code=lang_code,
                            voice_sample_path=sample["wav_path"],
                            seed=seed,
                            exaggeration=float(cb_exaggeration),
                            cfg_weight=float(cb_cfg_weight),
                            temperature=float(cb_temperature),
                            repetition_penalty=float(cb_repetition_penalty),
                            top_p=float(cb_top_p),
                        )
                        engine_display = "Chatterbox Multilingual"
                    else:
                        progress(0.1, desc="Loading Chatterbox TTS model...")
                        audio_data, sr = tts_manager.generate_voice_clone_chatterbox(
                            text=text_to_generate,
                            voice_sample_path=sample["wav_path"],
                            seed=seed,
                            exaggeration=float(cb_exaggeration),
                            cfg_weight=float(cb_cfg_weight),
                            temperature=float(cb_temperature),
                            repetition_penalty=float(cb_repetition_penalty),
                            top_p=float(cb_top_p),
                        )
                        engine_display = "Chatterbox"

                    wavs = [audio_data]
                    cache_status = "no caching (Chatterbox)"

                progress(0.8, desc="Saving audio...")
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                filename_stem = f"{safe_name}_{timestamp}"

                # Build metadata text
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Sample: {sample_name}
                    Engine: {engine_display}
                    Language: {language}
                    Seed: {seed}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                # Always save WAV to temp first
                temp_path = save_audio_to_temp(wavs[0], sr, TEMP_DIR, filename_stem)

                manual_save = _user_config.get("manual_save", False)
                if manual_save:
                    # Leave in temp, user decides whether to save
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(temp_path), f"Generated using {engine_display}. {cache_status}\n{seed_msg}\nClick 'Save to Output' to keep this result.", metadata_out, gr.update(interactive=True)
                else:
                    # Auto-save to output in chosen format
                    output_format = _user_config.get("output_format", "wav")
                    output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return str(output_path), f"Generated using {engine_display}. {cache_status}\n{seed_msg}", "", gr.update()

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def load_sample_from_lister(lister_value):
            """Load audio, text, and info for the selected sample from FileLister."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name)

        # Connect event handlers for Voice Clone tab
        components['sample_lister'].change(
            load_sample_from_lister,
            inputs=[components['sample_lister']],
            outputs=[components['sample_audio'], components['sample_text'], components['sample_info']]
        )

        # Double-click = play sample audio
        components['sample_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-clone-sample-audio .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # Auto-refresh samples when tab is selected (preserve selection)
        def refresh_samples_keep_selection(lister_value):
            """Refresh sample list while preserving the current selection."""
            new_files = get_sample_choices()
            prev_selected = []
            if lister_value:
                prev = lister_value.get("selected", [])
                new_names = set(new_files)
                prev_selected = [s for s in prev if s in new_names]
            return {"files": [{"name": f, "date": ""} for f in new_files], "selected": prev_selected}

        components['voice_clone_tab'].select(
            refresh_samples_keep_selection,
            inputs=[components['sample_lister']],
            outputs=[components['sample_lister']]
        )

        # Wire up emotion preset handlers (same pattern as original)
        if 'update_from_emotion' in components.get('qwen_params', {}):
            components['qwen_emotion_preset'].change(
                components['qwen_params']['update_from_emotion'],
                inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
                outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty']]
            )

            components['qwen_emotion_intensity'].change(
                components['qwen_params']['update_from_emotion'],
                inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
                outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty']]
            )

        # Emotion save button
        components['qwen_save_emotion_btn'].click(
            fn=None,
            inputs=[components['qwen_emotion_preset']],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="qwen_emotion_"
            )
        )

        # Emotion delete button
        components['qwen_delete_emotion_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="qwen_emotion_"
            )
        )

        # Handler for emotion save from input modal
        def handle_qwen_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Clone emotion save."""
            if not input_value or not input_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), ""
                emotion_name = "_".join(parts[2:-1])

                # Use shared helper to process save result
                from modules.core_components.emotion_manager import process_save_emotion_result
                save_result = save_emotion_handler(emotion_name, intensity, temp, rep_pen, top_p)
                return process_save_emotion_result(save_result, shared_state)

            return gr.update(), gr.update()

        input_trigger.change(
            handle_qwen_emotion_input,
            inputs=[input_trigger, components['qwen_emotion_intensity'], components['qwen_temperature'],
                    components['qwen_repetition_penalty'], components['qwen_top_p']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        # Handler for emotion delete from confirmation modal
        def delete_qwen_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches qwen_emotion_."""
            if not confirm_value or not confirm_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            # Call the delete handler and discard the clear_trigger (3rd value)
            from modules.core_components.emotion_manager import process_delete_emotion_result
            delete_result = delete_emotion_handler(confirm_value, emotion_name)
            dropdown_update, status_msg, _clear = process_delete_emotion_result(delete_result, shared_state)
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_qwen_emotion_wrapper,
            inputs=[confirm_trigger, components['qwen_emotion_preset']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        def generate_from_lister(lister_value, *args):
            """Extract sample name from lister and pass to generate."""
            return generate_audio_handler(get_selected_sample_name(lister_value), *args)

        components['generate_btn'].click(
            restore_fn, outputs=restore_outputs
        ).then(
            generate_from_lister,
            inputs=[components['sample_lister'], components['text_input'], components['language_dropdown'], components['seed_input'], components['clone_model_dropdown'],
                    components['qwen_do_sample'], components['qwen_temperature'], components['qwen_top_k'], components['qwen_top_p'], components['qwen_repetition_penalty'],
                    components['qwen_max_new_tokens'],
                    components['vv_do_sample'], components['vv_temperature'], components['vv_top_k'], components['vv_top_p'], components['vv_repetition_penalty'],
                    components['vv_cfg_scale'], components['vv_num_steps'], components['vv_sentences_per_chunk'],
                    components['luxtts_num_steps'], components['luxtts_t_shift'], components['luxtts_speed'], components['luxtts_return_smooth'],
                    components['luxtts_rms'], components['luxtts_ref_duration'], components['luxtts_guidance_scale'],
                    components['cb_exaggeration'], components['cb_cfg_weight'], components['cb_temperature'],
                    components['cb_repetition_penalty'], components['cb_top_p'], components['cb_language_dropdown']],

            outputs=[components['output_audio'], components['clone_status'], components['_result_metadata'], components['save_result_btn']]
        )

        # Save result button handler
        def save_result_handler(audio_path, metadata_text):
            """Save the temp result to output folder in chosen format."""
            if not audio_path:
                return "❌ No audio to save.", gr.update()
            try:
                output_format = _user_config.get("output_format", "wav")
                output_path = save_result_to_output(audio_path, OUTPUT_DIR, output_format, metadata_text or None)
                return f"Saved to: {output_path.name}", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Error saving: {str(e)}", gr.update()

        components['save_result_btn'].click(
            save_result_handler,
            inputs=[components['output_audio'], components['_result_metadata']],
            outputs=[components['clone_status'], components['save_result_btn']]
        )

        # Toggle language visibility based on model selection
        def toggle_language_visibility(model_selection):
            is_qwen = "Qwen" in model_selection
            is_cb_mtl = model_selection == "Chatterbox - Multilingual"
            return gr.update(visible=is_qwen), gr.update(visible=is_cb_mtl)

        components['clone_model_dropdown'].change(
            toggle_language_visibility,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['language_row'], components['cb_language_row']]
        )

        # Toggle accordion visibility based on engine
        def toggle_engine_params(model_selection):
            is_qwen = "Qwen" in model_selection
            is_vv = "VibeVoice" in model_selection
            is_lux = "LuxTTS" in model_selection
            is_cb = "Chatterbox" in model_selection
            return gr.update(visible=is_qwen), gr.update(visible=is_vv), gr.update(visible=is_lux), gr.update(visible=is_cb)

        components['clone_model_dropdown'].change(
            toggle_engine_params,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['qwen_params_accordion'], components['vv_params_accordion'], components['luxtts_params_accordion'], components['cb_params_accordion']]
        )

        # Save voice clone model selection
        components['clone_model_dropdown'].change(
            lambda x: save_preference("voice_clone_model", x),
            inputs=[components['clone_model_dropdown']],
            outputs=[]
        )

        # Save language selection
        components['language_dropdown'].change(
            lambda x: save_preference("language", x),
            inputs=[components['language_dropdown']],
            outputs=[]
        )

        # Refresh emotion dropdowns and auto-load first sample when tab is selected
        def on_tab_select(lister_value):
            """When tab is selected, refresh emotions and auto-load sample if not loaded."""
            emotion_update = gr.update(choices=shared_state['get_emotion_choices'](shared_state['_active_emotions']))

            # Auto-load selected sample
            sample_name = get_selected_sample_name(lister_value)
            if sample_name:
                audio, text, info = load_sample_details(sample_name)
                return emotion_update, audio, text, info

            return emotion_update, None, "", ""

        components['voice_clone_tab'].select(
            on_tab_select,
            inputs=[components['sample_lister']],
            outputs=[components['qwen_emotion_preset'], components['sample_audio'],
                     components['sample_text'], components['sample_info']]
        )

        # Restore saved params when accordion is opened
        components['qwen_params_accordion'].expand(restore_fn, outputs=restore_outputs)
        components['vv_params_accordion'].expand(restore_fn, outputs=restore_outputs)
        components['luxtts_params_accordion'].expand(restore_fn, outputs=restore_outputs)
        components['cb_params_accordion'].expand(restore_fn, outputs=restore_outputs)


# Export for tab registry
get_tool_class = lambda: VoiceCloneTool


if __name__ == "__main__":
    """Standalone testing of Voice Clone tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceCloneTool, port=7862, title="Voice Clone - Standalone")
