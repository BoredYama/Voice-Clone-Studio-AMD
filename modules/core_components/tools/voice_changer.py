"""
Voice Changer Tab

Change the voice in an audio file to match a target voice using Chatterbox VC.
No text needed — works directly on audio.

Standalone testing:
    python -m modules.core_components.tools.voice_changer
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
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from gradio_filelister import FileLister


class VoiceChangerTool(Tool):
    """Voice Changer tool implementation."""

    config = ToolConfig(
        name="Voice Changer",
        module_name="tool_voice_changer",
        description="Change the voice in audio to match a target voice (Chatterbox VC)",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Changer tool UI."""
        components = {}

        get_sample_choices = shared_state['get_sample_choices']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']

        with gr.TabItem("Voice Changer") as voice_changer_tab:
            components['voice_changer_tab'] = voice_changer_tab
            gr.Markdown("Change the voice in any audio to match a target voice sample. <small>(Uses Chatterbox VC — no text input needed)</small>")
            with gr.Row():
                # Left column — Target Voice Sample (matches Voice Clone layout)
                with gr.Column(scale=1):
                    gr.Markdown("### Target Voice")

                    components['target_lister'] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    with gr.Row():
                        pass

                    components['target_audio'] = gr.Audio(
                        label="Target Voice Preview",
                        type="filepath",
                        interactive=False,
                        visible=True,
                        value=None,
                        elem_id="voice-convert-target-audio"
                    )

                    components['target_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['target_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                # Right column — Source Audio & Output (matches Voice Clone's right column)
                with gr.Column(scale=3):
                    gr.Markdown("### Source Audio")

                    components['source_audio'] = gr.Audio(
                        label="Upload or record the audio whose voice you want to change",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )

                    with gr.Row():
                        components['vc_clear_btn'] = gr.Button("Clear", scale=1, size="sm")
                        components['vc_clean_btn'] = gr.Button("AI Denoise", scale=2, size="sm", visible=DEEPFILTER_AVAILABLE)
                        components['vc_normalize_btn'] = gr.Button("Normalize Volume", scale=2, size="sm")
                        components['vc_mono_btn'] = gr.Button("Convert to Mono", scale=2, size="sm")

                    components['convert_btn'] = gr.Button(
                        "Convert Voice", variant="primary", size="lg"
                    )

                    components['output_audio'] = gr.Audio(
                        label="Converted Audio",
                        type="filepath",
                    )

                    with gr.Row():
                        components['save_btn'] = gr.Button(
                            "Save to Output", variant="primary", interactive=False
                        )

                    components['convert_status'] = gr.Textbox(
                        label="Status", interactive=False, lines=2, max_lines=5
                    )

                    # Hidden state for metadata and temp path
                    components['temp_output_path'] = gr.State(value=None)
                    components['suggested_name'] = gr.State(value="")
                    components['metadata_text'] = gr.State(value="")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Changer tab events."""

        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        TEMP_DIR = shared_state['TEMP_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        show_input_modal_js = shared_state['show_input_modal_js']
        input_trigger = shared_state['input_trigger']
        convert_audio_format = shared_state['convert_audio_format']
        embed_metadata = shared_state['embed_metadata']
        user_config = shared_state.get('_user_config', {})

        # Audio processing utilities
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']

        tts_manager = get_tts_manager()

        def get_selected_sample_name(lister_value):
            """Extract selected sample name from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def on_target_select(lister_value):
            """Load target voice preview, text, and info when selected."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name)

        def convert_voice(source_audio, target_lister_value, progress=gr.Progress()):
            """Run voice conversion."""
            if source_audio is None:
                return None, gr.update(interactive=False), None, "", "", "Please upload or record source audio."

            target_name = get_selected_sample_name(target_lister_value)
            if not target_name:
                return None, gr.update(interactive=False), None, "", "", "Please select a target voice sample."

            # Find target sample wav path
            samples = get_available_samples()

            target_wav = None
            for s in samples:
                if s["name"] == target_name:
                    target_wav = s["wav_path"]
                    break

            if not target_wav:
                return None, gr.update(interactive=False), None, "", "", f"Target sample '{target_name}' not found."

            try:
                progress(0.1, desc="Loading Chatterbox VC model...")

                # source_audio is a filepath from gr.Audio(type="filepath")
                src_temp = source_audio

                progress(0.4, desc="Converting voice...")
                audio_data, sr = tts_manager.generate_voice_convert_chatterbox(
                    source_audio_path=src_temp,
                    target_voice_path=target_wav,
                )

                progress(0.8, desc="Saving to temp...")

                # Save to temp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_target = "".join(c if c.isalnum() else "_" for c in target_name)
                temp_filename = f"vc_{safe_target}_{timestamp}.wav"
                temp_path = str(TEMP_DIR / temp_filename)
                sf.write(temp_path, audio_data, sr)

                suggested = f"vc_{safe_target}"

                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Voice Conversion
                    Target Voice: {target_name}
                    Engine: Chatterbox VC
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()

                return (
                    temp_path,
                    gr.update(interactive=True),
                    temp_path,
                    suggested,
                    metadata_out,
                    f"Voice changed to match '{target_name}'."
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = str(e)
                if "Content-Length" in msg or "Too little data" in msg:
                    msg = "Recording too short. Please record at least 10 seconds of audio."
                else:
                    msg = f"❌ Error: {msg}"
                return None, gr.update(interactive=False), None, "", "", msg

        components['convert_btn'].click(
            convert_voice,
            inputs=[components['source_audio'], components['target_lister']],
            outputs=[
                components['output_audio'],
                components['save_btn'],
                components['temp_output_path'],
                components['suggested_name'],
                components['metadata_text'],
                components['convert_status'],
            ]
        )

        # --- Audio processing buttons ---
        components['vc_clear_btn'].click(
            lambda: (None, ""),
            outputs=[components['source_audio'], components['convert_status']]
        )

        components['vc_normalize_btn'].click(
            normalize_audio,
            inputs=[components['source_audio']],
            outputs=[components['source_audio'], components['convert_status']]
        )

        components['vc_mono_btn'].click(
            convert_to_mono,
            inputs=[components['source_audio']],
            outputs=[components['source_audio'], components['convert_status']]
        )

        components['vc_clean_btn'].click(
            clean_audio,
            inputs=[components['source_audio']],
            outputs=[components['source_audio'], components['convert_status']]
        )

        # Target voice preview + text + info
        components['target_lister'].change(
            on_target_select,
            inputs=[components['target_lister']],
            outputs=[components['target_audio'], components['target_text'], components['target_info']]
        )

        # Double-click = play target audio
        components['target_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-convert-target-audio .play-pause-button'); if (btn) btn.click(); }, 150); }"
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

        components['voice_changer_tab'].select(
            refresh_samples_keep_selection,
            inputs=[components['target_lister']],
            outputs=[components['target_lister']]
        )

        # Save workflow: button → input modal → copy to output
        save_vc_modal_js = show_input_modal_js(
            title="Save Voice Changer Output",
            message="Enter a filename for this changed audio:",
            placeholder="e.g., vc_my_voice, converted_sample",
            context="save_vc_"
        )

        components['save_btn'].click(
            fn=None,
            inputs=[components['suggested_name']],
            outputs=None,
            js=f"""
            (suggestedName) => {{
                const openModal = {save_vc_modal_js};
                openModal(suggestedName);
            }}
            """
        )

        def handle_vc_input_modal(input_value, temp_path, metadata_text):
            """Process save modal submission for Voice Changer."""
            if not input_value or not input_value.startswith("save_vc_"):
                return gr.update(), gr.update()

            # Check for cancel
            raw = input_value[len("save_vc_"):]
            parts = raw.rsplit("_", 1)
            if len(parts) >= 2 and parts[0] == "cancel":
                return gr.update(), gr.update()

            chosen_name = parts[0] if len(parts) > 1 else raw

            # Clean filename
            clean_name = "".join(
                c if c.isalnum() or c in "-_ " else "" for c in chosen_name
            ).strip().replace(" ", "_")

            if not clean_name:
                clean_name = "voice_changer"

            if not temp_path or not Path(temp_path).exists():
                return gr.update(interactive=False), "❌ Temp file not found. Please convert again."

            # Copy to output folder in chosen format
            output_format = user_config.get("output_format", "wav")
            output_path = OUTPUT_DIR / f"{clean_name}.{output_format}"
            # Avoid overwriting
            counter = 1
            while output_path.exists():
                output_path = OUTPUT_DIR / f"{clean_name}_{counter}.{output_format}"
                counter += 1

            convert_audio_format(temp_path, output_path, output_format)

            # Embed metadata in audio file
            if metadata_text:
                embed_metadata(output_path, metadata_text)

            return gr.update(interactive=False), f"Saved as {output_path.name}"

        input_trigger.change(
            handle_vc_input_modal,
            inputs=[input_trigger, components['temp_output_path'], components['metadata_text']],
            outputs=[components['save_btn'], components['convert_status']]
        )


# Export for tab registry
get_tool_class = lambda: VoiceChangerTool


if __name__ == "__main__":
    """Standalone testing of Voice Changer tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceChangerTool, port=7866, title="Voice Changer - Standalone")
