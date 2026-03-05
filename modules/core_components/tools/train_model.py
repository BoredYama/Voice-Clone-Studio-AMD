"""
Train Model Tab

Train custom voice models using finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from gradio_filelister import FileLister
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.constants import (
    QWEN_TRAINING_DEFAULTS,
    VIBEVOICE_TRAINING_DEFAULTS,
)


class TrainModelTool(Tool):
    """Train Model tool implementation."""

    config = ToolConfig(
        name="Train Model",
        module_name="tool_train_model",
        description="Train custom voice models",
        enabled=True,
        category="training"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Train Model tool UI.

        Sliders are initialized with default values from constants.
        Saved user preferences are applied on tab select via
        create_param_restore_handler (same pattern as voice_clone).
        """
        components = {}

        format_help_html = shared_state['format_help_html']
        get_dataset_folders = shared_state['get_dataset_folders']
        _user_config = shared_state['_user_config']

        saved_model_type = _user_config.get("train_model_type", "Qwen3")
        is_qwen = saved_model_type == "Qwen3"

        q = QWEN_TRAINING_DEFAULTS
        vv = VIBEVOICE_TRAINING_DEFAULTS

        with gr.TabItem("Train Model") as train_tab:
            components['train_tab'] = train_tab
            gr.Markdown("Train a custom voice model using your finetuning dataset")
            with gr.Row():
                # Left column - Dataset selection
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Selection")

                    components['model_type_radio'] = gr.Radio(
                        choices=["Qwen3", "VibeVoice"],
                        value=saved_model_type,
                        label="Model Type",
                        info="Select training engine"
                    )

                    components['train_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Training Dataset",
                        info="Select prepared subfolder",
                        interactive=True
                    )

                    components['refresh_train_folder_btn'] = gr.Button("Refresh Datasets", size="sm", visible=False)

                    # Qwen3: needs reference audio selection (starts visible=True for DOM rendering)
                    with gr.Group(visible=True) as qwen_ref_section:
                        components['qwen_ref_section'] = qwen_ref_section

                        components['ref_audio_lister'] = FileLister(
                            value=[],
                            height=150,
                            show_footer=False,
                            interactive=True,
                        )

                        components['ref_audio_preview'] = gr.Audio(
                            label="Preview",
                            type="filepath",
                            interactive=False,
                            elem_id="train-ref-audio-preview"
                        )

                    with gr.Row():
                        components['start_training_btn'] = gr.Button("Start Training", variant="primary", size="lg")
                        components['stop_training_btn'] = gr.Button("Stop Training", variant="stop", size="lg", interactive=False)

                    train_quick_guide = dedent("""\
                        **Quick Guide:**
                        1. Select model type (Qwen3 or VibeVoice)
                        2. Select dataset folder
                        3. (Qwen3 only) Select a reference audio file from the dataset
                        4. Configure parameters as needed (optional)
                        5. Start training & Enter Name (defaults work well for most cases)

                        *See Help Guide tab -> Train Model for detailed instructions*
                    """)
                    gr.HTML(
                        value=format_help_html(train_quick_guide),
                        container=True,
                        padding=True)

                # Right column - Training configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")
                    with gr.Accordion("Training Settings", open=False) as train_accordion:
                        components['train_accordion'] = train_accordion

                        # --- Qwen3 training parameters ---
                        with gr.Group(visible=True) as qwen_params_section:
                            components['qwen_params_section'] = qwen_params_section

                            with gr.Row():
                                components['qwen_batch_size'] = gr.Slider(
                                    minimum=1, maximum=10, value=q["batch_size"], step=1,
                                    label="Batch Size",
                                    info="Reduce if you get out of memory errors"
                                )

                                components['qwen_learning_rate'] = gr.Slider(
                                    minimum=1e-6, maximum=1e-3, value=q["learning_rate"],
                                    label="Learning Rate",
                                    info="Default: 2e-6"
                                )

                            with gr.Row():
                                components['qwen_num_epochs'] = gr.Slider(
                                    minimum=1, maximum=100, value=q["num_epochs"], step=1,
                                    label="Number of Epochs",
                                    info="How many times to train on the full dataset"
                                )

                                components['qwen_save_interval'] = gr.Slider(
                                    minimum=0, maximum=10, value=q["save_interval"], step=1,
                                    label="Save Interval (Epochs)",
                                    info="Save checkpoint every N epochs (0 = no intermediate saves)"
                                )

                        # --- VibeVoice training parameters ---
                        with gr.Group(visible=True) as vv_params_section:
                            components['vv_params_section'] = vv_params_section

                            with gr.Row():
                                components['vv_batch_size'] = gr.Slider(
                                    minimum=1, maximum=10, value=vv["batch_size"], step=1,
                                    label="Batch Size",
                                    info="Reduce if you get out of memory errors"
                                )

                                components['vv_learning_rate'] = gr.Slider(
                                    minimum=1e-6, maximum=1e-3, value=vv["learning_rate"],
                                    label="Learning Rate",
                                    info="Default: 5e-5"
                                )

                            with gr.Row():
                                components['vv_num_epochs'] = gr.Slider(
                                    minimum=1, maximum=100, value=vv["num_epochs"], step=1,
                                    label="Number of Epochs",
                                    info="How many times to train on the full dataset"
                                )

                                components['vv_save_interval'] = gr.Slider(
                                    minimum=0, maximum=10, value=vv["save_interval"], step=1,
                                    label="Save Interval (Epochs)",
                                    info="Save checkpoint every N epochs (0 = no intermediate saves)"
                                )

                            gr.Markdown("#### VibeVoice Advanced")

                            with gr.Row():
                                components['vv_ddpm_batch_mul'] = gr.Slider(
                                    minimum=1, maximum=16, value=vv["ddpm_batch_mul"], step=1,
                                    label="DDPM Batch Multiplier",
                                    info="Repeat diffusion samples per batch (4 recommended)"
                                )

                                components['vv_diffusion_loss_weight'] = gr.Slider(
                                    minimum=0.0, maximum=5.0, value=vv["diffusion_loss_weight"], step=0.1,
                                    label="Diffusion Loss Weight",
                                    info="Weight for diffusion head loss"
                                )

                            with gr.Row():
                                components['vv_ce_loss_weight'] = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=vv["ce_loss_weight"], step=0.01,
                                    label="CE Loss Weight",
                                    info="Weight for cross-entropy loss"
                                )

                                components['vv_voice_prompt_drop'] = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=vv["voice_prompt_drop"], step=0.1,
                                    label="Voice Prompt Drop Rate",
                                    info="1.0 = drop all prompts (single-speaker). 0.0 = keep all."
                                )

                            with gr.Row():
                                components['vv_gradient_accumulation'] = gr.Slider(
                                    minimum=1, maximum=32, value=vv["gradient_accumulation"], step=1,
                                    label="Gradient Accumulation Steps",
                                    info="Effective batch size = batch_size * gradient_accumulation"
                                )

                                components['vv_warmup_steps'] = gr.Slider(
                                    minimum=0, maximum=500, value=vv["warmup_steps"], step=1,
                                    label="Warmup Steps",
                                    info="Linearly ramp up LR (keep low for small datasets)"
                                )

                            with gr.Row():
                                components['vv_train_diffusion_head'] = gr.Checkbox(
                                    value=vv["train_diffusion_head"],
                                    label="Train Diffusion Head (default: True)",
                                    info="Full fine-tune of the diffusion prediction head"
                                )

                                components['vv_ema_enabled'] = gr.Checkbox(
                                    value=vv["ema_enabled"],
                                    label="EMA (default: On)",
                                    info="Smooths diffusion head weights during training (decay 0.99)"
                                )

                    components['training_status'] = gr.Textbox(
                        label="Status",
                        lines=3,
                        max_lines=20,
                        interactive=False
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Train Model tab events."""

        get_dataset_files = shared_state['get_dataset_files']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_trained_model_names = shared_state['get_trained_model_names']
        train_model = shared_state['train_model']
        train_vibevoice_model = shared_state['train_vibevoice_model']
        stop_training = shared_state['stop_training']
        tts_manager = shared_state.get('tts_manager')
        asr_manager = shared_state.get('asr_manager')
        input_trigger = shared_state['input_trigger']
        show_input_modal_js = shared_state['show_input_modal_js']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        save_preference = shared_state['save_preference']
        _user_config = shared_state['_user_config']
        wire_param_persistence = shared_state['wire_param_persistence']
        create_param_restore_handler = shared_state['create_param_restore_handler']

        def get_selected_ref_filename(lister_value):
            """Extract selected filename from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                return selected[0]
            return None

        # --- Model type toggle: show/hide sections ---
        def toggle_model_type(model_type):
            """Toggle visibility of Qwen3 vs VibeVoice specific sections."""
            is_qwen = model_type == "Qwen3"
            return (
                gr.update(visible=is_qwen),      # qwen_ref_section
                gr.update(visible=is_qwen),      # qwen_params_section
                gr.update(visible=not is_qwen),  # vv_params_section
            )

        components['model_type_radio'].change(
            toggle_model_type,
            inputs=[components['model_type_radio']],
            outputs=[
                components['qwen_ref_section'],
                components['qwen_params_section'],
                components['vv_params_section'],
            ]
        )

        # --- Folder change: update ref audio lister ---
        def update_ref_audio_lister(folder):
            """Update reference audio lister when folder changes."""
            files = get_dataset_files(folder)
            return files, None

        components['train_folder_dropdown'].change(
            update_ref_audio_lister,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['ref_audio_lister'], components['ref_audio_preview']]
        )

        # Auto-refresh datasets when tab is selected (preserve selection)
        def refresh_datasets_keep_selection(current_folder):
            """Refresh dataset list while preserving the current selection."""
            folder_choices = ["(Select Dataset)"] + get_dataset_folders()
            if current_folder and current_folder in folder_choices:
                return gr.update(choices=folder_choices, value=current_folder)
            return gr.update(choices=folder_choices, value="(Select Dataset)")

        # --- Ref audio preview on selection ---
        def load_ref_audio_preview(lister_value, folder):
            """Load reference audio preview from FileLister selection."""
            filename = get_selected_ref_filename(lister_value)
            if not folder or not filename or folder in ("(No folders)", "(Select Dataset)"):
                return None
            audio_path = DATASETS_DIR / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['ref_audio_lister'].change(
            load_ref_audio_preview,
            inputs=[components['ref_audio_lister'], components['train_folder_dropdown']],
            outputs=[components['ref_audio_preview']]
        )

        # Double-click = play preview
        components['ref_audio_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#train-ref-audio-preview .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # --- Settings persistence (same pattern as voice_clone) ---
        # Save model type on change
        components['model_type_radio'].change(
            lambda x: save_preference("train_model_type", x),
            inputs=[components['model_type_radio']],
            outputs=[]
        )

        # Auto-save & restore training parameters per engine
        param_map = {
            'training_qwen': [
                ('qwen_batch_size', 'batch_size'),
                ('qwen_learning_rate', 'learning_rate'),
                ('qwen_num_epochs', 'num_epochs'),
                ('qwen_save_interval', 'save_interval'),
            ],
            'training_vv': [
                ('vv_batch_size', 'batch_size'),
                ('vv_learning_rate', 'learning_rate'),
                ('vv_num_epochs', 'num_epochs'),
                ('vv_save_interval', 'save_interval'),
                ('vv_ddpm_batch_mul', 'ddpm_batch_mul'),
                ('vv_diffusion_loss_weight', 'diffusion_loss_weight'),
                ('vv_ce_loss_weight', 'ce_loss_weight'),
                ('vv_voice_prompt_drop', 'voice_prompt_drop'),
                ('vv_train_diffusion_head', 'train_diffusion_head'),
                ('vv_gradient_accumulation', 'gradient_accumulation'),
                ('vv_warmup_steps', 'warmup_steps'),
                ('vv_ema_enabled', 'ema_enabled'),
            ],
        }

        wire_param_persistence(components, _user_config, param_map)

        restore_fn, restore_outputs = create_param_restore_handler(
            components, _user_config, param_map
        )

        # Restore saved params when accordion is opened
        components['train_accordion'].expand(restore_fn, outputs=restore_outputs)

        # --- Start Training: 2-step modal with dynamic validation ---
        # Hidden JSON to pass existing model names to JS for validation
        components['existing_models_json'] = gr.JSON(value=[], visible=False)
        # Hidden state to pass model type to modal handler
        components['train_model_type_state'] = gr.State(value="Qwen3")

        def fetch_existing_models_and_type(model_type):
            """Fetch current model list and model type before opening modal."""
            return get_trained_model_names(), model_type

        # Build the base modal JS using show_input_modal_js
        base_modal_js = show_input_modal_js(
            title="Start Training",
            message="Enter a name for this trained voice model:",
            placeholder="e.g., MyVoice, Female-Narrator, John-Doe",
            submit_button_text="Start Training",
            context="train_model_"
        )

        # Wrap to inject validation and existing-model overwrite confirmation
        open_modal_js = f"""
        (existingModels) => {{
            window.inputModalValidation = (value) => {{
                if (!value || value.trim().length === 0) {{
                    return 'Please enter a model name';
                }}
                return null;
            }};
            window.inputModalExistingFiles = existingModels || [];
            const openModal = {base_modal_js};
            openModal('');
        }}
        """

        # Apply saved params then open modal
        components['start_training_btn'].click(
            fn=restore_fn,
            outputs=restore_outputs
        ).then(
            fn=fetch_existing_models_and_type,
            inputs=[components['model_type_radio']],
            outputs=[components['existing_models_json'], components['train_model_type_state']]
        ).then(
            fn=None,
            inputs=[components['existing_models_json']],
            outputs=None,
            js=open_modal_js
        )

        # --- Handle training modal submission ---
        def activate_stop_btn(input_value):
            """Enable stop button when training is about to start."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update(), gr.update()
            # Disable Start, enable Stop
            return gr.update(interactive=False), gr.update(interactive=True)

        def handle_train_model_input(input_value, model_type, folder, ref_lister,
                                     qwen_batch_size, qwen_lr, qwen_epochs, qwen_save_interval,
                                     vv_batch_size, vv_lr, vv_epochs, vv_save_interval,
                                     vv_ddpm_batch_mul, vv_diffusion_loss_weight,
                                     vv_ce_loss_weight, vv_voice_prompt_drop,
                                     vv_train_diffusion_head, vv_gradient_accumulation,
                                     vv_warmup_steps, vv_ema_enabled, progress=gr.Progress()):
            """Process input modal submission for training."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update()

            # Format: "train_model_SpeakerName_timestamp"
            parts = input_value.split("_")
            if len(parts) < 3:
                return gr.update()

            speaker_name = "_".join(parts[2:-1])

            # Unload all models to free VRAM before training
            if tts_manager:
                tts_manager.unload_all()
            if asr_manager:
                asr_manager.unload_all()

            if model_type == "VibeVoice":
                return train_vibevoice_model(
                    folder, speaker_name, vv_batch_size, vv_lr, vv_epochs,
                    vv_save_interval,
                    vv_ddpm_batch_mul, vv_diffusion_loss_weight,
                    vv_ce_loss_weight, vv_voice_prompt_drop,
                    vv_train_diffusion_head, vv_gradient_accumulation,
                    vv_warmup_steps, 0.99 if vv_ema_enabled else 0.0, progress
                )
            else:
                ref_audio = get_selected_ref_filename(ref_lister)
                return train_model(
                    folder, speaker_name, ref_audio, qwen_batch_size, qwen_lr,
                    qwen_epochs, qwen_save_interval, progress
                )

        def deactivate_stop_btn():
            """Re-enable Start, disable Stop after training finishes."""
            return gr.update(interactive=True), gr.update(interactive=False)

        input_trigger.change(
            activate_stop_btn,
            inputs=[input_trigger],
            outputs=[components['start_training_btn'], components['stop_training_btn']]
        ).then(
            handle_train_model_input,
            inputs=[
                input_trigger,
                components['train_model_type_state'],
                components['train_folder_dropdown'],
                components['ref_audio_lister'],
                # Qwen params
                components['qwen_batch_size'],
                components['qwen_learning_rate'],
                components['qwen_num_epochs'],
                components['qwen_save_interval'],
                # VV params
                components['vv_batch_size'],
                components['vv_learning_rate'],
                components['vv_num_epochs'],
                components['vv_save_interval'],
                components['vv_ddpm_batch_mul'],
                components['vv_diffusion_loss_weight'],
                components['vv_ce_loss_weight'],
                components['vv_voice_prompt_drop'],
                components['vv_train_diffusion_head'],
                components['vv_gradient_accumulation'],
                components['vv_warmup_steps'],
                components['vv_ema_enabled'],
            ],
            outputs=[components['training_status']]
        ).then(
            deactivate_stop_btn,
            outputs=[components['start_training_btn'], components['stop_training_btn']]
        )

        # --- Stop Training button ---
        def handle_stop_training():
            """Stop the active training subprocess."""
            stop_training()
            return "Stopping training... please wait."

        components['stop_training_btn'].click(
            handle_stop_training,
            outputs=[components['training_status']]
        )

        # Tab select: refresh datasets and set correct section visibility
        train_toggle_outputs = [
            components['qwen_ref_section'],
            components['qwen_params_section'],
            components['vv_params_section'],
        ]

        components['train_tab'].select(
            refresh_datasets_keep_selection,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['train_folder_dropdown']]
        ).then(
            toggle_model_type,
            inputs=[components['model_type_radio']],
            outputs=train_toggle_outputs
        )

        # Set correct initial visibility on page load (tab.select doesn't fire for the first tab)
        app = shared_state.get('app')
        if app:
            app.load(
                toggle_model_type,
                inputs=[components['model_type_radio']],
                outputs=train_toggle_outputs
            )


# Export for tab registry
get_tool_class = lambda: TrainModelTool

if __name__ == "__main__":
    """Standalone testing of Train Model tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(TrainModelTool, port=7863, title="Train Model - Standalone")