"""
Emotion Management Module

Handles loading, saving, and managing emotion presets for voice generation.
Emotions are stored in a standalone emotions.json file so users can reset
config.json without losing their custom emotion presets.
"""

import json
from pathlib import Path


# Path to standalone emotions file (next to config.json in project root)
EMOTIONS_FILE = Path(__file__).parent.parent.parent / "emotions.json"


# Core emotions - hardcoded defaults (read-only backup)
CORE_EMOTIONS = {
    # Core emotions
    "angry": {"temp": 0.3, "penalty": 0.1, "top_p": 0.05},
    "sad": {"temp": -0.2, "penalty": -0.05, "top_p": -0.1},
    "happy": {"temp": 0.2, "penalty": 0.05, "top_p": 0.05},
    "fearful": {"temp": 0.25, "penalty": 0.15, "top_p": 0.0},
    "disgusted": {"temp": 0.1, "penalty": 0.1, "top_p": -0.05},
    "surprised": {"temp": 0.35, "penalty": 0.1, "top_p": 0.1},

    # Intensity variations
    "furious": {"temp": 0.5, "penalty": 0.2, "top_p": 0.1},
    "enraged": {"temp": 0.6, "penalty": 0.25, "top_p": 0.15},
    "annoyed": {"temp": 0.15, "penalty": 0.05, "top_p": 0.0},
    "melancholic": {"temp": -0.3, "penalty": -0.1, "top_p": -0.15},
    "disappointed": {"temp": -0.15, "penalty": -0.05, "top_p": -0.05},
    "ecstatic": {"temp": 0.5, "penalty": 0.2, "top_p": 0.15},
    "content": {"temp": -0.1, "penalty": 0.0, "top_p": 0.0},
    "terrified": {"temp": 0.4, "penalty": 0.25, "top_p": 0.05},
    "anxious": {"temp": 0.2, "penalty": 0.15, "top_p": 0.0},

    # Social/Interpersonal
    "sarcastic": {"temp": 0.15, "penalty": 0.05, "top_p": 0.0},
    "mocking": {"temp": 0.2, "penalty": 0.1, "top_p": 0.05},
    "condescending": {"temp": 0.1, "penalty": 0.05, "top_p": -0.05},
    "dismissive": {"temp": 0.05, "penalty": 0.0, "top_p": -0.1},
    "contemptuous": {"temp": 0.15, "penalty": 0.1, "top_p": -0.05},
    "sympathetic": {"temp": -0.1, "penalty": -0.05, "top_p": 0.0},
    "apologetic": {"temp": -0.15, "penalty": -0.05, "top_p": -0.05},
    "pleading": {"temp": 0.1, "penalty": 0.1, "top_p": 0.0},
    "begging": {"temp": 0.2, "penalty": 0.15, "top_p": 0.05},

    # Confidence spectrum
    "confident": {"temp": -0.1, "penalty": -0.05, "top_p": 0.0},
    "arrogant": {"temp": 0.0, "penalty": 0.0, "top_p": -0.1},
    "smug": {"temp": 0.05, "penalty": 0.0, "top_p": -0.05},
    "insecure": {"temp": 0.1, "penalty": 0.1, "top_p": 0.0},
    "hesitant": {"temp": 0.15, "penalty": 0.15, "top_p": 0.0},
    "timid": {"temp": -0.2, "penalty": 0.1, "top_p": -0.1},
    "defeated": {"temp": -0.25, "penalty": -0.1, "top_p": -0.1},
    "hopeless": {"temp": -0.3, "penalty": -0.15, "top_p": -0.15},

    # Energy levels
    "excited": {"temp": 0.4, "penalty": 0.15, "top_p": 0.1},
    "enthusiastic": {"temp": 0.3, "penalty": 0.1, "top_p": 0.05},
    "energetic": {"temp": 0.35, "penalty": 0.15, "top_p": 0.1},
    "bored": {"temp": -0.2, "penalty": -0.1, "top_p": -0.1},
    "tired": {"temp": -0.25, "penalty": -0.1, "top_p": -0.1},
    "exhausted": {"temp": -0.3, "penalty": -0.15, "top_p": -0.15},
    "lethargic": {"temp": -0.35, "penalty": -0.15, "top_p": -0.15},
    "calm": {"temp": -0.3, "penalty": -0.1, "top_p": -0.15},

    # Stress/Tension
    "nervous": {"temp": 0.2, "penalty": 0.2, "top_p": 0.05},
    "panicked": {"temp": 0.45, "penalty": 0.25, "top_p": 0.1},
    "frantic": {"temp": 0.5, "penalty": 0.3, "top_p": 0.15},
    "stressed": {"temp": 0.25, "penalty": 0.15, "top_p": 0.05},
    "overwhelmed": {"temp": 0.3, "penalty": 0.2, "top_p": 0.05},
    "relaxed": {"temp": -0.25, "penalty": -0.1, "top_p": -0.1},
    "serene": {"temp": -0.35, "penalty": -0.15, "top_p": -0.15},

    # Attitude/Mood
    "playful": {"temp": 0.25, "penalty": 0.1, "top_p": 0.05},
    "teasing": {"temp": 0.2, "penalty": 0.1, "top_p": 0.05},
    "flirtatious": {"temp": 0.15, "penalty": 0.05, "top_p": 0.05},
    "seductive": {"temp": 0.1, "penalty": 0.0, "top_p": 0.0},
    "mysterious": {"temp": -0.1, "penalty": 0.0, "top_p": -0.05},
    "ominous": {"temp": -0.15, "penalty": 0.05, "top_p": -0.1},
    "menacing": {"temp": 0.1, "penalty": 0.1, "top_p": -0.05},
    "threatening": {"temp": 0.2, "penalty": 0.15, "top_p": 0.0},
    "suspicious": {"temp": 0.1, "penalty": 0.1, "top_p": 0.0},
    "paranoid": {"temp": 0.25, "penalty": 0.2, "top_p": 0.05},

    # Physical states
    "drunk": {"temp": 0.4, "penalty": 0.25, "top_p": 0.1},
    "breathless": {"temp": 0.3, "penalty": 0.2, "top_p": 0.1},
    "whispering": {"temp": -0.2, "penalty": -0.05, "top_p": -0.1},
    "shouting": {"temp": 0.4, "penalty": 0.2, "top_p": 0.1},
    "in_pain": {"temp": 0.3, "penalty": 0.2, "top_p": 0.05},
    "sick": {"temp": -0.2, "penalty": -0.05, "top_p": -0.1},

    # Complex emotions (pre-mixed)
    "bitter": {"temp": 0.1, "penalty": 0.05, "top_p": -0.05},  # sad + angry
    "manic": {"temp": 0.5, "penalty": 0.3, "top_p": 0.15},  # excited + nervous
    "reluctant": {"temp": -0.1, "penalty": 0.05, "top_p": -0.05},  # defeated + nervous
    "desperate": {"temp": 0.3, "penalty": 0.2, "top_p": 0.05},  # fearful + pleading
    "jealous": {"temp": 0.2, "penalty": 0.1, "top_p": 0.0},  # angry + sad
    "frustrated": {"temp": 0.25, "penalty": 0.1, "top_p": 0.0},  # angry + defeated
    "relieved": {"temp": -0.15, "penalty": -0.05, "top_p": 0.0},  # happy + calm
    "guilty": {"temp": -0.1, "penalty": 0.05, "top_p": -0.05},  # sad + nervous
    "ashamed": {"temp": -0.2, "penalty": -0.05, "top_p": -0.1},  # defeated + sad
    "proud": {"temp": 0.0, "penalty": 0.0, "top_p": 0.0},  # confident + happy
    "vengeful": {"temp": 0.3, "penalty": 0.15, "top_p": 0.05},  # angry + confident
    "remorseful": {"temp": -0.15, "penalty": -0.05, "top_p": -0.05},  # sad + apologetic

    # Character-specific
    "rick_like": {"temp": 0.3, "penalty": 0.15, "top_p": 0.05},  # condescending + drunk + dismissive
    "jerry_like": {"temp": 0.1, "penalty": 0.15, "top_p": 0.0},  # nervous + insecure + apologetic
    "morty_like": {"temp": 0.25, "penalty": 0.25, "top_p": 0.05},  # anxious + overwhelmed
}


def _save_emotions_file(emotions):
    """Save emotions dictionary to emotions.json, sorted alphabetically.

    Args:
        emotions: Dictionary of emotion presets

    Returns:
        Sorted emotions dictionary
    """
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[0].lower()))
    with open(EMOTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_emotions, f, indent=2)
    return sorted_emotions


def _load_emotions_file():
    """Load emotions from emotions.json.

    Returns:
        Emotions dictionary, or None if file doesn't exist or is invalid
    """
    if EMOTIONS_FILE.exists():
        try:
            with open(EMOTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_emotions_from_config(config=None):
    """Load emotions from emotions.json, with migration from config.json.

    Priority:
        1. emotions.json exists -> use it
        2. config has "emotions" key -> migrate to emotions.json
        3. Neither -> create emotions.json with CORE_EMOTIONS

    Args:
        config: Optional config dictionary (used only for one-time migration
                from the old config.json format)

    Returns:
        Dictionary of emotion presets
    """
    # 1. Try emotions.json first
    emotions = _load_emotions_file()
    if emotions:
        return emotions

    # 2. Migrate from config.json if present
    if config and "emotions" in config and config["emotions"]:
        emotions = config["emotions"]
        _save_emotions_file(emotions)
        # Remove stale emotions from config.json after migration
        del config["emotions"]
        try:
            config_file = Path(__file__).parent.parent.parent / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print("Migrated emotions from config.json to emotions.json")
        except Exception:
            pass
        return emotions

    # 3. First time - create emotions.json with core defaults
    return _save_emotions_file(CORE_EMOTIONS.copy())


def save_emotion(emotions_dict, emotion_name, temp, penalty, top_p, intensity):
    """Save or update an emotion preset to emotions.json.

    Calculates differences from default values and divides by intensity multiplier.
    Saves emotions in alphabetical order (case-insensitive).

    Args:
        emotions_dict: Current emotions dictionary (will be updated)
        emotion_name: Name of the emotion
        temp: Current temperature value (with intensity applied)
        penalty: Current penalty value (with intensity applied)
        top_p: Current top_p value (with intensity applied)
        intensity: Current intensity multiplier

    Returns:
        Tuple: (success: bool, message: str, updated_emotions: dict)
    """
    if not emotion_name or not emotion_name.strip():
        return False, "❌ Emotion name cannot be empty", None

    emotion_name = emotion_name.strip()

    # Default values (neutral baseline)
    DEFAULT_TEMP = 0.9
    DEFAULT_PENALTY = 1.05
    DEFAULT_TOP_P = 1.0

    # Calculate differences from defaults, then divide by intensity
    # Formula: saved_value = (current_value - default_value) / intensity
    if intensity == 0:
        return False, "❌ Intensity cannot be zero", None

    diff_temp = round((temp - DEFAULT_TEMP) / intensity, 3)
    diff_penalty = round((penalty - DEFAULT_PENALTY) / intensity, 3)
    diff_top_p = round((top_p - DEFAULT_TOP_P) / intensity, 3)

    # Add/update emotion (store as differences)
    emotions_dict[emotion_name] = {
        "temp": diff_temp,
        "penalty": diff_penalty,
        "top_p": diff_top_p
    }

    # Save to emotions.json
    try:
        sorted_emotions = _save_emotions_file(emotions_dict)
        return True, f"Saved emotion: {emotion_name}", sorted_emotions
    except Exception as e:
        return False, f"❌ Failed to save: {str(e)}", None


def delete_emotion(emotions_dict, emotion_name):
    """Delete an emotion preset from emotions.json.

    Args:
        emotions_dict: Current emotions dictionary (will be updated)
        emotion_name: Name of the emotion to delete

    Returns:
        Tuple: (success: bool, message: str, updated_emotions: dict)
    """
    if not emotion_name or not emotion_name.strip():
        return False, "❌ No emotion selected", None

    emotion_name = emotion_name.strip()

    if emotion_name not in emotions_dict:
        return False, f"❌ Emotion not found: {emotion_name}", None

    # Prevent deleting if it's the last emotion
    if len(emotions_dict) <= 1:
        return False, "❌ Cannot delete the last emotion", None

    # Delete emotion
    del emotions_dict[emotion_name]

    # Save to emotions.json
    try:
        sorted_emotions = _save_emotions_file(emotions_dict)
        return True, f"Deleted emotion: {emotion_name}", sorted_emotions
    except Exception as e:
        return False, f"❌ Failed to delete: {str(e)}", None


def reset_emotions_to_core():
    """Reset emotions to CORE_EMOTIONS defaults.

    Returns:
        Tuple: (success: bool, message: str, updated_emotions: dict)
    """
    try:
        sorted_emotions = _save_emotions_file(CORE_EMOTIONS.copy())
        return True, "Reset to core emotions", sorted_emotions
    except Exception as e:
        return False, f"❌ Failed to reset: {str(e)}", None


def get_emotion_choices(emotions_dict):
    """Get sorted list of emotion names for dropdown.

    Args:
        emotions_dict: Dictionary of emotions

    Returns:
        List with (None) first (no emotion), then emotion names sorted alphabetically (case-insensitive)
    """
    return ["(None)"] + sorted(emotions_dict.keys(), key=lambda x: x.lower())


def calculate_emotion_values(emotions_dict, emotion_name, intensity, baseline_temp=0.9, baseline_top_p=1.0, baseline_penalty=1.05):
    """
    Calculate adjusted generation parameters based on emotion and intensity.

    Args:
        emotions_dict: Dictionary of emotion presets
        emotion_name: Name of emotion from emotions_dict or empty string
        intensity: Multiplier for emotion strength (0-2.0)
        baseline_temp: Default temperature value
        baseline_top_p: Default top_p value
        baseline_penalty: Default repetition penalty value

    Returns:
        Tuple of (temperature, top_p, repetition_penalty, intensity_to_display)
    """
    # Check if using defaults - return baseline values
    if not emotion_name or emotion_name == "(None)":
        return baseline_temp, baseline_top_p, baseline_penalty, 1.0

    # Extract emotion key from dropdown selection
    emotion_key = emotion_name.split(" (")[0] if " (" in emotion_name else emotion_name

    # Retrieve adjustment values or use zero adjustments
    adjustments = emotions_dict.get(emotion_key, {"temp": 0.0, "penalty": 0.0, "top_p": 0.0})

    # Calculate adjusted values with intensity scaling
    adjusted_temp = baseline_temp + (adjustments["temp"] * intensity)
    adjusted_top_p = baseline_top_p + (adjustments["top_p"] * intensity)
    adjusted_penalty = baseline_penalty + (adjustments["penalty"] * intensity)

    # Clamp to valid ranges
    final_temp = min(max(adjusted_temp, 0.1), 1.99)  # Max 1.99 to avoid model validation errors
    final_top_p = min(max(adjusted_top_p, 0.0), 1.0)
    final_penalty = min(max(adjusted_penalty, 1.0), 2.0)

    return final_temp, final_top_p, final_penalty, intensity


def handle_save_emotion(emotions_dict, emotion_name, intensity, temp, penalty, top_p):
    """Handle saving an emotion preset - returns data for UI update.

    Args:
        emotions_dict: Current emotions dictionary
        emotion_name: Name to save
        intensity: Current intensity value
        temp: Current temperature value
        penalty: Current penalty value
        top_p: Current top_p value

    Returns:
        Tuple: (success: bool, message: str, updated_emotions: dict, new_choices: list, emotion_to_select: str)
    """
    # Extract emotion name if it has description
    emotion_key = emotion_name.split(" (")[0] if " (" in emotion_name else emotion_name

    if not emotion_key or emotion_key.strip() == "":
        return False, "❌ Please select or enter an emotion name", emotions_dict, [], ""

    # Save to emotions.json
    success, message, updated_emotions = save_emotion(
        emotions_dict, emotion_key, temp, penalty, top_p, intensity
    )

    if success:
        # Return updated choices
        new_choices = get_emotion_choices(updated_emotions)
        return True, message, updated_emotions, new_choices, emotion_key
    else:
        return False, message, emotions_dict, [], ""


def handle_delete_emotion(emotions_dict, confirm_value, emotion_name):
    """Handle deleting an emotion preset (called after confirmation) - returns data for UI update.

    Args:
        emotions_dict: Current emotions dictionary
        confirm_value: Confirmation trigger value
        emotion_name: Name to delete

    Returns:
        Tuple: (success: bool, message: str, updated_emotions: dict, new_choices: list, clear_trigger: str)
    """
    # If confirmation was cancelled (empty string), do nothing
    if not confirm_value:
        return False, "", emotions_dict, [], ""

    # Extract emotion name if it has description
    emotion_key = emotion_name.split(" (")[0] if " (" in emotion_name else emotion_name

    if not emotion_key or emotion_key.strip() == "":
        return False, "❌ Please select an emotion to delete", emotions_dict, [], ""

    # Delete from emotions.json
    success, message, updated_emotions = delete_emotion(
        emotions_dict, emotion_key
    )

    if success:
        # Return updated choices and select (None) for no emotion
        new_choices = get_emotion_choices(updated_emotions)
        # Select first item (None) = no emotion
        emotion_to_select = "(None)"
        return True, message, updated_emotions, new_choices, emotion_to_select
    else:
        return False, message, emotions_dict, [], ""


# ============================================================================
# UI Helper Functions (Gradio-specific)
# ============================================================================

def process_save_emotion_result(save_result, shared_state):
    """Process emotion save result and return UI updates.

    This helper encapsulates the common pattern of:
    1. Unpacking the 5-tuple from handle_save_emotion
    2. Updating shared_state with new emotions
    3. Creating Gradio dropdown update

    Args:
        save_result: Tuple from handle_save_emotion (success, message, updated_emotions, new_choices, emotion_to_select)
        shared_state: Dictionary containing '_active_emotions' key

    Returns:
        Tuple: (dropdown_update, message) for Gradio outputs
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for UI helpers")

    success, message, updated_emotions, new_choices, emotion_to_select = save_result

    # Update shared state if successful
    if success:
        shared_state['_active_emotions'] = updated_emotions

    # Return dropdown update and message
    return (
        gr.update(choices=new_choices, value=emotion_to_select) if new_choices else gr.update(),
        message
    )


def process_delete_emotion_result(delete_result, shared_state):
    """Process emotion delete result and return UI updates.

    This helper encapsulates the common pattern of:
    1. Unpacking the 5-tuple from handle_delete_emotion
    2. Updating shared_state with new emotions
    3. Creating Gradio dropdown update

    Args:
        delete_result: Tuple from handle_delete_emotion (success, message, updated_emotions, new_choices, emotion_to_select)
        shared_state: Dictionary containing '_active_emotions' key

    Returns:
        Tuple: (dropdown_update, message, clear_trigger) for Gradio outputs
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for UI helpers")

    success, message, updated_emotions, new_choices, emotion_to_select = delete_result

    # Update shared state if successful
    if success:
        shared_state['_active_emotions'] = updated_emotions
        # Select the emotion returned (empty string for "no emotion")
        return gr.update(choices=new_choices, value=emotion_to_select), message, ""
    elif message:  # Error message
        return gr.update(), message, ""
    else:  # Cancelled
        return gr.update(), "", ""
