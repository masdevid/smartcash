"""
File: smartcash/ui/dataset/preprocessing/handlers/reset_handler.py
Deskripsi: Handler untuk tombol reset konfigurasi preprocessing dataset
"""

from typing import Dict, Any, Optional

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel

def handle_reset_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol reset konfigurasi preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Log reset konfigurasi
        log_message(ui_components, "Reset konfigurasi preprocessing dataset...", "info", "🔄")
        
        # Update UI state
        update_status_panel(ui_components, "info", "Reset konfigurasi preprocessing...")
        
        # Reset konfigurasi
        reset_preprocessing_config(ui_components)
        
        # Log hasil
        log_message(ui_components, "Konfigurasi preprocessing berhasil direset ke default", "success", "✅")
        
        # Update UI state
        update_status_panel(ui_components, "success", "Konfigurasi direset ke default")
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_status_panel(ui_components, "error", f"Error saat reset konfigurasi: {error_message}")
        log_message(ui_components, f"Error saat reset konfigurasi preprocessing: {error_message}", "error", "❌")
    
    finally:
        # Re-enable tombol setelah operasi selesai
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def reset_preprocessing_config(ui_components: Dict[str, Any]) -> None:
    """
    Reset konfigurasi preprocessing ke default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Default preprocessing options
    defaults = {
        'resolution': '640x640',
        'normalization': 'minmax',
        'augmentation': False,
        'split': 'Train Only',
        'validation_items': ['validate_image_format', 'validate_label_format', 
                             'validate_image_dimensions', 'validate_bounding_box']
    }
    
    # Get preprocessing options widget
    if 'preprocess_options' in ui_components:
        options = ui_components['preprocess_options']
        
        # Reset resolution
        if hasattr(options, 'resolution') and hasattr(options.resolution, 'value'):
            options.resolution.value = defaults['resolution']
        
        # Reset normalization
        if hasattr(options, 'normalization') and hasattr(options.normalization, 'value'):
            options.normalization.value = defaults['normalization']
        
        # Reset augmentation
        if hasattr(options, 'augmentation') and hasattr(options.augmentation, 'value'):
            options.augmentation.value = defaults['augmentation']
    
    # Reset split selector
    if 'split_selector' in ui_components and hasattr(ui_components['split_selector'], 'value'):
        ui_components['split_selector'].value = defaults['split']
    
    # Reset validation options
    if 'validation_options' in ui_components:
        validation_options = ui_components['validation_options']
        if hasattr(validation_options, 'set_selected') and callable(validation_options.set_selected):
            try:
                validation_options.set_selected(defaults['validation_items'])
            except Exception:
                # Ignore errors
                pass 