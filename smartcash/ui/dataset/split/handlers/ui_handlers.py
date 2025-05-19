"""
File: smartcash/ui/dataset/split/handlers/ui_handlers.py
Deskripsi: Handler untuk UI components split dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.config_handlers import get_split_config, update_config_from_ui, update_ui_from_config
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def handle_split_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle UI components untuk split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Get config
        config = get_split_config(ui_components)
        
        # Update UI from config
        update_ui_from_config(ui_components, config)
        
        logger.info("✅ Split UI berhasil diupdate")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat update split UI: {str(e)}")
        return ui_components

def on_slider_change(train_slider, val_slider, test_slider, total_label) -> None:
    """
    Handler untuk perubahan nilai slider.
    
    Args:
        train_slider: Slider untuk proporsi data training
        val_slider: Slider untuk proporsi data validasi
        test_slider: Slider untuk proporsi data testing
        total_label: Label untuk menampilkan total proporsi
    """
    # Hitung total proporsi
    total = round(train_slider.value + val_slider.value + test_slider.value, 2)
    
    # Tentukan warna berdasarkan total
    color = 'green' if total == 1.0 else 'red'
    
    # Update label total
    total_label.value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"

def validate_sliders(train_slider, val_slider, test_slider) -> bool:
    """
    Validasi nilai slider untuk memastikan total = 1.
    
    Args:
        train_slider: Slider untuk proporsi data training
        val_slider: Slider untuk proporsi data validasi
        test_slider: Slider untuk proporsi data testing
        
    Returns:
        Boolean yang menunjukkan validitas nilai slider
    """
    # Hitung total proporsi
    total = round(train_slider.value + val_slider.value + test_slider.value, 2)
    
    # Kembalikan hasil validasi
    return total == 1.0
