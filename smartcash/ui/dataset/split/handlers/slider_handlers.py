"""
File: smartcash/ui/dataset/split/handlers/slider_handlers.py
Deskripsi: Handler untuk slider events dan auto-adjustment ratio
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import COLORS
from smartcash.ui.dataset.split.handlers.defaults import normalize_split_ratios

logger = get_logger(__name__)


def setup_slider_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup event handlers untuk slider ratio."""
    try:
        sliders = ['train_slider', 'valid_slider', 'test_slider']
        available_sliders = [s for s in sliders if s in ui_components]
        
        if len(available_sliders) == 3:
            for slider_key in available_sliders:
                ui_components[slider_key].observe(
                    _create_slider_handler(ui_components), 
                    names='value'
                )
            logger.debug("🎚️ Slider handlers berhasil dipasang")
        else:
            logger.warning("⚠️ Tidak semua slider tersedia untuk handler")
            
    except Exception as e:
        logger.error(f"💥 Error setup slider handlers: {str(e)}")


def _create_slider_handler(ui_components: Dict[str, Any]):
    """Buat handler untuk slider dengan auto-adjustment."""
    def on_slider_change(change):
        try:
            _adjust_ratios_and_update_total(ui_components, change['owner'])
        except Exception as e:
            logger.error(f"💥 Error slider adjustment: {str(e)}")
    return on_slider_change


def _adjust_ratios_and_update_total(ui_components: Dict[str, Any], changed_slider) -> None:
    """Adjust ratio dan update total label."""
    train_slider = ui_components['train_slider']
    valid_slider = ui_components['valid_slider'] 
    test_slider = ui_components['test_slider']
    total_label = ui_components['total_label']
    
    # Auto-adjust other sliders berdasarkan yang berubah
    if changed_slider == train_slider:
        remaining = 1.0 - train_slider.value
        valid_slider.value = round(remaining / 2, 2)
        test_slider.value = round(remaining / 2, 2)
    elif changed_slider == valid_slider:
        test_slider.value = round(1.0 - train_slider.value - valid_slider.value, 2)
    elif changed_slider == test_slider:
        valid_slider.value = round(1.0 - train_slider.value - test_slider.value, 2)
    
    # Update total label
    total = round(train_slider.value + valid_slider.value + test_slider.value, 2)
    color = COLORS['success'] if total == 1.0 else COLORS['danger']
    total_label.value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"