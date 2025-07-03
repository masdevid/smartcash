"""
File: smartcash/ui/dataset/split/handlers/slider_handlers.py
Deskripsi: Handler untuk slider events dengan centralized error handling
"""

from typing import Dict, Any, List, Callable
import logging

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors

# Import constants
from smartcash.ui.utils.constants import COLORS

# Logger
logger = logging.getLogger(__name__)

@handle_ui_errors(log_error=True)
def setup_slider_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup slider handlers dengan centralized error handling
    
    Args:
        ui_components: Dictionary of UI components
    """
    logger.debug("Menyiapkan slider handlers untuk split ratio")
    
    # Define slider names
    sliders = ['train_slider', 'valid_slider', 'test_slider']
    
    # Check which sliders are available in UI components
    available_sliders = [s for s in sliders if s in ui_components and hasattr(ui_components[s], 'observe')]
    
    # Only setup handlers if all sliders are available
    if len(available_sliders) == 3:
        for slider in available_sliders:
            ui_components[slider].observe(_create_slider_handler(ui_components), names='value')
        logger.debug("Slider handlers berhasil disiapkan")
    else:
        logger.warning(f"Tidak semua slider tersedia. Ditemukan: {available_sliders}")


def _create_slider_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create slider handler dengan auto-adjustment logic
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Slider change handler function
    """
    @handle_ui_errors(log_error=True)
    def on_slider_change(change):
        """Handle slider value changes
        
        Args:
            change: Change event from slider
        """
        _adjust_ratios_and_update_total(ui_components, change['owner'])
        
    return on_slider_change


@handle_ui_errors(log_error=True)
def _adjust_ratios_and_update_total(ui_components: Dict[str, Any], changed_slider) -> None:
    """Adjust ratios dan update total dengan centralized error handling
    
    Args:
        ui_components: Dictionary of UI components
        changed_slider: Slider that was changed
    """
    # Get all sliders
    sliders = {name: ui_components.get(f'{name}_slider') for name in ['train', 'valid', 'test']}
    
    # Ensure all sliders exist
    if not all(sliders.values()):
        logger.warning("Tidak semua slider ditemukan dalam UI components")
        return
    
    # Auto-adjustment logic dengan conditional updates
    if changed_slider == sliders['train']:
        remaining = 1.0 - sliders['train'].value
        sliders['valid'].value = sliders['test'].value = round(remaining / 2, 2)
        logger.debug(f"Train slider diubah ke {sliders['train'].value}, menyesuaikan valid dan test ke {sliders['valid'].value}")
    elif changed_slider == sliders['valid']:
        sliders['test'].value = round(1.0 - sliders['train'].value - sliders['valid'].value, 2)
        logger.debug(f"Valid slider diubah ke {sliders['valid'].value}, menyesuaikan test ke {sliders['test'].value}")
    elif changed_slider == sliders['test']:
        sliders['valid'].value = round(1.0 - sliders['train'].value - sliders['test'].value, 2)
        logger.debug(f"Test slider diubah ke {sliders['test'].value}, menyesuaikan valid ke {sliders['valid'].value}")
    
    # Update total label
    _update_total_display(ui_components, sliders)


@handle_ui_errors(log_error=True)
def _update_total_display(ui_components: Dict[str, Any], sliders: Dict[str, Any]) -> None:
    """Update total display dengan centralized error handling
    
    Args:
        ui_components: Dictionary of UI components
        sliders: Dictionary of slider components
    """
    if 'total_label' not in ui_components:
        logger.warning("Total label tidak ditemukan dalam UI components")
        return
    
    # Calculate total
    total = round(sum(slider.value for slider in sliders.values()), 2)
    
    # Determine color based on total
    is_valid = total == 1.0
    color = COLORS.get('success', '#28a745') if is_valid else COLORS.get('danger', '#dc3545')
    
    # Update label
    ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
    
    # Update save button state if available
    if 'save_button' in ui_components:
        ui_components['save_button'].disabled = not is_valid
    
    # Log result
    if is_valid:
        logger.debug(f"Total ratio valid: {total}")
    else:
        logger.warning(f"Total ratio tidak valid: {total}, seharusnya 1.0")


@handle_ui_errors(log_error=True)
def get_total_ratio(ui_components: Dict[str, Any]) -> float:
    """Calculate total ratio dari semua sliders
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Total ratio as float
    """
    return round(sum(
        getattr(ui_components.get(f'{name}_slider', type('', (), {'value': 0})()), 'value', 0) 
        for name in ['train', 'valid', 'test']
    ), 2)


@handle_ui_errors(log_error=True)
def is_valid_total(ui_components: Dict[str, Any]) -> bool:
    """Check if total ratio is valid (equals 1.0)
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Boolean indicating if total is valid
    """
    return get_total_ratio(ui_components) == 1.0
