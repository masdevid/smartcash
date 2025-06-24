"""
File: smartcash/ui/dataset/split/handlers/slider_handlers.py
Deskripsi: Handler untuk slider events dengan one-liner approach
"""

from typing import Dict, Any


def setup_slider_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup slider handlers dengan consolidated approach"""
    sliders = ['train_slider', 'valid_slider', 'test_slider']
    available_sliders = [s for s in sliders if s in ui_components and hasattr(ui_components[s], 'observe')]
    
    # One-liner handler setup
    [ui_components[slider].observe(_create_slider_handler(ui_components), names='value') for slider in available_sliders if len(available_sliders) == 3]


def _create_slider_handler(ui_components: Dict[str, Any]):
    """Create slider handler dengan auto-adjustment logic"""
    def on_slider_change(change):
        _adjust_ratios_and_update_total(ui_components, change['owner'])
    return on_slider_change


def _adjust_ratios_and_update_total(ui_components: Dict[str, Any], changed_slider) -> None:
    """Adjust ratios dan update total dengan one-liner calculations"""
    sliders = {name: ui_components[f'{name}_slider'] for name in ['train', 'valid', 'test']}
    
    # Auto-adjustment logic dengan conditional updates
    if changed_slider == sliders['train']:
        remaining = 1.0 - sliders['train'].value
        sliders['valid'].value = sliders['test'].value = round(remaining / 2, 2)
    elif changed_slider == sliders['valid']:
        sliders['test'].value = round(1.0 - sliders['train'].value - sliders['valid'].value, 2)
    elif changed_slider == sliders['test']:
        sliders['valid'].value = round(1.0 - sliders['train'].value - sliders['test'].value, 2)
    
    # Update total label dengan one-liner styling
    _update_total_display(ui_components, sliders)


def _update_total_display(ui_components: Dict[str, Any], sliders: Dict[str, Any]) -> None:
    """Update total display dengan consolidated styling"""
    if 'total_label' not in ui_components:
        return
    
    try:
        from smartcash.ui.utils.constants import COLORS
        total = round(sum(slider.value for slider in sliders.values()), 2)
        color = COLORS.get('success', '#28a745') if total == 1.0 else COLORS.get('danger', '#dc3545')
        ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
    except Exception:
        pass  # Silent fail untuk prevent handler errors


# One-liner utilities
get_total_ratio = lambda ui_components: round(sum(getattr(ui_components.get(f'{name}_slider', type('', (), {'value': 0})()), 'value', 0) for name in ['train', 'valid', 'test']), 2)
is_valid_total = lambda ui_components: get_total_ratio(ui_components) == 1.0