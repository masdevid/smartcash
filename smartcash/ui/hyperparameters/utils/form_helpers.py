# File: smartcash/ui/hyperparameters/utils/form_helpers.py
# Deskripsi: Utility functions untuk form hyperparameters - menggunakan fallback_utils

from typing import Any, List, Tuple, Dict
import ipywidgets as widgets
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe

logger = get_logger(__name__)


def create_slider_widget(value: float, min_val: float, max_val: float, step: float, 
                        description: str, format_spec: str = '.2f') -> widgets.FloatSlider:
    """Buat float slider widget dengan validasi 📊"""
    return try_operation_safe(
        operation=lambda: widgets.FloatSlider(
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            description=description,
            readout_format=format_spec,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        ),
        fallback_value=widgets.FloatSlider(value=0.01, min=0.001, max=1.0, step=0.001, description=description),
        logger=logger,
        operation_name=f"creating slider widget {description}"
    )


def create_int_slider_widget(value: int, min_val: int, max_val: int, 
                            description: str, step: int = 1) -> widgets.IntSlider:
    """Buat integer slider widget dengan validasi 🔢"""
    return try_operation_safe(
        operation=lambda: widgets.IntSlider(
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            description=description,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        ),
        fallback_value=widgets.IntSlider(value=16, min=1, max=64, step=1, description=description),
        logger=logger,
        operation_name=f"creating int slider widget {description}"
    )


def create_dropdown_widget(value: str, options: List[str], description: str) -> widgets.Dropdown:
    """Buat dropdown widget dengan validasi 📋"""
    return try_operation_safe(
        operation=lambda: widgets.Dropdown(
            value=value if value in options else options[0],
            options=options,
            description=description,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        ),
        fallback_value=widgets.Dropdown(options=options or ['default'], description=description),
        logger=logger,
        operation_name=f"creating dropdown widget {description}"
    )


def create_checkbox_widget(value: bool, description: str) -> widgets.Checkbox:
    """Buat checkbox widget dengan validasi ☑️"""
    return try_operation_safe(
        operation=lambda: widgets.Checkbox(
            value=value,
            description=description,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        ),
        fallback_value=widgets.Checkbox(value=False, description=description),
        logger=logger,
        operation_name=f"creating checkbox widget {description}"
    )


def create_summary_cards_widget(config: Dict[str, Any]) -> widgets.HTML:
    """Buat summary cards untuk konfigurasi hyperparameters 📄"""
    return try_operation_safe(
        operation=lambda: _create_summary_cards_content(config),
        fallback_value=widgets.HTML(value="<div>📄 Summary loading...</div>"),
        logger=logger,
        operation_name="creating summary cards"
    )


def _create_summary_cards_content(config: Dict[str, Any]) -> widgets.HTML:
    """Helper untuk membuat konten summary cards"""
    training = config.get('training', {})
    optimizer = config.get('optimizer', {})
    scheduler = config.get('scheduler', {})
    
    html_content = f"""
    <div style="display: flex; gap: 15px; margin: 15px 0; flex-wrap: wrap;">
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">🎯 Training</h4>
            <div style="font-size: 13px; color: #5a6c7d;">
                <div>Epochs: <strong>{training.get('epochs', 100)}</strong></div>
                <div>Batch Size: <strong>{training.get('batch_size', 16)}</strong></div>
                <div>Learning Rate: <strong>{training.get('learning_rate', 0.01):.4f}</strong></div>
            </div>
        </div>
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">⚙️ Optimizer</h4>
            <div style="font-size: 13px; color: #5a6c7d;">
                <div>Type: <strong>{optimizer.get('name', 'AdamW').upper()}</strong></div>
                <div>Weight Decay: <strong>{optimizer.get('weight_decay', 0.0001):.4f}</strong></div>
                <div>Momentum: <strong>{optimizer.get('momentum', 0.937):.3f}</strong></div>
            </div>
        </div>
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">📈 Scheduler</h4>
            <div style="font-size: 13px; color: #5a6c7d;">
                <div>Type: <strong>{scheduler.get('name', 'CosineAnnealingLR').replace('_', ' ').title()}</strong></div>
                <div>Warm Epochs: <strong>{scheduler.get('warmup_epochs', 3)}</strong></div>
                <div>Min LR: <strong>{scheduler.get('min_lr', 1e-6):.1e}</strong></div>
            </div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=html_content)


def validate_form_values(form_widgets: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validasi nilai form hyperparameters ✅"""
    errors = []
    
    try:
        # Validasi training parameters
        if 'epochs_slider' in form_widgets:
            epochs = form_widgets['epochs_slider'].value
            if epochs < 1 or epochs > 1000:
                errors.append("Epochs harus antara 1-1000")
        
        if 'batch_size_slider' in form_widgets:
            batch_size = form_widgets['batch_size_slider'].value
            if batch_size < 1 or batch_size > 128:
                errors.append("Batch size harus antara 1-128")
        
        if 'learning_rate_slider' in form_widgets:
            lr = form_widgets['learning_rate_slider'].value
            if lr <= 0 or lr > 1:
                errors.append("Learning rate harus antara 0-1")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"❌ Error validating form: {e}")
        return False, [f"Validation error: {str(e)}"]