

# File: smartcash/ui/hyperparameters/components/ui_form.py
# Deskripsi: UI form components untuk hyperparameters configuration

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple, Optional, Callable
from smartcash.ui.hyperparameters.handlers.defaults import get_hyperparameters_ui_config
from smartcash.common.logger import get_logger
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel

logger = get_logger(__name__)


def create_hyperparameters_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form components untuk hyperparameters configuration"""
    try:
        ui_config = get_hyperparameters_ui_config()
        components = {}
        
        # Create form fields
        for section in ui_config['form_sections']:
            for field_name in section['fields']:
                if field_name in ui_config['field_configs']:
                    field_config = ui_config['field_configs'][field_name]
                    components[field_name] = _create_field_widget(field_name, field_config, config)
        
        # Add save and reset buttons
        save_reset_buttons = create_save_reset_buttons(
            on_save_click=lambda b: _on_save_click(b, components, status_panel),
            on_reset_click=lambda b: _on_reset_click(b, components, config, status_panel),
            with_sync_info=False
        )
        
        # Add status panel
        status_panel = create_status_panel("Siap mengkonfigurasi hyperparameters", "info")
        
        # Merge all components
        components.update({
            'save_button': save_reset_buttons['save_button'],
            'reset_button': save_reset_buttons['reset_button'],
            'save_reset_container': save_reset_buttons['container'],
            'status_panel': status_panel
        })
        
        return components
        
    except Exception as e:
        logger.error(f"❌ Error creating hyperparameters form: {e}")
        raise


def _create_field_widget(field_name: str, field_config: Dict[str, Any], config: Dict[str, Any]) -> widgets.Widget:
    """Create individual form field widget"""
    value = _get_field_value(field_name, config)
    
    if field_config['type'] == 'IntSlider':
        return widgets.IntSlider(
            value=value,
            min=field_config['min'],
            max=field_config['max'],
            step=field_config['step'],
            description=_format_field_name(field_name),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
    elif field_config['type'] == 'FloatSlider':
        return widgets.FloatSlider(
            value=value,
            min=field_config['min'],
            max=field_config['max'],
            step=field_config['step'],
            description=_format_field_name(field_name),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
    elif field_config['type'] == 'Dropdown':
        return widgets.Dropdown(
            value=value,
            options=field_config['options'],
            description=_format_field_name(field_name),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
    elif field_config['type'] == 'Checkbox':
        return widgets.Checkbox(
            value=value,
            description=_format_field_name(field_name),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
    else:
        return widgets.HTML(f"Unsupported widget type: {field_config['type']}")


def _get_field_value(field_name: str, config: Dict[str, Any]) -> Any:
    """Get field value from config with proper path resolution"""
    field_map = {
        'epochs': ['training', 'epochs'],
        'batch_size': ['training', 'batch_size'],
        'learning_rate': ['training', 'learning_rate'],
        'image_size': ['training', 'image_size'],
        'optimizer_type': ['optimizer', 'type'],
        'weight_decay': ['optimizer', 'weight_decay'],
        'momentum': ['optimizer', 'momentum'],
        'scheduler_type': ['scheduler', 'type'],
        'warmup_epochs': ['scheduler', 'warmup_epochs'],
        'min_lr': ['scheduler', 'min_lr'],
        'box_loss_gain': ['loss', 'box_loss_gain'],
        'cls_loss_gain': ['loss', 'cls_loss_gain'],
        'obj_loss_gain': ['loss', 'obj_loss_gain'],
        'early_stopping_enabled': ['early_stopping', 'enabled'],
        'patience': ['early_stopping', 'patience'],
        'min_delta': ['early_stopping', 'min_delta']
    }
    
    path = field_map.get(field_name, [])
    value = config
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _format_field_name(field_name: str) -> str:
    """Format field name for display"""
    field_labels = {
        'epochs': 'Epochs',
        'batch_size': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'image_size': 'Image Size',
        'optimizer_type': 'Optimizer',
        'weight_decay': 'Weight Decay',
        'momentum': 'Momentum',
        'scheduler_type': 'Scheduler',
        'warmup_epochs': 'Warmup Epochs',
        'min_lr': 'Min LR',
        'box_loss_gain': 'Box Loss',
        'cls_loss_gain': 'Class Loss',
        'obj_loss_gain': 'Object Loss',
        'early_stopping_enabled': 'Enable Early Stop',
        'patience': 'Patience',
        'min_delta': 'Min Delta'
    }
    return field_labels.get(field_name, field_name.replace('_', ' ').title())


def _on_save_click(button: widgets.Button, components: Dict[str, Any], status_panel: widgets.HTML) -> None:
    """Handle save button click"""
    try:
        # Update status panel
        status_panel.value = "<div class='alert alert-success'>Menyimpan konfigurasi...</div>"
        # Actual save logic will be handled by the parent component
    except Exception as e:
        status_panel.value = f"<div class='alert alert-danger'>Gagal menyimpan: {str(e)}</div>"


def _on_reset_click(button: widgets.Button, components: Dict[str, Any], 
                    default_config: Dict[str, Any], status_panel: widgets.HTML) -> None:
    """Handle reset button click"""
    try:
        # Reset all fields to default values
        for field_name, widget in components.items():
            if hasattr(widget, 'value') and not isinstance(widget, (widgets.Button, widgets.HTML)):
                widget.value = _get_field_value(field_name, default_config)
        status_panel.value = "<div class='alert alert-info'>Konfigurasi direset ke nilai default</div>"
    except Exception as e:
        status_panel.value = f"<div class='alert alert-danger'>Gagal mereset: {str(e)}</div>"
    
class HyperparametersForm:
    """Form UI untuk konfigurasi hyperparameters (legacy compatibility)"""
    
    def __init__(self, config: Dict[str, Any], on_change: Optional[Callable] = None):
        self.config = config
        self.on_change = on_change or (lambda: None)
        self.components = create_hyperparameters_form(config)
        self._setup_widget_handlers()
    
    def _setup_widget_handlers(self) -> None:
        """Setup change handlers for all widgets"""
        for widget in self.components.values():
            if hasattr(widget, 'observe'):
                widget.observe(lambda _: self.on_change(), names='value')
    
    def get_widget(self) -> widgets.Widget:
        """Get main form widget"""
        return self.components.get('form', widgets.HTML("⚠️ Form not created"))
    
    def get_config(self) -> Dict[str, Any]:
        """Get current config from form"""
        return self.config  # Config is updated in real-time by the parent component

