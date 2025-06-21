

# File: smartcash/ui/hyperparameters/components/ui_form.py
# Deskripsi: UI form components untuk hyperparameters configuration

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from smartcash.ui.hyperparameters.handlers.defaults import get_hyperparameters_ui_config
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class HyperparametersForm:
    """Form UI untuk konfigurasi hyperparameters 📝"""
    
    def __init__(self, config: Dict[str, Any], on_change: Optional[Callable] = None):
        self.config = config
        self.on_change = on_change or (lambda: None)
        self.ui_config = get_hyperparameters_ui_config()
        self.widgets = {}
        self._create_form()
    
    def _create_form(self) -> None:
        """Buat form widgets berdasarkan konfigurasi 🏗️"""
        try:
            self.widgets['sections'] = []
            
            for section_config in self.ui_config['form_sections']:
                section_widget = self._create_section(section_config)
                self.widgets['sections'].append(section_widget)
                
            # Create main form container
            self.widgets['form'] = widgets.VBox(
                children=self.widgets['sections'],
                layout=widgets.Layout(
                    width='100%',
                    gap='15px',
                    padding='10px'
                )
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating hyperparameters form: {e}")
            self.widgets['form'] = widgets.HTML("⚠️ Error creating form")
    
    def _create_section(self, section_config: Dict[str, Any]) -> widgets.Widget:
        """Buat section widget dengan accordion layout 📋"""
        section_widgets = []
        
        for field_name in section_config['fields']:
            field_widget = self._create_field_widget(field_name)
            if field_widget:
                section_widgets.append(field_widget)
        
        section_box = widgets.VBox(
            children=section_widgets,
            layout=widgets.Layout(gap='8px', padding='10px')
        )
        
        accordion = widgets.Accordion(
            children=[section_box],
            titles=[section_config['title']],
            selected_index=0
        )
        
        return accordion
    
    def _create_field_widget(self, field_name: str) -> Optional[widgets.Widget]:
        """Buat widget untuk field tertentu ⚙️"""
        try:
            if field_name not in self.ui_config['field_configs']:
                return None
                
            field_config = self.ui_config['field_configs'][field_name]
            current_value = self._get_field_value(field_name)
            
            # Create widget based on type
            if field_config['type'] == 'IntSlider':
                widget = widgets.IntSlider(
                    value=current_value,
                    min=field_config['min'],
                    max=field_config['max'],
                    step=field_config['step'],
                    description=self._format_field_name(field_name),
                    style={'description_width': '150px'}
                )
            elif field_config['type'] == 'FloatSlider':
                widget = widgets.FloatSlider(
                    value=current_value,
                    min=field_config['min'],
                    max=field_config['max'],
                    step=field_config['step'],
                    description=self._format_field_name(field_name),
                    style={'description_width': '150px'}
                )
            elif field_config['type'] == 'FloatLogSlider':
                widget = widgets.FloatLogSlider(
                    value=current_value,
                    min=field_config['min'],
                    max=field_config['max'],
                    step=field_config['step'],
                    description=self._format_field_name(field_name),
                    style={'description_width': '150px'}
                )
            elif field_config['type'] == 'Dropdown':
                widget = widgets.Dropdown(
                    value=current_value,
                    options=field_config['options'],
                    description=self._format_field_name(field_name),
                    style={'description_width': '150px'}
                )
            elif field_config['type'] == 'Checkbox':
                widget = widgets.Checkbox(
                    value=current_value,
                    description=self._format_field_name(field_name),
                    style={'description_width': '150px'}
                )
            else:
                return None
            
            # Bind change handler
            widget.observe(self._on_widget_change, names='value')
            self.widgets[field_name] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"❌ Error creating field widget {field_name}: {e}")
            return None
    
    def _get_field_value(self, field_name: str) -> Any:
        """Ambil nilai field dari config 📊"""
        # Map field names to config paths
        field_map = {
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size', 
            'learning_rate': 'training.learning_rate',
            'image_size': 'training.image_size',
            'optimizer_type': 'optimizer.type',
            'weight_decay': 'optimizer.weight_decay',
            'momentum': 'optimizer.momentum',
            'scheduler_type': 'scheduler.type',
            'warmup_epochs': 'scheduler.warmup_epochs',
            'min_lr': 'scheduler.min_lr',
            'box_loss_gain': 'loss.box_loss_gain',
            'cls_loss_gain': 'loss.cls_loss_gain',
            'obj_loss_gain': 'loss.obj_loss_gain',
            'early_stopping_enabled': 'early_stopping.enabled',
            'patience': 'early_stopping.patience',
            'min_delta': 'early_stopping.min_delta'
        }
        
        if field_name not in field_map:
            return None
            
        path = field_map[field_name].split('.')
        value = self.config
        
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _format_field_name(self, field_name: str) -> str:
        """Format field name untuk display 💫"""
        # Format field names untuk UI yang lebih readable
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
    
    def _on_widget_change(self, change) -> None:
        """Handle perubahan widget value 🔄"""
        try:
            # Update config berdasarkan widget change
            self._update_config_from_widgets()
            
            # Trigger change callback
            self.on_change()
            
        except Exception as e:
            logger.error(f"❌ Error handling widget change: {e}")
    
    def _update_config_from_widgets(self) -> None:
        """Update config dari nilai widgets ⚡"""
        # Reverse mapping dari _get_field_value
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
        
        for field_name, widget in self.widgets.items():
            if (field_name in field_map and hasattr(widget, 'value')):
                path = field_map[field_name]
                self._set_nested_value(self.config, path, widget.value)
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any) -> None:
        """Set nilai nested dalam config dict 🎯"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get_widget(self) -> widgets.Widget:
        """Ambil main form widget 📦"""
        return self.widgets.get('form', widgets.HTML("⚠️ Form not created"))
    
    def get_config(self) -> Dict[str, Any]:
        """Ambil current config dari form 📋"""
        self._update_config_from_widgets()
        return self.config

