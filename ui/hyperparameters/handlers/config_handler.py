"""
File: smartcash/ui/hyperparameters/handlers/config_handler.py
Deskripsi: Config handler untuk hyperparameters dengan field mapping essentials
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.ui.hyperparameters.handlers.defaults import get_default_hyperparameters_config


class HyperparametersConfigHandler(BaseConfigHandler):
    """Config handler untuk hyperparameters dengan field mapping essentials backend"""
    
    def __init__(self, config_filename: str = 'hyperparameters_config', module_name: str = 'hyperparameters'):
        super().__init__(config_filename, module_name)
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config dengan field mapping yang simplified 🔄"""
        
        # Extract config sections
        training = config.get('training', {})
        optimizer = config.get('optimizer', {})
        scheduler = config.get('scheduler', {})
        loss = config.get('loss', {})
        early_stopping = config.get('early_stopping', {})
        checkpoint = config.get('checkpoint', {})
        
        # Helper untuk set widget value
        set_val = lambda key, val: setattr(ui_components.get(key, type('obj', (object,), {})()), 'value', val)
        
        # Field mappings essentials - hanya yang ada di UI
        field_mappings = [
            # Training essentials
            ('epochs_slider', training, 'epochs', 100),
            ('batch_size_slider', training, 'batch_size', 16),
            ('learning_rate_slider', training, 'learning_rate', 0.01),
            ('image_size_slider', training, 'image_size', 640),
            
            # Optimizer essentials
            ('optimizer_dropdown', optimizer, 'type', 'SGD'),
            ('weight_decay_slider', optimizer, 'weight_decay', 0.0005),
            
            # Scheduler essentials
            ('scheduler_dropdown', scheduler, 'type', 'cosine'),
            ('warmup_epochs_slider', scheduler, 'warmup_epochs', 3),
            
            # Loss essentials
            ('box_loss_gain_slider', loss, 'box_loss_gain', 0.05),
            ('cls_loss_gain_slider', loss, 'cls_loss_gain', 0.5),
            ('obj_loss_gain_slider', loss, 'obj_loss_gain', 1.0),
            
            # Control essentials
            ('early_stopping_checkbox', early_stopping, 'enabled', True),
            ('patience_slider', early_stopping, 'patience', 15),
            ('save_best_checkbox', checkpoint, 'save_best', True),
            ('checkpoint_metric_dropdown', checkpoint, 'metric', 'mAP_0.5')
        ]
        
        # Apply updates dengan one-liner
        [set_val(component_key, source_config.get(config_key, default_value)) 
         for component_key, source_config, config_key, default_value in field_mappings]
        
        # Update summary cards jika ada
        self._update_summary_cards(ui_components, config)
    
    def update_config_from_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Update config dari UI components dengan field mapping essentials 🔄"""
        
        # Helper untuk get widget value
        get_val = lambda key: getattr(ui_components.get(key, type('obj', (object,), {'value': None})()), 'value', None)
        
        # Update config sections
        config.setdefault('training', {}).update({
            'epochs': get_val('epochs_slider'),
            'batch_size': get_val('batch_size_slider'),
            'learning_rate': get_val('learning_rate_slider'),
            'image_size': get_val('image_size_slider')
        })
        
        config.setdefault('optimizer', {}).update({
            'type': get_val('optimizer_dropdown'),
            'weight_decay': get_val('weight_decay_slider')
        })
        
        config.setdefault('scheduler', {}).update({
            'type': get_val('scheduler_dropdown'),
            'warmup_epochs': get_val('warmup_epochs_slider')
        })
        
        config.setdefault('loss', {}).update({
            'box_loss_gain': get_val('box_loss_gain_slider'),
            'cls_loss_gain': get_val('cls_loss_gain_slider'),
            'obj_loss_gain': get_val('obj_loss_gain_slider')
        })
        
        config.setdefault('early_stopping', {}).update({
            'enabled': get_val('early_stopping_checkbox'),
            'patience': get_val('patience_slider')
        })
        
        config.setdefault('checkpoint', {}).update({
            'save_best': get_val('save_best_checkbox'),
            'metric': get_val('checkpoint_metric_dropdown')
        })
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default hyperparameters configuration"""
        return get_default_hyperparameters_config()
    
    def _update_summary_cards(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update summary cards dengan config terbaru"""
        if 'summary_cards' not in ui_components:
            return
            
        training = config.get('training', {})
        optimizer = config.get('optimizer', {})
        loss = config.get('loss', {})
        early_stopping = config.get('early_stopping', {})
        checkpoint = config.get('checkpoint', {})
        
        summary_data = {
            'Training': f"Epochs: {training.get('epochs', 100)}, Batch: {training.get('batch_size', 16)}, LR: {training.get('learning_rate', 0.01):.4f}",
            'Optimizer': f"{optimizer.get('type', 'SGD')} (decay: {optimizer.get('weight_decay', 0.0005):.4f})",
            'Loss': f"Box: {loss.get('box_loss_gain', 0.05):.2f}, Cls: {loss.get('cls_loss_gain', 0.5):.1f}, Obj: {loss.get('obj_loss_gain', 1.0):.1f}",
            'Control': f"Early Stop: {'On' if early_stopping.get('enabled', True) else 'Off'}, Save Best: {'On' if checkpoint.get('save_best', True) else 'Off'}"
        }
        
        # Update summary cards content
        ui_components['summary_cards'].children = tuple([
            self._create_summary_card(title, content) 
            for title, content in summary_data.items()
        ])
    
    def _create_summary_card(self, title: str, content: str) -> Any:
        """Create summary card widget"""
        from ipywidgets import HTML
        return HTML(f"""
            <div style='background: #f8f9fa; border: 1px solid #dee2e6; 
                        border-radius: 6px; padding: 8px; margin: 2px;'>
                <strong style='color: #495057;'>{title}:</strong><br>
                <span style='color: #6c757d; font-size: 0.9em;'>{content}</span>
            </div>
        """)


# Removed field mappings yang tidak digunakan:
# - mixed_precision_checkbox
# - gradient_accumulation_slider  
# - gradient_clipping_slider
# - momentum_slider
# - min_delta_slider