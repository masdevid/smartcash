"""
File: smartcash/ui/hyperparameters/handlers/config_handler.py
Deskripsi: Config handler untuk hyperparameters dengan extract/update yang clean dan summary cards
"""

from typing import Dict, Any
import datetime
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.hyperparameters.handlers.defaults import get_default_hyperparameters_config


class HyperparametersConfigHandler(ConfigHandler):
    """Config handler untuk hyperparameters dengan summary cards dan clean operations"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan one-liner style"""
        
        # One-liner safe getter
        get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components and hasattr(ui_components.get(key), 'value') else default
        
        return {
            '_base_': 'base_config.yaml',
            
            'training': {
                'epochs': get_val('epochs_slider', 100),
                'batch_size': get_val('batch_size_slider', 16),
                'learning_rate': get_val('learning_rate_slider', 0.01),
                'image_size': get_val('image_size_slider', 640),
                'mixed_precision': get_val('mixed_precision_checkbox', True),
                'gradient_accumulation': get_val('gradient_accumulation_slider', 1),
                'gradient_clipping': get_val('gradient_clipping_slider', 1.0)
            },
            
            'optimizer': {
                'type': get_val('optimizer_dropdown', 'SGD'),
                'weight_decay': get_val('weight_decay_slider', 0.0005),
                'momentum': get_val('momentum_slider', 0.937)
            },
            
            'scheduler': {
                'type': get_val('scheduler_dropdown', 'cosine'),
                'warmup_epochs': get_val('warmup_epochs_slider', 3)
            },
            
            'loss': {
                'box_loss_gain': get_val('box_loss_gain_slider', 0.05),
                'cls_loss_gain': get_val('cls_loss_gain_slider', 0.5),
                'obj_loss_gain': get_val('obj_loss_gain_slider', 1.0)
            },
            
            'early_stopping': {
                'enabled': get_val('early_stopping_checkbox', True),
                'patience': get_val('patience_slider', 15),
                'min_delta': get_val('min_delta_slider', 0.001)
            },
            
            'checkpoint': {
                'save_best': get_val('save_best_checkbox', True),
                'metric': get_val('checkpoint_metric_dropdown', 'mAP_0.5')
            },
            
            'config_version': '1.0',
            'updated_at': datetime.datetime.now().isoformat(),
            'module_name': 'hyperparameters'
        }
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config dengan one-liner assignments"""
        
        # One-liner safe setter
        set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
        
        # Extract nested configs dengan fallbacks
        training = config.get('training', {})
        optimizer = config.get('optimizer', {})
        scheduler = config.get('scheduler', {})
        loss = config.get('loss', {})
        early_stopping = config.get('early_stopping', {})
        checkpoint = config.get('checkpoint', {})
        
        # Update UI dengan batch assignments
        field_mappings = [
            ('epochs_slider', training, 'epochs', 100),
            ('batch_size_slider', training, 'batch_size', 16),
            ('learning_rate_slider', training, 'learning_rate', 0.01),
            ('image_size_slider', training, 'image_size', 640),
            ('mixed_precision_checkbox', training, 'mixed_precision', True),
            ('gradient_accumulation_slider', training, 'gradient_accumulation', 1),
            ('gradient_clipping_slider', training, 'gradient_clipping', 1.0),
            ('optimizer_dropdown', optimizer, 'type', 'SGD'),
            ('weight_decay_slider', optimizer, 'weight_decay', 0.0005),
            ('momentum_slider', optimizer, 'momentum', 0.937),
            ('scheduler_dropdown', scheduler, 'type', 'cosine'),
            ('warmup_epochs_slider', scheduler, 'warmup_epochs', 3),
            ('box_loss_gain_slider', loss, 'box_loss_gain', 0.05),
            ('cls_loss_gain_slider', loss, 'cls_loss_gain', 0.5),
            ('obj_loss_gain_slider', loss, 'obj_loss_gain', 1.0),
            ('early_stopping_checkbox', early_stopping, 'enabled', True),
            ('patience_slider', early_stopping, 'patience', 15),
            ('min_delta_slider', early_stopping, 'min_delta', 0.001),
            ('save_best_checkbox', checkpoint, 'save_best', True),
            ('checkpoint_metric_dropdown', checkpoint, 'metric', 'mAP_0.5')
        ]
        
        # Apply updates dengan one-liner
        [set_val(component_key, source_config.get(config_key, default_value)) 
         for component_key, source_config, config_key, default_value in field_mappings]
        
        # Update summary cards jika ada
        self._update_summary_cards(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default hyperparameters configuration"""
        return get_default_hyperparameters_config()
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan hyperparameters-specific success message"""
        self._update_status_panel(ui_components, "💾 Konfigurasi hyperparameter berhasil disimpan", "success")
        self._update_summary_cards(ui_components, config)
        self.logger.success("💾 Hyperparameters config berhasil disimpan")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan hyperparameters-specific reset message"""
        self._update_status_panel(ui_components, "🔄 Konfigurasi hyperparameter berhasil direset", "success")
        self._update_summary_cards(ui_components, config)
        self.logger.success("🔄 Hyperparameters config berhasil direset")
    
    def _update_summary_cards(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update summary cards dengan config hyperparameters yang tersimpan - always visible dengan 4 cards compact"""
        if 'summary_cards' not in ui_components:
            return
        
        training = config.get('training', {})
        optimizer = config.get('optimizer', {})
        scheduler = config.get('scheduler', {})
        early_stopping = config.get('early_stopping', {})
        
        summary_html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0;">
            <div style="background: #e3f2fd; padding: 8px; border-radius: 6px; border-left: 3px solid #2196f3;">
                <h6 style="margin: 0 0 4px 0; color: #1976d2; font-size: 12px;">📊 Training</h6>
                <p style="margin: 1px 0; font-size: 11px;">Epochs: <strong>{training.get('epochs', 100)}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">Batch: <strong>{training.get('batch_size', 16)}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">LR: <strong>{training.get('learning_rate', 0.01)}</strong></p>
            </div>
            <div style="background: #f3e5f5; padding: 8px; border-radius: 6px; border-left: 3px solid #9c27b0;">
                <h6 style="margin: 0 0 4px 0; color: #7b1fa2; font-size: 12px;">⚙️ Optimizer</h6>
                <p style="margin: 1px 0; font-size: 11px;">Type: <strong>{optimizer.get('type', 'SGD')}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">WD: <strong>{optimizer.get('weight_decay', 0.0005)}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">Mom: <strong>{optimizer.get('momentum', 0.937)}</strong></p>
            </div>
            <div style="background: #e8f5e8; padding: 8px; border-radius: 6px; border-left: 3px solid #4caf50;">
                <h6 style="margin: 0 0 4px 0; color: #388e3c; font-size: 12px;">📈 Scheduler</h6>
                <p style="margin: 1px 0; font-size: 11px;">Type: <strong>{scheduler.get('type', 'cosine')}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">Warmup: <strong>{scheduler.get('warmup_epochs', 3)}</strong></p>
            </div>
            <div style="background: #fff3e0; padding: 8px; border-radius: 6px; border-left: 3px solid #ff9800;">
                <h6 style="margin: 0 0 4px 0; color: #f57c00; font-size: 12px;">🛑 Early Stop</h6>
                <p style="margin: 1px 0; font-size: 11px;">Enabled: <strong>{'Ya' if early_stopping.get('enabled', True) else 'Tidak'}</strong></p>
                <p style="margin: 1px 0; font-size: 11px;">Patience: <strong>{early_stopping.get('patience', 15)}</strong></p>
            </div>
        </div>
        """
        
        ui_components['summary_cards'].value = summary_html