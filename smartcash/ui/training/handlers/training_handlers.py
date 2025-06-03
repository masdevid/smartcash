"""
File: smartcash/ui/training/handlers/training_handlers.py
Deskripsi: Consolidated training handlers dengan integrasi YAML config dan model service
"""

import torch
from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.common.config.manager import get_config_manager

def setup_all_training_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk training UI dengan config YAML integration"""
    try:
        # Setup button handlers dengan one-liner registration
        _setup_training_button_handlers(ui_components, config)
        _setup_config_handlers(ui_components, config)
        
        # Log successful setup
        logger = ui_components.get('logger')
        logger and logger.success("✅ Training handlers berhasil disetup") and logger.info("   • Button handlers: start, stop, reset, refresh") and logger.info("   • Config handlers: refresh config")
        
        return ui_components
        
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Error setup training handlers: {str(e)}")
        ui_components['handler_error'] = str(e)
        return ui_components

def _setup_training_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol training dengan YAML config integration"""
    
    def handle_start_training(button):
        """Handler untuk tombol mulai training - delegasi ke training service"""
        try:
            # Reset dan prepare UI
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            _set_button_states(ui_components, training_active=True)
            _show_progress_container(ui_components)
            
            # Load training config dari YAML
            training_config = _load_yaml_training_config()
            
            # Validasi training manager
            training_manager = ui_components.get('training_manager')
            if not training_manager:
                ui_components['logger'].error("❌ Training manager tidak tersedia")
                return _set_button_states(ui_components, training_active=False)
            
            # Start training
            ui_components['logger'].info("🚀 Memulai training...")
            training_manager.start_training(training_config)
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error memulai training: {str(e)}")
            _set_button_states(ui_components, training_active=False)
    
    def handle_stop_training(button):
        """Handler untuk tombol stop training"""
        try:
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            training_manager = ui_components.get('training_manager')
            training_manager and ui_components['logger'].info("🛑 Menghentikan training...") and training_manager.stop_training()
            
            _set_button_states(ui_components, training_active=False)
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error menghentikan training: {str(e)}")
    
    def handle_reset_metrics(button):
        """Handler untuk reset metrics dan chart"""
        try:
            # Clear all outputs dengan one-liner
            [ui_components.get(output) and ui_components[output].clear_output(wait=True) 
             for output in ['log_output', 'chart_output', 'metrics_output']]
            
            # Reset training manager metrics
            training_manager = ui_components.get('training_manager')
            training_manager and hasattr(training_manager, 'reset_metrics') and training_manager.reset_metrics()
            
            ui_components['logger'].success("✅ Metrics dan chart berhasil direset")
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error reset metrics: {str(e)}")
    
    def handle_validate_model(button):
        """Handler untuk validasi model readiness dengan YAML config"""
        try:
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            
            model_manager = ui_components.get('model_manager')
            if not model_manager:
                return ui_components['logger'].error("❌ Model manager tidak tersedia")
            
            ui_components['logger'].info("🔍 Memvalidasi model readiness...")
            
            # Build model jika belum
            if not model_manager.is_model_built():
                ui_components['logger'].info("🔄 Model belum dibangun, membangun model...")
                model_manager.build_model()
            
            # Validasi dengan config info
            yaml_config = _load_yaml_training_config()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model_info = {
                'model_type': yaml_config.get('model', {}).get('type', 'efficient_basic'),
                'backbone': yaml_config.get('model', {}).get('backbone', 'efficientnet_b4'),
                'device': device,
                'batch_size': yaml_config.get('batch_size', 16),
                'epochs': yaml_config.get('epochs', 100),
                'learning_rate': yaml_config.get('learning_rate', 0.001)
            }
            
            ui_components['logger'].success("✅ Model validation berhasil:")
            [ui_components['logger'].info(f"   • {key}: {value}") for key, value in model_info.items()]
                
        except Exception as e:
            ui_components['logger'].error(f"❌ Error validasi model: {str(e)}")
    
    def handle_refresh_config(button):
        """Handler untuk refresh konfigurasi dari YAML files"""
        try:
            ui_components.get('log_output') and ui_components['log_output'].clear_output(wait=True)
            ui_components['logger'].info("🔄 Refreshing configuration dari YAML files...")
            
            # Load fresh config dari YAML
            fresh_config = _load_all_yaml_configs()
            
            # Trigger config update callback
            config_callback = ui_components.get('config_update_callback')
            config_callback and config_callback(fresh_config)
            
            ui_components['logger'].success("✅ Konfigurasi berhasil direfresh dari:")
            [ui_components['logger'].info(f"   • {config_name}") 
             for config_name in ['model_config.yaml', 'training_config.yaml', 'hyperparameters_config.yaml']]
                
        except Exception as e:
            ui_components['logger'].error(f"❌ Error refresh config: {str(e)}")
    
    # Register handlers dengan one-liner
    handler_mapping = {
        'start_button': handle_start_training, 'stop_button': handle_stop_training, 
        'reset_button': handle_reset_metrics, 'validate_button': handle_validate_model, 
        'refresh_button': handle_refresh_config
    }
    [ui_components.get(btn_key) and ui_components[btn_key].on_click(handler) 
     for btn_key, handler in handler_mapping.items()]

def _setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk konfigurasi dengan YAML integration"""
    
    def handle_save_config(button):
        """Handler untuk save config ke YAML files"""
        try:
            config_manager = get_config_manager()
            current_config = _get_current_ui_config(ui_components)
            
            # Save ke multiple YAML files sesuai kategori
            config_mapping = {
                'model_config.yaml': current_config.get('model', {}),
                'training_config.yaml': current_config.get('training', {}),
                'hyperparameters_config.yaml': current_config.get('hyperparameters', {})
            }
            
            saved_files = []
            [config_manager.save_config(config_data, config_name.replace('.yaml', '')) and saved_files.append(config_name)
             for config_name, config_data in config_mapping.items() if config_data]
            
            ui_components['logger'].success(f"✅ Konfigurasi disimpan ke: {', '.join(saved_files)}")
            
        except Exception as e:
            ui_components['logger'].error(f"❌ Error save config: {str(e)}")
    
    # Register save handler jika ada save button
    ui_components.get('save_button') and ui_components['save_button'].on_click(handle_save_config)

def _load_yaml_training_config() -> Dict[str, Any]:
    """Load konfigurasi training dari Google Drive YAML files dengan inheritance"""
    try:
        # Set config manager untuk menggunakan Drive path
        config_manager = get_config_manager(base_dir='/content/drive/MyDrive/SmartCash')
        
        # Load configs dengan inheritance order sesuai _base_ dalam YAML
        base_config = config_manager.get_config('base_config') or {}
        hyperparams = config_manager.get_config('hyperparameters_config') or {}
        model_config = config_manager.get_config('model_config') or {}
        training_config = config_manager.get_config('training_config') or {}
        
        # Merge dengan inheritance order (base < hyperparams < model < training)
        merged_config = {**base_config, **hyperparams, **model_config, **training_config}
        
        # Validasi pretrained model path dari Drive
        pretrained_models_path = '/content/drive/MyDrive/SmartCash/models'
        model_type = merged_config.get('model', {}).get('type', 'efficient_basic')
        
        # Extract specific training parameters sesuai YAML structure
        return {
            'epochs': merged_config.get('epochs', 100),
            'batch_size': merged_config.get('batch_size', 16),
            'learning_rate': merged_config.get('learning_rate', 0.001),
            'weight_decay': merged_config.get('weight_decay', 0.0005),
            'optimizer': merged_config.get('optimizer', 'Adam'),
            'scheduler': merged_config.get('scheduler', 'cosine'),
            'early_stopping': merged_config.get('early_stopping', {}).get('enabled', True),
            'patience': merged_config.get('early_stopping', {}).get('patience', 15),
            'save_best': merged_config.get('save_best', {}).get('enabled', True),
            'save_interval': merged_config.get('training_utils', {}).get('log_metrics_every', 10),
            'model_type': model_type,
            'backbone': merged_config.get('model', {}).get('backbone', 'efficientnet_b4'),
            'layer_mode': merged_config.get('training_utils', {}).get('layer_mode', 'single'),
            'mixed_precision': merged_config.get('training_utils', {}).get('mixed_precision', True),
            'pretrained_models_path': pretrained_models_path
        }
        
    except Exception:
        # Fallback ke default config sesuai YAML values
        return {
            'epochs': 100, 'batch_size': 16, 'learning_rate': 0.001, 'weight_decay': 0.0005,
            'optimizer': 'Adam', 'scheduler': 'cosine', 'early_stopping': True, 'patience': 15,
            'save_best': True, 'save_interval': 10, 'model_type': 'efficient_basic',
            'backbone': 'efficientnet_b4', 'layer_mode': 'single', 'mixed_precision': True
        }

def _load_all_yaml_configs() -> Dict[str, Any]:
    """Load semua konfigurasi dari Google Drive YAML files untuk refresh"""
    try:
        # Set config manager untuk Drive path
        config_manager = get_config_manager(base_dir='/content/drive/MyDrive/SmartCash')
        
        # Load all YAML configs sesuai struktur file di Drive
        configs = {
            'model': config_manager.get_config('model_config') or {},
            'training': config_manager.get_config('training_config') or {},
            'hyperparameters': config_manager.get_config('hyperparameters_config') or {},
            'base': config_manager.get_config('base_config') or {}
        }
        
        return configs
        
    except Exception:
        return {}

def _get_current_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract current config dari UI components untuk save"""
    try:
        # Get config dari UI state atau training manager
        training_manager = ui_components.get('training_manager')
        if training_manager and hasattr(training_manager, 'config'):
            return training_manager.config
        
        # Fallback ke stored config
        return ui_components.get('config', {})
        
    except Exception:
        return {}

def _set_button_states(ui_components: Dict[str, Any], training_active: bool) -> None:
    """Set button states dengan one-liner untuk training mode"""
    start_button, stop_button = ui_components.get('start_button'), ui_components.get('stop_button')
    start_button and (setattr(start_button, 'disabled', training_active), 
                     setattr(start_button, 'description', "🔄 Training..." if training_active else "🚀 Mulai Training"))
    stop_button and setattr(stop_button, 'disabled', not training_active)

def _show_progress_container(ui_components: Dict[str, Any]) -> None:
    """Show progress container dengan one-liner"""
    progress_container = ui_components.get('progress_container')
    progress_container and hasattr(progress_container, 'layout') and setattr(progress_container.layout, 'display', 'flex')

# One-liner utilities untuk validation dan status
validate_training_setup = lambda ui: {'has_model_manager': 'model_manager' in ui, 'has_training_manager': 'training_manager' in ui, 'has_logger': 'logger' in ui}
get_training_status = lambda ui: ui.get('training_manager', {}).get_training_status() if ui.get('training_manager') else None
log_validation_results = lambda ui, validation: [ui['logger'].info(f"   • {key}: {'✅' if value else '❌'}") for key, value in validation.items()] if ui.get('logger') else None