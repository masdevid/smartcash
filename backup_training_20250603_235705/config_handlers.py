"""
File: smartcash/ui/training/handlers/config_handlers.py
Deskripsi: Handlers untuk konfigurasi training dengan integrasi config manager
"""

from typing import Dict, Any
from smartcash.ui.training.components.training_form import update_config_tabs_in_form

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk refresh config dari tabs"""
    
    def handle_refresh_config(button):
        """Handler untuk refresh config dari semua modul"""
        try:
            # Reset log output
            log_output = ui_components.get('log_output')
            log_output and log_output.clear_output(wait=True)
            
            logger = ui_components.get('logger')
            if logger:
                logger.info("🔄 Refreshing configuration dari semua modul...")
            
            # Load fresh config dari config manager
            fresh_config = _load_fresh_config()
            
            # Update config tabs dengan fresh config
            update_config_tabs_in_form(ui_components, fresh_config)
            
            # Trigger config update callback jika ada
            config_callback = ui_components.get('config_update_callback')
            if config_callback:
                config_callback(fresh_config)
            
            if logger:
                logger.success("✅ Konfigurasi berhasil direfresh dari:")
                logger.info("   • Model config")
                logger.info("   • Training config") 
                logger.info("   • Hyperparameters config")
                logger.info("   • Backbone config")
                logger.info("   • Paths config")
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Error refresh config: {str(e)}")
    
    # Register refresh handler ke refresh button
    refresh_button = ui_components.get('refresh_button')
    if refresh_button:
        refresh_button.on_click(handle_refresh_config)
    else:
        logger = ui_components.get('logger')
        if logger:
            logger.warning("⚠️ Refresh button tidak ditemukan untuk setup handler")

def _load_fresh_config() -> Dict[str, Any]:
    """Load fresh config dari semua modul menggunakan config manager"""
    try:
        from smartcash.common.config.manager import get_config_manager
        config_manager = get_config_manager()
        
        # Load configs dari berbagai modul
        configs = {
            'model': config_manager.get_config('model') or _get_default_model_config(),
            'training': config_manager.get_config('training') or _get_default_training_config(),
            'hyperparameters': config_manager.get_config('hyperparameters') or _get_default_hyperparams_config(),
            'backbone': config_manager.get_config('backbone') or _get_default_backbone_config(),
            'detector': config_manager.get_config('detector') or {},
            'strategy': config_manager.get_config('strategy') or _get_default_strategy_config(),
            'paths': config_manager.get_config('paths') or _get_default_paths_config()
        }
        
        return configs
        
    except Exception as e:
        # Return default configs jika gagal load
        return {
            'model': _get_default_model_config(),
            'training': _get_default_training_config(),
            'hyperparameters': _get_default_hyperparams_config(),
            'backbone': _get_default_backbone_config(),
            'strategy': _get_default_strategy_config(),
            'paths': _get_default_paths_config()
        }

def _get_default_model_config() -> Dict[str, Any]:
    """Default model config"""
    return {
        'model_type': 'efficient_optimized',
        'backbone': 'efficientnet_b4',
        'detection_layers': ['banknote'],
        'num_classes': 7,
        'batch_size': 16,
        'transfer_learning': True
    }

def _get_default_training_config() -> Dict[str, Any]:
    """Default training config"""
    return {
        'epochs': 100,
        'early_stopping': True,
        'patience': 10,
        'save_best': True,
        'save_interval': 10
    }

def _get_default_hyperparams_config() -> Dict[str, Any]:
    """Default hyperparameters config"""
    return {
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'image_size': 640,
        'optimizer': {'type': 'SGD'}
    }

def _get_default_backbone_config() -> Dict[str, Any]:
    """Default backbone config"""
    return {
        'type': 'efficientnet_b4',
        'pretrained': True,
        'freeze': False
    }

def _get_default_strategy_config() -> Dict[str, Any]:
    """Default training strategy config"""
    return {
        'multi_scale': False,
        'layer_mode': 'single'
    }

def _get_default_paths_config() -> Dict[str, Any]:
    """Default paths config"""
    return {
        'data_dir': '/data/preprocessed',
        'checkpoint_dir': 'runs/train/checkpoints',
        'tensorboard_dir': 'runs/tensorboard'
    }