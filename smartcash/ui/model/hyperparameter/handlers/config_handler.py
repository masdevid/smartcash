"""
File: smartcash/ui/model/hyperparameter/handlers/config_handler.py
Deskripsi: Handler untuk konfigurasi hyperparameter model
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def get_hyperparameter_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get konfigurasi hyperparameter model.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi hyperparameter model
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('model')
        
        # Ensure config structure
        if not config:
            config = get_default_hyperparameter_config()
        elif 'hyperparameter' not in config:
            config['hyperparameter'] = get_default_hyperparameter_config()['hyperparameter']
            
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat get hyperparameter config: {str(e)}")
        return get_default_hyperparameter_config()

def get_default_hyperparameter_config() -> Dict[str, Any]:
    """
    Get konfigurasi default hyperparameter model.
    
    Returns:
        Dictionary konfigurasi default hyperparameter model
    """
    return {
        'hyperparameter': {
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'fl_gamma': 0.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': -1,
            'cache': False,
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'fl_gamma': 0.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True
        }
    }

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get current config
        config = get_hyperparameter_config(ui_components)
        
        # Update loss options
        if 'loss_options' in ui_components:
            loss_options = ui_components['loss_options']
            if hasattr(loss_options, 'children') and len(loss_options.children) >= 4:
                # Update box loss
                config['hyperparameter']['box'] = loss_options.children[0].value
                
                # Update cls loss
                config['hyperparameter']['cls'] = loss_options.children[1].value
                
                # Update dfl loss
                config['hyperparameter']['dfl'] = loss_options.children[2].value
                
                # Update fl gamma
                config['hyperparameter']['fl_gamma'] = loss_options.children[3].value
        
        # Update training options
        if 'training_options' in ui_components:
            training_options = ui_components['training_options']
            if hasattr(training_options, 'children') and len(training_options.children) >= 4:
                # Update label smoothing
                config['hyperparameter']['label_smoothing'] = training_options.children[0].value
                
                # Update nbs
                config['hyperparameter']['nbs'] = training_options.children[1].value
                
                # Update overlap mask
                config['hyperparameter']['overlap_mask'] = training_options.children[2].value
                
                # Update mask ratio
                config['hyperparameter']['mask_ratio'] = training_options.children[3].value
        
        # Update model options
        if 'model_options' in ui_components:
            model_options = ui_components['model_options']
            if hasattr(model_options, 'children') and len(model_options.children) >= 4:
                # Update dropout
                config['hyperparameter']['dropout'] = model_options.children[0].value
                
                # Update val
                config['hyperparameter']['val'] = model_options.children[1].value
                
                # Update plots
                config['hyperparameter']['plots'] = model_options.children[2].value
                
                # Update save
                config['hyperparameter']['save'] = model_options.children[3].value
            
        # Save config
        config_manager = get_config_manager()
        config_manager.set_module_config('model', config)
        
        logger.info("✅ Konfigurasi hyperparameter berhasil diupdate dari UI")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update config dari UI: {str(e)}")
        return get_hyperparameter_config(ui_components)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Ensure config structure
        if not config:
            config = get_default_hyperparameter_config()
        elif 'hyperparameter' not in config:
            config['hyperparameter'] = get_default_hyperparameter_config()['hyperparameter']
            
        # Update UI components
        if 'loss_options' in ui_components:
            loss_options = ui_components['loss_options']
            if hasattr(loss_options, 'children') and len(loss_options.children) >= 4:
                # Update box loss
                loss_options.children[0].value = config['hyperparameter']['box']
                
                # Update cls loss
                loss_options.children[1].value = config['hyperparameter']['cls']
                
                # Update dfl loss
                loss_options.children[2].value = config['hyperparameter']['dfl']
                
                # Update fl gamma
                loss_options.children[3].value = config['hyperparameter']['fl_gamma']
        
        # Update training options
        if 'training_options' in ui_components:
            training_options = ui_components['training_options']
            if hasattr(training_options, 'children') and len(training_options.children) >= 4:
                # Update label smoothing
                training_options.children[0].value = config['hyperparameter']['label_smoothing']
                
                # Update nbs
                training_options.children[1].value = config['hyperparameter']['nbs']
                
                # Update overlap mask
                training_options.children[2].value = config['hyperparameter']['overlap_mask']
                
                # Update mask ratio
                training_options.children[3].value = config['hyperparameter']['mask_ratio']
        
        # Update model options
        if 'model_options' in ui_components:
            model_options = ui_components['model_options']
            if hasattr(model_options, 'children') and len(model_options.children) >= 4:
                # Update dropout
                model_options.children[0].value = config['hyperparameter']['dropout']
                
                # Update val
                model_options.children[1].value = config['hyperparameter']['val']
                
                # Update plots
                model_options.children[2].value = config['hyperparameter']['plots']
                
                # Update save
                model_options.children[3].value = config['hyperparameter']['save']
            
        logger.info("✅ UI hyperparameter berhasil diupdate dari konfigurasi")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat update UI dari config: {str(e)}")
        return ui_components 