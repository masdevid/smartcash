"""
File: smartcash/ui/dataset/preprocessing_config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset
"""

from typing import Dict, Any
import os
import yaml
from pathlib import Path

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi existing
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    if config is None:
        config = {}
    
    # Ensure necessary sections exist
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    preproc_config = config['preprocessing']
    
    # Get preprocessing options
    preprocess_options = ui_components.get('preprocess_options')
    if preprocess_options and hasattr(preprocess_options, 'children'):
        # Image size
        img_size = preprocess_options.children[0].value
        preproc_config['img_size'] = [img_size, img_size]
        
        # Ensure normalization section exists
        if 'normalization' not in preproc_config:
            preproc_config['normalization'] = {}
            
        # Normalization options
        preproc_config['normalization']['enabled'] = preprocess_options.children[1].value
        preproc_config['normalization']['preserve_aspect_ratio'] = preprocess_options.children[2].value
        
        # Cache and workers
        preproc_config['enabled'] = preprocess_options.children[3].value
        preproc_config['num_workers'] = preprocess_options.children[4].value
    
    # Get validation options
    validation_options = ui_components.get('validation_options')
    if validation_options and hasattr(validation_options, 'children'):
        # Ensure validate section exists
        if 'validate' not in preproc_config:
            preproc_config['validate'] = {}
            
        # Validation options
        preproc_config['validate']['enabled'] = validation_options.children[0].value
        preproc_config['validate']['fix_issues'] = validation_options.children[1].value
        preproc_config['validate']['move_invalid'] = validation_options.children[2].value
        preproc_config['validate']['invalid_dir'] = validation_options.children[3].value
    
    # Output directory
    if 'output_dir' not in preproc_config:
        preproc_config['output_dir'] = 'data/preprocessed'
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing ke file.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan keberhasilan operasi
    """
    # Try to use ConfigManager if available
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            return config_manager.save_config(config, config_path, backup=True)
    except ImportError:
        pass
    
    # Fallback: save directly to file
    try:
        # Pastikan direktori ada
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml") -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dari file.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    # Try to use ConfigManager if available
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            return config_manager.load_config(config_path)
    except ImportError:
        pass
    
    # Fallback: load directly from file
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {"preprocessing": {}}