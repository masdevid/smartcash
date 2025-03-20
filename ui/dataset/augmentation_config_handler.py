"""
File: smartcash/ui/dataset/augmentation_config_handler.py
Deskripsi: Handler konfigurasi augmentasi dengan penanganan lokasi dataset
"""

from typing import Dict, Any, Optional
import os, yaml
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Update konfigurasi dari UI components."""
    config = config or {}
    config['augmentation'] = config.get('augmentation', {})
    
    # Map UI types to config types
    type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 
                'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
    
    # Get augmentation types from UI
    aug_types = [type_map.get(t, 'combined') for t in ui_components['aug_options'].children[0].value]
    
    # Get directories from inputs
    data_dir = ui_components.get('data_dir', 'data')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    # Update dari input lokasi jika tersedia
    if 'data_dir_input' in ui_components:
        data_dir = ui_components['data_dir_input'].value
    if 'output_dir_input' in ui_components:
        augmented_dir = ui_components['output_dir_input'].value
    
    # Use relative paths for storage
    rel_augmented_dir = os.path.relpath(augmented_dir, os.getcwd()) if os.path.isabs(augmented_dir) else augmented_dir
    
    # Update config
    config['augmentation'].update({
        'enabled': True,
        'types': aug_types,
        'num_variations': ui_components['aug_options'].children[1].value,
        'output_prefix': ui_components['aug_options'].children[2].value,
        'process_bboxes': ui_components['aug_options'].children[3].value if len(ui_components['aug_options'].children) > 3 else True,
        'validate_results': ui_components['aug_options'].children[4].value if len(ui_components['aug_options'].children) > 4 else True,
        'resume': False,  # Selalu false karena opsi ini dihilangkan
        'output_dir': rel_augmented_dir
    })
    
    # Update data directory
    config['data'] = config.get('data', {})
    config['data']['dir'] = os.path.relpath(data_dir, os.getcwd()) if os.path.isabs(data_dir) else data_dir
    
    # Update advanced options jika tersedia
    if (pos := ui_components.get('position_options')) and hasattr(pos, 'children') and len(pos.children) >= 6:
        config['augmentation']['position'] = {k: pos.children[i].value for i, k in enumerate(['fliplr', 'flipud', 'degrees', 'translate', 'scale', 'shear'])}
    if (light := ui_components.get('lighting_options')) and hasattr(light, 'children') and len(light.children) >= 6:
        config['augmentation']['lighting'] = {k: light.children[i].value for i, k in enumerate(['hsv_h', 'hsv_s', 'hsv_v', 'contrast', 'brightness', 'compress'])}
    if (ext := ui_components.get('extreme_options')) and hasattr(ext, 'children') and len(ext.children) >= 3:
        config['augmentation']['extreme'] = {k: ext.children[i].value for i, k in enumerate(['rotation_min', 'rotation_max', 'probability'])}
    
    return config

def save_augmentation_config(config: Dict[str, Any], config_path: str = "configs/augmentation_config.yaml") -> bool:
    """Simpan konfigurasi augmentasi ke file."""
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            # Update config internal
            config_manager.config.update(config)
            # Simpan ke file
            config_manager.save_config(config_path, create_dirs=True)
            return True
    except ImportError:
        # Fallback: simpan langsung dengan yaml
        try:
            path = Path(config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            return True
        except Exception:
            return False
    return False

def load_augmentation_config(config_path: str = "configs/augmentation_config.yaml") -> Dict[str, Any]:
    """Load konfigurasi augmentasi dari file."""
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            # Coba load dari config manager
            if config_manager.config:
                return config_manager.config
            # Atau load dari file
            return config_manager.load_config(config_path)
    except (ImportError, FileNotFoundError):
        # Fallback: load langsung dengan yaml
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
    
    # Default config jika tidak ada
    return {
        "augmentation": {
            "enabled": True,
            "num_variations": 2,
            "output_prefix": "aug",
            "process_bboxes": True,
            "output_dir": "data/augmented",
            "validate_results": True,
            "resume": False,
            "types": ["combined", "position", "lighting"],
            "position": {"fliplr": 0.5, "flipud": 0.0, "degrees": 15, "translate": 0.1, "scale": 0.1, "shear": 0.0},
            "lighting": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "contrast": 0.3, "brightness": 0.3, "compress": 0.0},
            "extreme": {"rotation_min": 30, "rotation_max": 90, "probability": 0.3}
        }
    }

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Update UI components dari konfigurasi."""
    if not config or 'augmentation' not in config:
        return ui_components
        
    aug_config = config['augmentation']
    
    # Update lokasi dataset jika tersedia
    if 'data' in config and 'dir' in config['data']:
        if 'data_dir_input' in ui_components:
            ui_components['data_dir_input'].value = config['data']['dir']
            ui_components['data_dir'] = config['data']['dir']
    
    # Update lokasi output
    if 'output_dir' in aug_config:
        if 'output_dir_input' in ui_components:
            ui_components['output_dir_input'].value = aug_config['output_dir']
            ui_components['augmented_dir'] = aug_config['output_dir']
    
    try:
        # Update augmentation types
        if 'types' in aug_config:
            type_map = {'combined': 'Combined (Recommended)', 'position': 'Position Variations', 
                      'lighting': 'Lighting Variations', 'extreme_rotation': 'Extreme Rotation'}
            ui_components['aug_options'].children[0].value = [type_map.get(t, 'Combined (Recommended)') 
                                                         for t in aug_config['types'] 
                                                         if t in type_map.keys()]
        
        # Update inputs dengan values dari config
        # Define field mappings
        options_map = {
            1: 'num_variations',
            2: 'output_prefix',
            3: 'process_bboxes', 
            4: 'validate_results'
        }
        
        # Update aug_options
        for idx, field in options_map.items():
            if idx < len(ui_components['aug_options'].children) and field in aug_config:
                ui_components['aug_options'].children[idx].value = aug_config[field]
        
        # Update position options
        if 'position' in aug_config and 'position_options' in ui_components:
            pos_fields = ['fliplr', 'flipud', 'degrees', 'translate', 'scale', 'shear']
            for idx, field in enumerate(pos_fields):
                if idx < len(ui_components['position_options'].children) and field in aug_config['position']:
                    ui_components['position_options'].children[idx].value = aug_config['position'][field]
        
        # Update lighting options
        if 'lighting' in aug_config and 'lighting_options' in ui_components:
            light_fields = ['hsv_h', 'hsv_s', 'hsv_v', 'contrast', 'brightness', 'compress']
            for idx, field in enumerate(light_fields):
                if idx < len(ui_components['lighting_options'].children) and field in aug_config['lighting']:
                    ui_components['lighting_options'].children[idx].value = aug_config['lighting'][field]
        
        # Update extreme options
        if 'extreme' in aug_config and 'extreme_options' in ui_components:
            ext_fields = ['rotation_min', 'rotation_max', 'probability']
            for idx, field in enumerate(ext_fields):
                if idx < len(ui_components['extreme_options'].children) and field in aug_config['extreme']:
                    ui_components['extreme_options'].children[idx].value = aug_config['extreme'][field]
                    
    except Exception as e:
        # Log error jika tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].warning(f"⚠️ Error updating UI from config: {str(e)}")
    
    return ui_components