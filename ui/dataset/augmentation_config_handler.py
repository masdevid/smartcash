"""
File: smartcash/ui/dataset/augmentation_config_handler.py
Deskripsi: Handler konfigurasi augmentasi dengan dukungan balancing kelas dan sumber preprocessed
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
    
    # Dapatkan jumlah workers dari UI jika tersedia
    num_workers = 4  # Default value
    if len(ui_components['aug_options'].children) > 5:
        num_workers = ui_components['aug_options'].children[5].value
    
    # Cek opsi balancing kelas (opsi baru)
    target_balance = False
    if len(ui_components['aug_options'].children) > 6:
        target_balance = ui_components['aug_options'].children[6].value
    
    # Update config
    config['augmentation'].update({
        'enabled': True,
        'types': aug_types,
        'num_variations': ui_components['aug_options'].children[1].value,
        'output_prefix': ui_components['aug_options'].children[2].value,
        'process_bboxes': ui_components['aug_options'].children[3].value if len(ui_components['aug_options'].children) > 3 else True,
        'validate_results': ui_components['aug_options'].children[4].value if len(ui_components['aug_options'].children) > 4 else True,
        'num_workers': num_workers,
        'target_balance': target_balance,
        'resume': False
    })
    
    # Update preprocessing default lokasi untuk source_dir
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    # Pastikan source_dir tersedia di config
    if 'preprocessed_dir' not in config['preprocessing']:
        config['preprocessing']['preprocessed_dir'] = 'data/preprocessed'
    
    # Pastikan file_prefix untuk penamaan file preprocessed tersedia
    if 'file_prefix' not in config['preprocessing']:
        config['preprocessing']['file_prefix'] = 'rp'
    
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
    
    # Jika config tidak bisa dimuat, gunakan default config
    return load_default_augmentation_config()

def load_default_augmentation_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk augmentasi dataset dengan support balancing dan sumber preprocessed.
    
    Returns:
        Dictionary konfigurasi default
    """
    # Default config dengan nilai standar
    return {
        "augmentation": {
            "enabled": True,
            "types": ["combined", "position", "lighting"],
            "num_variations": 2,
            "output_prefix": "aug",
            "process_bboxes": True,
            "validate_results": True,
            "resume": False,
            "num_workers": 4,
            "target_balance": True,
            "position": {
                "fliplr": 0.5,
                "flipud": 0.0,
                "degrees": 15,
                "translate": 0.1,
                "scale": 0.05,  # Dikurangi dari 0.1 menjadi 0.05 untuk batasi scaling
                "shear": 0.0,
                "rotation_prob": 0.5,
                "max_angle": 15,
                "flip_prob": 0.5,
                "scale_ratio": 0.05  # Dikurangi dari 0.1 menjadi 0.05
            },
            "lighting": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "contrast": 0.3,
                "brightness": 0.3,
                "compress": 0.2,
                "brightness_prob": 0.5,
                "brightness_limit": 0.3,
                "contrast_prob": 0.5,
                "contrast_limit": 0.3
            },
            "extreme": {
                "rotation_min": 30,
                "rotation_max": 90,
                "probability": 0.3
            }
        },
        "preprocessing": {
            "file_prefix": "rp",
            "preprocessed_dir": "data/preprocessed"
        }
    }

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Update UI components dari konfigurasi."""
    if not config or 'augmentation' not in config: config = load_default_augmentation_config()
    aug_config = config['augmentation']
    
    try:
        # Update augmentation types dengan conversion map untuk UI
        if 'types' in aug_config:
            type_map = {'combined': 'Combined (Recommended)', 'position': 'Position Variations', 
                       'lighting': 'Lighting Variations', 'extreme_rotation': 'Extreme Rotation'}
            ui_components['aug_options'].children[0].value = [type_map.get(t, 'Combined (Recommended)') 
                                                         for t in aug_config['types'] 
                                                         if t in type_map.keys()]
        
        # Update inputs dengan values dari config dengan mapping field-index
        options_map = {1: 'num_variations', 2: 'output_prefix', 3: 'process_bboxes', 
                      4: 'validate_results', 5: 'num_workers', 6: 'target_balance'}
        
        # Update semua fields berdasarkan mapping
        for idx, field in options_map.items():
            if idx < len(ui_components['aug_options'].children) and field in aug_config:
                ui_components['aug_options'].children[idx].value = aug_config[field]
                    
    except Exception as e:
        # Log error jika tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].warning(f"⚠️ Error updating UI from config: {str(e)}")
    
    return ui_components