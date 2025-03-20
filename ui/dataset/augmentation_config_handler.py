from typing import Dict, Any, Optional
import os, yaml
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    config = config or {}
    config['augmentation'] = config.get('augmentation', {})
    type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
    
    aug_types = [type_map.get(t, 'combined') for t in ui_components['aug_options'].children[0].value]
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    rel_augmented_dir = os.path.relpath(augmented_dir, os.getcwd()) if os.path.isabs(augmented_dir) else augmented_dir
    config['augmentation'].update({
        'enabled': True,
        'types': aug_types,
        'num_variations': ui_components['aug_options'].children[1].value,
        'output_prefix': ui_components['aug_options'].children[2].value,
        'process_bboxes': ui_components['aug_options'].children[3].value if len(ui_components['aug_options'].children) > 3 else True,
        'validate_results': ui_components['aug_options'].children[4].value if len(ui_components['aug_options'].children) > 4 else True,
        'resume': ui_components['aug_options'].children[5].value if len(ui_components['aug_options'].children) > 5 else True,
        'output_dir': rel_augmented_dir
    })
    
    if (pos := ui_components.get('position_options')) and hasattr(pos, 'children') and len(pos.children) >= 6:
        config['augmentation']['position'] = {k: pos.children[i].value for i, k in enumerate(['fliplr', 'flipud', 'degrees', 'translate', 'scale', 'shear'])}
    if (light := ui_components.get('lighting_options')) and hasattr(light, 'children') and len(light.children) >= 6:
        config['augmentation']['lighting'] = {k: light.children[i].value for i, k in enumerate(['hsv_h', 'hsv_s', 'hsv_v', 'contrast', 'brightness', 'compress'])}
    if (ext := ui_components.get('extreme_options')) and hasattr(ext, 'children') and len(ext.children) >= 3:
        config['augmentation']['extreme'] = {k: ext.children[i].value for i, k in enumerate(['rotation_min', 'rotation_max', 'probability'])}
    
    return config

def save_augmentation_config(config: Dict[str, Any], config_path: str = "configs/augmentation_config.yaml") -> bool:
    try:
        from smartcash.common.config import get_config_manager
        return bool((cm := get_config_manager()) and cm.config.update(config) or cm.save_config(config_path, create_dirs=True))
    except ImportError:
        return bool((path := Path(config_path)).parent.mkdir(parents=True, exist_ok=True) or yaml.dump(config, open(path, 'w'), default_flow_style=False))

def load_augmentation_config(config_path: str = "configs/augmentation_config.yaml") -> Dict[str, Any]:
    try:
        from smartcash.common.config import get_config_manager
        return (cm := get_config_manager()) and (cm.config or cm.load_config(config_path)) or {}
    except (ImportError, FileNotFoundError):
        return os.path.exists(config_path) and yaml.safe_load(open(config_path, 'r')) or {"augmentation": {"enabled": True, "num_variations": 2, "output_prefix": "aug", "process_bboxes": True, "output_dir": "data/augmented", "validate_results": True, "resume": True, "types": ["combined", "position", "lighting"], "position": {"fliplr": 0.5, "flipud": 0.0, "degrees": 15, "translate": 0.1, "scale": 0.1, "shear": 0.0}, "lighting": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "contrast": 0.3, "brightness": 0.3, "compress": 0.0}, "extreme": {"rotation_min": 30, "rotation_max": 90, "probability": 0.3}}}

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if not config or 'augmentation' not in config: return ui_components
    aug_config = config['augmentation']
    
    try:
        if 'types' in aug_config:
            ui_components['aug_options'].children[0].value = [dict(combined='Combined (Recommended)', position='Position Variations', lighting='Lighting Variations', extreme_rotation='Extreme Rotation').get(t, 'Combined (Recommended)') for t in aug_config['types'] if t in ui_components['aug_options'].children[0].options]
        
        # Preprocess mappings dan gunakan dict comprehension
        aug_map = {1: 'num_variations', 2: 'output_prefix', 3: 'process_bboxes', 4: 'validate_results', 5: 'resume'}
        pos_map = dict(enumerate(['fliplr', 'flipud', 'degrees', 'translate', 'scale', 'shear']))
        light_map = dict(enumerate(['hsv_h', 'hsv_s', 'hsv_v', 'contrast', 'brightness', 'compress']))
        ext_map = dict(enumerate(['rotation_min', 'rotation_max', 'probability']))
        
        # Update dengan satu pass per section
        aug_children = ui_components['aug_options'].children
        {aug_children[i].__setattr__('value', aug_config[k]) for i, k in aug_map.items() if k in aug_config and i < len(aug_children)}
        
        pos_children = ui_components['position_options'].children
        {'position' in aug_config and pos_children[i].__setattr__('value', aug_config['position'][k]) for i, k in pos_map.items() if k in aug_config.get('position', {}) and i < len(pos_children)}
        
        light_children = ui_components['lighting_options'].children
        {'lighting' in aug_config and light_children[i].__setattr__('value', aug_config['lighting'][k]) for i, k in light_map.items() if k in aug_config.get('lighting', {}) and i < len(light_children)}
        
        ext_children = ui_components['extreme_options'].children
        {'extreme' in aug_config and ext_children[i].__setattr__('value', aug_config['extreme'][k]) for i, k in ext_map.items() if k in aug_config.get('extreme', {}) and i < len(ext_children)}
        
    except Exception as e:
        ui_components.get('logger') and ui_components['logger'].warning(f"⚠️ Error updating UI from config: {str(e)}")
    
    return ui_components