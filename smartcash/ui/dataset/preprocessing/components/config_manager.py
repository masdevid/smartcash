"""
File: smartcash/ui/dataset/preprocessing/components/config_manager.py
Deskripsi: Manajemen konfigurasi untuk UI preprocessing
"""

from typing import Dict, Any


def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari konfigurasi dengan integrasi SimpleConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi dari SimpleConfigManager
    """
    try:
        # Extract preprocessing config
        preprocessing_config = config.get('preprocessing', {})
        
        # Update resolution
        img_size = preprocessing_config.get('img_size', (640, 640))
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            resolution_str = f"{img_size[0]}x{img_size[1]}"
            if 'resolution_dropdown' in ui_components:
                dropdown = ui_components['resolution_dropdown']
                if resolution_str in dropdown.options:
                    dropdown.value = resolution_str
        
        # Update normalization
        normalization = preprocessing_config.get('normalization', 'minmax')
        if 'normalization_dropdown' in ui_components:
            dropdown = ui_components['normalization_dropdown']
            if normalization in dropdown.options:
                dropdown.value = normalization
        
        # Update checkboxes
        checkbox_mappings = {
            'preserve_aspect_ratio_checkbox': preprocessing_config.get('preserve_aspect_ratio', True),
            'augmentation_checkbox': preprocessing_config.get('augmentation', False),
            'force_reprocess_checkbox': preprocessing_config.get('force_reprocess', False)
        }
        
        for checkbox_name, value in checkbox_mappings.items():
            if checkbox_name in ui_components:
                ui_components[checkbox_name].value = value
        
        # Update worker slider
        if 'worker_slider' in ui_components:
            num_workers = preprocessing_config.get('num_workers', 1)
            ui_components['worker_slider'].value = min(max(num_workers, 1), 4)  # Colab safe
        
        # Update split selector
        if 'split_selector' in ui_components and 'reverse_split_map' in ui_components:
            split = preprocessing_config.get('split', 'all')
            split_map = {'all': 'Semua Split', 'train': 'Training', 'val': 'Validasi', 'test': 'Testing'}
            display_value = split_map.get(split, 'Semua Split')
            
            dropdown = ui_components['split_selector']
            if display_value in dropdown.options:
                dropdown.value = display_value
        
    except Exception as e:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Error update UI dari config: {str(e)}")


def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract konfigurasi dari UI components untuk SimpleConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Konfigurasi preprocessing
    """
    config = {
        'preprocessing': {}
    }
    
    try:
        preprocessing_config = config['preprocessing']
        
        # Extract resolution
        if 'resolution_dropdown' in ui_components:
            resolution_str = ui_components['resolution_dropdown'].value
            if 'x' in resolution_str:
                width, height = map(int, resolution_str.split('x'))
                preprocessing_config['img_size'] = (width, height)
        
        # Extract other options
        option_mappings = {
            'normalization_dropdown': ('normalization', 'minmax'),
            'preserve_aspect_ratio_checkbox': ('preserve_aspect_ratio', True),
            'augmentation_checkbox': ('augmentation', False),
            'force_reprocess_checkbox': ('force_reprocess', False),
            'worker_slider': ('num_workers', 1)
        }
        
        for ui_key, (config_key, default_value) in option_mappings.items():
            if ui_key in ui_components:
                preprocessing_config[config_key] = ui_components[ui_key].value
            else:
                preprocessing_config[config_key] = default_value
        
        # Extract split
        if 'split_selector' in ui_components and 'reverse_split_map' in ui_components:
            display_value = ui_components['split_selector'].value
            reverse_map = ui_components['reverse_split_map']
            preprocessing_config['split'] = reverse_map.get(display_value, 'all')
        
        # Add paths
        preprocessing_config['output_dir'] = ui_components.get('preprocessed_dir', 'data/preprocessed')
        config['data'] = {'dir': ui_components.get('data_dir', 'data')}
        
    except Exception as e:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Error extract config dari UI: {str(e)}")
    
    return config