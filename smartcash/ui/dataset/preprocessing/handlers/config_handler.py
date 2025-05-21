"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk pemrosesan konfigurasi preprocessing dari UI
"""

from typing import Dict, Any
import json

def get_preprocessing_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi preprocessing dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    config = {}
    
    # Get preprocessing options dari area utama
    if 'preprocess_options' in ui_components:
        options = ui_components['preprocess_options']
        
        # Resolution
        if hasattr(options, 'resolution_dropdown') and hasattr(options.resolution_dropdown, 'value'):
            resolution = options.resolution_dropdown.value
            # Parse resolution string ke tuple jika dalam format WxH
            if isinstance(resolution, str) and 'x' in resolution:
                try:
                    width, height = resolution.split('x')
                    config['resolution'] = (int(width), int(height))
                except ValueError:
                    config['resolution'] = resolution
            else:
                config['resolution'] = resolution
        
        # Normalization
        if hasattr(options, 'normalization_dropdown') and hasattr(options.normalization_dropdown, 'value'):
            config['normalization'] = options.normalization_dropdown.value
        
        # Preserve aspect ratio
        if hasattr(options, 'preserve_aspect_ratio_checkbox') and hasattr(options.preserve_aspect_ratio_checkbox, 'value'):
            config['preserve_aspect_ratio'] = options.preserve_aspect_ratio_checkbox.value
            
        # Augmentation
        if hasattr(options, 'augmentation_checkbox') and hasattr(options.augmentation_checkbox, 'value'):
            config['augmentation'] = options.augmentation_checkbox.value
            
        # Force reprocess
        if hasattr(options, 'force_reprocess_checkbox') and hasattr(options.force_reprocess_checkbox, 'value'):
            config['force_reprocess'] = options.force_reprocess_checkbox.value
    
    # Get worker_slider (sekarang ada di kolom kedua)
    if 'worker_slider' in ui_components and hasattr(ui_components['worker_slider'], 'value'):
        config['num_workers'] = ui_components['worker_slider'].value
    
    # Get split_selector (sekarang ada di kolom kedua)
    if 'split_selector' in ui_components and hasattr(ui_components['split_selector'], 'value'):
        split_value = ui_components['split_selector'].value
        split_map = {
            'Train Only': 'train',
            'Validation Only': 'val',
            'Test Only': 'test',
            'All Splits': 'all'
        }
        config['split'] = split_map.get(split_value, 'all')
    
    # Get validation options
    if 'validation_options' in ui_components:
        validation_options = ui_components['validation_options']
        if hasattr(validation_options, 'get_selected') and callable(validation_options.get_selected):
            try:
                config['validation_items'] = validation_options.get_selected()
            except (AttributeError, TypeError):
                # Default validation items jika method tidak tersedia
                config['validation_items'] = ['validate_image_format', 'validate_label_format']
    
    # Get data directories
    config['data_dir'] = ui_components.get('data_dir', 'data')
    config['preprocessed_dir'] = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    # Log konfigurasi yang diambil (debug level)
    logger = ui_components.get('logger')
    if logger and hasattr(logger, 'debug'):
        try:
            logger.debug(f"📋 Config UI preprocessing: {json.dumps(str(config))}")
        except:
            pass
    
    return config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI komponen berdasarkan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
    """
    try:
        # Update opsi preprocessing utama
        if 'preprocess_options' in ui_components:
            options = ui_components['preprocess_options']
            
            # Resolution dropdown
            if hasattr(options, 'resolution_dropdown') and 'resolution' in config:
                resolution = config['resolution']
                if isinstance(resolution, tuple) and len(resolution) == 2:
                    resolution_str = f"{resolution[0]}x{resolution[1]}"
                    if hasattr(options.resolution_dropdown, 'options') and resolution_str in options.resolution_dropdown.options:
                        options.resolution_dropdown.value = resolution_str
                
            # Normalization dropdown
            if hasattr(options, 'normalization_dropdown') and 'normalization' in config:
                if config['normalization'] in options.normalization_dropdown.options:
                    options.normalization_dropdown.value = config['normalization']
                    
            # Preserve aspect ratio checkbox
            if hasattr(options, 'preserve_aspect_ratio_checkbox') and 'preserve_aspect_ratio' in config:
                options.preserve_aspect_ratio_checkbox.value = config.get('preserve_aspect_ratio', True)
                
            # Augmentation checkbox
            if hasattr(options, 'augmentation_checkbox') and 'augmentation' in config:
                options.augmentation_checkbox.value = config.get('augmentation', False)
                
            # Force reprocess checkbox
            if hasattr(options, 'force_reprocess_checkbox') and 'force_reprocess' in config:
                options.force_reprocess_checkbox.value = config.get('force_reprocess', False)
        
        # Update worker slider (di kolom kedua)
        if 'worker_slider' in ui_components and 'num_workers' in config:
            ui_components['worker_slider'].value = config.get('num_workers', 4)
            
        # Update split selector (di kolom kedua)
        if 'split_selector' in ui_components and 'split' in config:
            split_value = config['split']
            split_map = {
                'train': 'Train Only',
                'val': 'Validation Only',
                'test': 'Test Only',
                'all': 'All Splits'
            }
            ui_options = ui_components['split_selector'].options
            mapped_value = split_map.get(split_value, 'All Splits')
            if mapped_value in ui_options:
                ui_components['split_selector'].value = mapped_value
        
        # Update validation options
        if 'validation_options' in ui_components and 'validation_items' in config:
            validation_options = ui_components['validation_options']
            if hasattr(validation_options, 'set_selected') and callable(validation_options.set_selected):
                try:
                    validation_options.set_selected(config['validation_items'])
                except (AttributeError, TypeError):
                    pass
                    
    except Exception as e:
        # Get logger jika tersedia
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'warning'):
            logger.warning(f"⚠️ Error saat update UI dari config: {str(e)}")
        else:
            print(f"⚠️ Error saat update UI dari config: {str(e)}") 