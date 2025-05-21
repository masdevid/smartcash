"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk pemrosesan konfigurasi preprocessing dari UI dengan perbaikan masalah resolusi gambar
"""

from typing import Dict, Any
import json
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

def get_preprocessing_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi preprocessing dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    # Set nilai default untuk resolusi untuk mencegah error "Resolusi gambar harus diisi"
    config = {
        'resolution': DEFAULT_IMG_SIZE  # Default: (640, 640)
    }
    
    # Get preprocessing options dari options container
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
                    # Jika parsing gagal, gunakan nilai default yang sudah diset
                    pass
        
        # Normalization
        if hasattr(options, 'normalization_dropdown') and hasattr(options.normalization_dropdown, 'value'):
            config['normalization'] = options.normalization_dropdown.value
        else:
            config['normalization'] = 'minmax'  # Nilai default
        
        # Preserve aspect ratio
        if hasattr(options, 'preserve_aspect_ratio_checkbox') and hasattr(options.preserve_aspect_ratio_checkbox, 'value'):
            config['preserve_aspect_ratio'] = options.preserve_aspect_ratio_checkbox.value
        else:
            config['preserve_aspect_ratio'] = True  # Nilai default
            
        # Augmentation
        if hasattr(options, 'augmentation_checkbox') and hasattr(options.augmentation_checkbox, 'value'):
            config['augmentation'] = options.augmentation_checkbox.value
        else:
            config['augmentation'] = False  # Nilai default
            
        # Force reprocess
        if hasattr(options, 'force_reprocess_checkbox') and hasattr(options.force_reprocess_checkbox, 'value'):
            config['force_reprocess'] = options.force_reprocess_checkbox.value
        else:
            config['force_reprocess'] = False  # Nilai default
    
    # Get worker_slider (sekarang ada di komponen utama)
    if 'worker_slider' in ui_components and hasattr(ui_components['worker_slider'], 'value'):
        config['num_workers'] = ui_components['worker_slider'].value
    else:
        config['num_workers'] = 4  # Nilai default
    
    # Get split_selector (sekarang ada di komponen utama)
    if 'split_selector' in ui_components and hasattr(ui_components['split_selector'], 'value'):
        split_value = ui_components['split_selector'].value
        split_map = {
            'All Splits': 'all',
            'Train Only': 'train',
            'Validation Only': 'val',
            'Test Only': 'test'
        }
        config['split'] = split_map.get(split_value, 'all')
    else:
        config['split'] = 'all'  # Nilai default
    
    # Get validation options
    if 'validation_options' in ui_components:
        validation_options = ui_components['validation_options']
        if hasattr(validation_options, 'get_selected') and callable(validation_options.get_selected):
            try:
                config['validation_items'] = validation_options.get_selected()
            except (AttributeError, TypeError):
                # Default validation items jika method tidak tersedia
                config['validation_items'] = ['validate_image_format', 'validate_label_format']
        else:
            config['validation_items'] = ['validate_image_format', 'validate_label_format']  # Nilai default
    
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
        # Ekstrak preprocessing config jika ada
        preprocessing_config = config.get('preprocessing', {})
        
        # Update opsi preprocessing utama
        if 'preprocess_options' in ui_components:
            options = ui_components['preprocess_options']
            
            # Resolution dropdown
            resolution = preprocessing_config.get('img_size', config.get('resolution', DEFAULT_IMG_SIZE))
            if resolution and hasattr(options, 'resolution_dropdown'):
                if isinstance(resolution, tuple) and len(resolution) == 2:
                    resolution_str = f"{resolution[0]}x{resolution[1]}"
                    if hasattr(options.resolution_dropdown, 'options'):
                        # Cek apakah resolusi ada dalam opsi
                        if resolution_str in options.resolution_dropdown.options:
                            options.resolution_dropdown.value = resolution_str
                        else:
                            # Jika tidak ada dalam opsi, gunakan resolusi default
                            default_resolution = '640x640'
                            if default_resolution in options.resolution_dropdown.options:
                                options.resolution_dropdown.value = default_resolution
                
            # Normalization dropdown
            normalization = preprocessing_config.get('normalization', config.get('normalization', 'minmax'))
            if normalization and hasattr(options, 'normalization_dropdown'):
                if hasattr(options.normalization_dropdown, 'options'):
                    if normalization in options.normalization_dropdown.options:
                        options.normalization_dropdown.value = normalization
                    else:
                        # Jika tidak ada dalam opsi, gunakan normalisasi default
                        if 'minmax' in options.normalization_dropdown.options:
                            options.normalization_dropdown.value = 'minmax'
                    
            # Preserve aspect ratio checkbox
            preserve_ratio = preprocessing_config.get('preserve_aspect_ratio', config.get('preserve_aspect_ratio', True))
            if hasattr(options, 'preserve_aspect_ratio_checkbox'):
                options.preserve_aspect_ratio_checkbox.value = preserve_ratio
                
            # Augmentation checkbox
            augmentation = preprocessing_config.get('augmentation', config.get('augmentation', False))
            if hasattr(options, 'augmentation_checkbox'):
                options.augmentation_checkbox.value = augmentation
                
            # Force reprocess checkbox
            force_reprocess = preprocessing_config.get('force_reprocess', config.get('force_reprocess', False))
            if hasattr(options, 'force_reprocess_checkbox'):
                options.force_reprocess_checkbox.value = force_reprocess
        
        # Update worker slider (di komponen utama)
        num_workers = preprocessing_config.get('num_workers', config.get('num_workers', 4))
        if 'worker_slider' in ui_components and hasattr(ui_components['worker_slider'], 'value'):
            ui_components['worker_slider'].value = num_workers
            
        # Update split selector (di komponen utama)
        split_value = preprocessing_config.get('split', config.get('split', 'all'))
        if 'split_selector' in ui_components and hasattr(ui_components['split_selector'], 'value'):
            split_map = {
                'train': 'Train Only',
                'val': 'Validation Only',
                'test': 'Test Only',
                'all': 'All Splits'
            }
            mapped_value = split_map.get(split_value, 'All Splits')
            if hasattr(ui_components['split_selector'], 'options'):
                if mapped_value in ui_components['split_selector'].options:
                    ui_components['split_selector'].value = mapped_value
                else:
                    # Jika tidak ada dalam opsi, gunakan split default
                    if 'All Splits' in ui_components['split_selector'].options:
                        ui_components['split_selector'].value = 'All Splits'
        
        # Update validation options
        validation_items = preprocessing_config.get('validation_items', config.get('validation_items', []))
        if 'validation_options' in ui_components and validation_items:
            validation_options = ui_components['validation_options']
            if hasattr(validation_options, 'set_selected') and callable(validation_options.set_selected):
                try:
                    validation_options.set_selected(validation_items)
                except (AttributeError, TypeError):
                    pass
                    
    except Exception as e:
        # Get logger jika tersedia
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'warning'):
            logger.warning(f"⚠️ Error saat update UI dari config: {str(e)}")
        else:
            print(f"⚠️ Error saat update UI dari config: {str(e)}")