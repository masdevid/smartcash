"""
File: smartcash/ui/dataset/augmentation/handlers/config_mapper.py
Deskripsi: Mapper untuk memetakan nilai UI ke konfigurasi augmentasi dan sebaliknya
"""

from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config

logger = get_logger("augmentation_mapper")

def map_ui_to_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Memetakan nilai komponen UI ke konfigurasi augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan diupdate
        
    Returns:
        Konfigurasi yang sudah diupdate
    """
    if config is None:
        config = {}
    
    # Pastikan struktur dasar tersedia
    if 'augmentation' not in config:
        config['augmentation'] = {}
    
    aug_config = config['augmentation']
    
    try:
        # Dapatkan komponen UI
        aug_options = ui_components.get('aug_options')
        if not aug_options or not hasattr(aug_options, 'children') or len(aug_options.children) < 2:
            logger.warning("⚠️ Komponen aug_options tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Dapatkan tab container
        tab = aug_options.children[1]
        if not hasattr(tab, 'children') or len(tab.children) < 2:
            logger.warning("⚠️ Tab container tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Dapatkan tab dasar dan lanjutan
        basic_tab = tab.children[0]
        advanced_tab = tab.children[1]
        
        if not hasattr(basic_tab, 'children') or not hasattr(advanced_tab, 'children'):
            logger.warning("⚠️ Tab dasar atau lanjutan tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Ekstrak nilai dari tab dasar
        if len(basic_tab.children) >= 4:
            # Prefix (children[1])
            if hasattr(basic_tab.children[1], 'value'):
                prefix_value = basic_tab.children[1].value
                aug_config['prefix'] = prefix_value  # Untuk backward compatibility
                aug_config['output_prefix'] = prefix_value  # Untuk service
            
            # Jumlah variasi (children[2])
            if hasattr(basic_tab.children[2], 'value'):
                factor_value = int(basic_tab.children[2].value)
                aug_config['factor'] = factor_value  # Untuk backward compatibility
                aug_config['num_variations'] = factor_value  # Untuk service
        
        # Ekstrak nilai dari tab lanjutan
        if len(advanced_tab.children) >= 4:
            # Balance kelas (children[0])
            if hasattr(advanced_tab.children[0], 'value'):
                balance_value = bool(advanced_tab.children[0].value)
                aug_config['balance_classes'] = balance_value  # Untuk backward compatibility
                aug_config['target_balance'] = balance_value  # Untuk service
            
            # Target count (children[1])
            if hasattr(advanced_tab.children[1], 'value'):
                aug_config['target_count'] = int(advanced_tab.children[1].value)
            
            # Num workers (children[2])
            if hasattr(advanced_tab.children[2], 'value'):
                aug_config['num_workers'] = int(advanced_tab.children[2].value)
            
            # Move to preprocessed (children[3])
            if hasattr(advanced_tab.children[3], 'value'):
                aug_config['move_to_preprocessed'] = bool(advanced_tab.children[3].value)
        
        # Nilai tetap untuk service
        aug_config['types'] = ['combined']
        aug_config['split'] = 'train'
        aug_config['process_bboxes'] = True
        aug_config['validate_results'] = True
        aug_config['resume'] = False
        
        # Pastikan data path tersedia
        if 'data' not in config:
            config['data'] = {}
        if 'dataset_path' not in config['data']:
            config['data']['dataset_path'] = ui_components.get('data_dir', 'data/preprocessed')
        
        # Validasi konfigurasi
        config = validate_augmentation_config(config)
        
        # Log konfigurasi yang dihasilkan
        logger.debug(f"ℹ️ Konfigurasi augmentasi setelah mapping: {config}")
        
    except Exception as e:
        logger.error(f"❌ Error saat memetakan UI ke config: {str(e)}")
    
    return config

def map_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memetakan nilai konfigurasi augmentasi ke komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi augmentasi
        
    Returns:
        Dictionary komponen UI yang sudah diupdate
    """
    try:
        # Pastikan config valid
        if not config or not isinstance(config, dict) or 'augmentation' not in config:
            logger.warning("⚠️ Konfigurasi tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        aug_config = config['augmentation']
        
        # Dapatkan komponen UI
        aug_options = ui_components.get('aug_options')
        if not aug_options or not hasattr(aug_options, 'children') or len(aug_options.children) < 2:
            logger.warning("⚠️ Komponen aug_options tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Dapatkan tab container
        tab = aug_options.children[1]
        if not hasattr(tab, 'children') or len(tab.children) < 2:
            logger.warning("⚠️ Tab container tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Dapatkan tab dasar dan lanjutan
        basic_tab = tab.children[0]
        advanced_tab = tab.children[1]
        
        if not hasattr(basic_tab, 'children') or not hasattr(advanced_tab, 'children'):
            logger.warning("⚠️ Tab dasar atau lanjutan tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Update nilai di tab dasar
        if len(basic_tab.children) >= 4:
            # Prefix (children[1]) - Prioritaskan output_prefix untuk service
            if hasattr(basic_tab.children[1], 'value'):
                prefix_value = aug_config.get('output_prefix', aug_config.get('prefix', 'aug_'))
                basic_tab.children[1].value = prefix_value
            
            # Jumlah variasi (children[2]) - Prioritaskan num_variations untuk service
            if hasattr(basic_tab.children[2], 'value'):
                factor_value = aug_config.get('num_variations', aug_config.get('factor', 2))
                basic_tab.children[2].value = factor_value
        
        # Update nilai di tab lanjutan
        if len(advanced_tab.children) >= 4:
            # Balance kelas (children[0]) - Prioritaskan target_balance untuk service
            if hasattr(advanced_tab.children[0], 'value'):
                balance_value = aug_config.get('target_balance', aug_config.get('balance_classes', True))
                advanced_tab.children[0].value = balance_value
            
            # Target count (children[1])
            if hasattr(advanced_tab.children[1], 'value'):
                advanced_tab.children[1].value = aug_config.get('target_count', 1000)
            
            # Num workers (children[2])
            if hasattr(advanced_tab.children[2], 'value'):
                advanced_tab.children[2].value = aug_config.get('num_workers', 4)
            
            # Move to preprocessed (children[3])
            if hasattr(advanced_tab.children[3], 'value'):
                advanced_tab.children[3].value = aug_config.get('move_to_preprocessed', True)
        
        # Pastikan data path tersedia
        if 'data' not in config:
            config['data'] = {}
        if 'dataset_path' not in config['data']:
            config['data']['dataset_path'] = ui_components.get('data_dir', 'data/preprocessed')
        
        # Simpan referensi config ke ui_components
        ui_components['config'] = config
        
        # Log konfigurasi yang dihasilkan
        logger.debug(f"ℹ️ UI berhasil diperbarui dari konfigurasi: {config}")
        
    except Exception as e:
        logger.error(f"❌ Error saat memetakan config ke UI: {str(e)}")
    
    return ui_components

def extract_augmentation_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak parameter augmentasi dari komponen UI untuk digunakan dalam proses augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary parameter augmentasi
    """
    # Dapatkan konfigurasi
    config = ui_components.get('config', {})
    aug_config = config.get('augmentation', {})
    
    # Parameter default
    params = {
        'split': 'train',
        'augmentation_types': ['combined'],
        'num_variations': 2,
        'output_prefix': 'aug_',
        'validate_results': True,
        'process_bboxes': True,
        'target_balance': True,
        'num_workers': 4,
        'move_to_preprocessed': True,
        'target_count': 1000
    }
    
    # Update dari konfigurasi
    if aug_config:
        # Prioritaskan parameter yang digunakan oleh service
        params['num_variations'] = aug_config.get('num_variations', aug_config.get('factor', params['num_variations']))
        params['output_prefix'] = aug_config.get('output_prefix', aug_config.get('prefix', params['output_prefix']))
        params['validate_results'] = aug_config.get('validate_results', params['validate_results'])
        params['process_bboxes'] = aug_config.get('process_bboxes', params['process_bboxes'])
        params['target_balance'] = aug_config.get('target_balance', aug_config.get('balance_classes', params['target_balance']))
        params['num_workers'] = aug_config.get('num_workers', params['num_workers'])
        params['move_to_preprocessed'] = aug_config.get('move_to_preprocessed', params['move_to_preprocessed'])
        params['target_count'] = aug_config.get('target_count', params['target_count'])
    
    # Log parameter yang dihasilkan
    logger.debug(f"ℹ️ Parameter augmentasi yang diekstrak: {params}")
    
    return params
