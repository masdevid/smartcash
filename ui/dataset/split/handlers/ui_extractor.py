"""
File: smartcash/ui/dataset/split/handlers/ui_extractor.py
Deskripsi: Ekstraksi nilai dari komponen UI split dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.defaults import normalize_split_ratios, validate_split_ratios

logger = get_logger(__name__)


def extract_ui_values(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak nilai dari komponen UI untuk konfigurasi split.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict berisi nilai yang diekstrak dari UI
    """
    try:
        # Ekstrak ratio values
        train_ratio = ui_components.get('train_slider', {}).value if 'train_slider' in ui_components else 0.7
        valid_ratio = ui_components.get('valid_slider', {}).value if 'valid_slider' in ui_components else 0.15
        test_ratio = ui_components.get('test_slider', {}).value if 'test_slider' in ui_components else 0.15
        
        # Validasi dan normalisasi ratio
        is_valid, message = validate_split_ratios(train_ratio, valid_ratio, test_ratio)
        if not is_valid:
            logger.warning(f"⚠️ {message}, melakukan normalisasi...")
            train_ratio, valid_ratio, test_ratio = normalize_split_ratios(train_ratio, valid_ratio, test_ratio)
        
        # Ekstrak nilai lainnya
        ui_values = {
            'data': {
                'split_ratios': {
                    'train': train_ratio,
                    'valid': valid_ratio,
                    'test': test_ratio
                },
                'stratified_split': ui_components.get('stratified_checkbox', {}).value if 'stratified_checkbox' in ui_components else True,
                'random_seed': ui_components.get('random_seed', {}).value if 'random_seed' in ui_components else 42
            },
            'split_settings': {
                'backup_before_split': ui_components.get('backup_checkbox', {}).value if 'backup_checkbox' in ui_components else True,
                'backup_dir': ui_components.get('backup_dir', {}).value if 'backup_dir' in ui_components else 'data/splits_backup',
                'dataset_path': ui_components.get('dataset_path', {}).value if 'dataset_path' in ui_components else 'data',
                'preprocessed_path': ui_components.get('preprocessed_path', {}).value if 'preprocessed_path' in ui_components else 'data/preprocessed'
            }
        }
        
        logger.debug(f"📊 UI values extracted: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")
        return ui_values
        
    except Exception as e:
        logger.error(f"💥 Error extracting UI values: {str(e)}")
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        return get_default_split_config()


def extract_ratio_values(ui_components: Dict[str, Any]) -> tuple[float, float, float]:
    """
    Ekstrak hanya nilai ratio dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (train_ratio, valid_ratio, test_ratio)
    """
    try:
        train_ratio = ui_components.get('train_slider', {}).value if 'train_slider' in ui_components else 0.7
        valid_ratio = ui_components.get('valid_slider', {}).value if 'valid_slider' in ui_components else 0.15
        test_ratio = ui_components.get('test_slider', {}).value if 'test_slider' in ui_components else 0.15
        
        return train_ratio, valid_ratio, test_ratio
        
    except Exception as e:
        logger.error(f"💥 Error extracting ratio values: {str(e)}")
        return 0.7, 0.15, 0.15


def validate_extracted_values(ui_values: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validasi nilai yang diekstrak dari UI.
    
    Args:
        ui_values: Dictionary nilai UI
        
    Returns:
        Tuple (is_valid, message)
    """
    try:
        # Validasi struktur
        if 'data' not in ui_values or 'split_ratios' not in ui_values['data']:
            return False, "Struktur konfigurasi tidak valid"
        
        # Validasi ratio
        ratios = ui_values['data']['split_ratios']
        return validate_split_ratios(
            ratios.get('train', 0),
            ratios.get('valid', 0), 
            ratios.get('test', 0)
        )
        
    except Exception as e:
        return False, f"Error validasi: {str(e)}"