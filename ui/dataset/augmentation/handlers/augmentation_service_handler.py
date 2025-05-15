"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_service_handler.py
Deskripsi: Handler untuk interaksi dengan layanan augmentasi dataset
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService

logger = get_logger("augmentation_service_handler")

def get_augmentation_service(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Optional[AugmentationService]:
    """
    Dapatkan instance AugmentationService dengan konfigurasi yang sesuai.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Instance AugmentationService atau None jika gagal
    """
    try:
        # Cek apakah sudah ada di ui_components
        if 'augmentation_service' in ui_components and ui_components['augmentation_service']:
            return ui_components['augmentation_service']
        
        # Dapatkan logger
        logger = ui_components.get('logger')
        
        # Dapatkan data_dir dari ui_components atau default
        data_dir = ui_components.get('data_dir', 'data')
        
        # Buat instance baru
        augmentation_service = AugmentationService(config=config, data_dir=data_dir, logger=logger)
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](augmentation_service)
        
        # Simpan ke ui_components
        ui_components['augmentation_service'] = augmentation_service
        
        return augmentation_service
    except Exception as e:
        if logger:
            logger.error(f"❌ Gagal membuat augmentation service: {str(e)}")
        return None

def register_progress_callback(augmentation_service: AugmentationService, callback: Callable) -> None:
    """
    Register callback untuk progress tracking pada augmentation service.
    
    Args:
        augmentation_service: Instance AugmentationService
        callback: Fungsi callback untuk progress tracking
    """
    if augmentation_service and callback and callable(callback):
        try:
            augmentation_service.register_progress_callback(callback)
        except Exception as e:
            logger.warning(f"⚠️ Gagal register progress callback: {str(e)}")

def execute_augmentation(
    augmentation_service: AugmentationService, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Eksekusi augmentasi dataset dengan parameter yang diberikan.
    
    Args:
        augmentation_service: Instance AugmentationService
        params: Parameter augmentasi
        
    Returns:
        Dictionary hasil augmentasi
    """
    if not augmentation_service:
        return {"status": "error", "message": "Augmentation service tidak tersedia"}
    
    try:
        # Ekstrak parameter
        split = params.get('split', 'train')
        augmentation_types = params.get('augmentation_types', ['combined'])
        num_variations = params.get('num_variations', 2)
        output_prefix = params.get('output_prefix', 'aug')
        validate_results = params.get('validate_results', True)
        process_bboxes = params.get('process_bboxes', True)
        target_balance = params.get('target_balance', True)
        num_workers = params.get('num_workers', 4)
        move_to_preprocessed = params.get('move_to_preprocessed', True)
        target_count = params.get('target_count', 1000)
        
        # Eksekusi augmentasi
        result = augmentation_service.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            validate_results=validate_results,
            process_bboxes=process_bboxes,
            target_balance=target_balance,
            num_workers=num_workers,
            move_to_preprocessed=move_to_preprocessed,
            target_count=target_count
        )
        
        return result
    except Exception as e:
        logger.error(f"❌ Error saat eksekusi augmentasi: {str(e)}")
        return {"status": "error", "message": f"Error saat eksekusi augmentasi: {str(e)}"}

def stop_augmentation(augmentation_service: AugmentationService) -> bool:
    """
    Hentikan proses augmentasi yang sedang berjalan.
    
    Args:
        augmentation_service: Instance AugmentationService
        
    Returns:
        Boolean status keberhasilan
    """
    if not augmentation_service:
        return False
    
    try:
        augmentation_service.stop_processing()
        return True
    except Exception as e:
        logger.error(f"❌ Error saat menghentikan augmentasi: {str(e)}")
        return False
