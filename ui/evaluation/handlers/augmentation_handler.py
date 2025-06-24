"""
File: smartcash/ui/evaluation/handlers/augmentation_handler.py
Deskripsi: Handler untuk augmentasi gambar pada proses evaluasi (UI layer)
"""

from typing import Dict, Any, List, Callable
from smartcash.ui.utils.logger_bridge import log_to_service
from smartcash.dataset.evaluation.evaluation_augmentation import EvaluationAugmentationStrategy, get_augmentation_pipeline as get_aug_pipeline

def get_augmentation_pipeline(scenario_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Membuat pipeline augmentasi berdasarkan tipe skenario dengan one-liner style."""
    
    try:
        # Dapatkan info skenario dari config dengan one-liner
        scenario_info = config.get('scenario', {}).get('scenarios', {}).get(scenario_id, {})
        if not scenario_info:
            return {'success': False, 'error': f"Skenario dengan ID {scenario_id} tidak ditemukan"}
        
        # Dapatkan tipe augmentasi dari skenario
        aug_type = scenario_info.get('augmentation_type', 'default')
        
        # Buat pipeline menggunakan fungsi dari evaluation_augmentation untuk kompatibilitas dengan test
        pipeline = get_aug_pipeline({'augmentation_type': aug_type})
        
        return {'success': True, 'pipeline': pipeline, 'type': aug_type}
    except Exception as e:
        return {'success': False, 'error': f"Error membuat augmentation pipeline: {str(e)}"}

def apply_augmentation_to_batch(images: List, bboxes: List, class_labels: List, aug_pipeline: Callable, max_workers: int = 4, logger=None) -> Dict[str, Any]:
    """Terapkan augmentasi pada batch gambar secara paralel dengan one-liner style."""
    # Wrapper untuk log_to_service jika logger diberikan
    def log_wrapper(message, level="info"): logger and log_to_service(logger, message, level)
    
    # Delegasikan ke implementasi di dataset module
    try:
        log_wrapper(f"🔄 Menerapkan augmentasi pada {len(images)} gambar dengan {max_workers} workers")
        result = EvaluationAugmentationStrategy.apply_augmentation_to_batch(images=images, bboxes=bboxes, class_labels=class_labels, 
                                                                         aug_pipeline=aug_pipeline, max_workers=max_workers, logger=logger)
        log_wrapper(f"✅ Augmentasi selesai pada {len(result.get('images', []))} gambar", "success")
        return result
    except Exception as e:
        log_wrapper(f"❌ Error dalam augmentasi batch: {str(e)}", "error")
        # Fallback: kembalikan data asli tanpa augmentasi dengan one-liner style
        return {'success': False, 'error': str(e), 'images': images, 'bboxes': bboxes, 'class_labels': class_labels, 'errors': [str(e)]}
