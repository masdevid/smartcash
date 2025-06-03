"""
File: smartcash/ui/training/handlers/validation_handler.py
Deskripsi: Handler untuk model validation dan readiness check
"""

from typing import Dict, Any
from smartcash.ui.training.utils.training_status_utils import update_training_status
from smartcash.ui.training.utils.validation_utils import (
    check_model_status, check_pretrained_models, check_training_config,
    check_gpu_status, check_detection_layers
)
from smartcash.ui.training.utils.training_display_utils import display_validation_results


def handle_validate_model(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Handle model readiness dan training preparation validation"""
    logger = ui_components.get('logger')
    model_readiness_display = ui_components.get('model_readiness_display')
    
    if not model_readiness_display:
        logger and logger.warning("⚠️ Model readiness display tidak tersedia")
        return
    
    logger and logger.info("🔍 Memulai validasi model readiness...")
    
    # Perform validation checks
    validation_results = _perform_validation_checks(ui_components, config)
    
    # Display results
    display_validation_results(model_readiness_display, validation_results)
    
    # Update status
    overall_status = _get_overall_validation_status(validation_results)
    update_training_status(ui_components, overall_status['message'], overall_status['type'])
    
    logger and logger.info(f"🔍 Validasi selesai: {overall_status['message']}")


def _perform_validation_checks(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform semua validation checks"""
    return {
        'model_status': check_model_status(ui_components),
        'pretrained_models': check_pretrained_models(),
        'training_config': check_training_config(config),
        'gpu_status': check_gpu_status(),
        'detection_layers': check_detection_layers(config)
    }


def _get_overall_validation_status(results: Dict[str, Any]) -> Dict[str, str]:
    """Get overall validation status"""
    issues = []
    warnings = []
    
    # Check critical issues
    if not results['model_status']['model_ready']:
        issues.append("Model tidak siap")
    
    if not results['training_config']['valid']:
        issues.append("Training config invalid")
    
    if not results['gpu_status']['cuda_available']:
        issues.append("GPU tidak tersedia")
    
    if results['detection_layers']['invalid_layers']:
        issues.append("Detection layers invalid")
    
    # Check warnings
    if not results['pretrained_models']['available_models']:
        warnings.append("Pre-trained models tidak tersedia")
    
    if results['training_config']['warnings']:
        warnings.extend(results['training_config']['warnings'])
    
    # Determine overall status
    if issues:
        return {
            'message': f"❌ Critical issues: {', '.join(issues)}",
            'type': 'error'
        }
    elif warnings:
        return {
            'message': f"⚠️ Warnings: {', '.join(warnings)} - training tetap bisa dilakukan",
            'type': 'warning'
        }
    else:
        return {
            'message': "✅ Semua validasi berhasil - model siap untuk training",
            'type': 'success'
        }


def validate_model_before_training(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Quick validation sebelum training dimulai"""
    # Gunakan training manager jika tersedia
    from smartcash.ui.training.services import TrainingServiceManager
    training_manager = ui_components.get('training_manager')
    
    # Buat training manager jika belum ada
    if not training_manager:
        training_manager = TrainingServiceManager(ui_components, config)
        ui_components['training_manager'] = training_manager
        
    # Validasi kesiapan training menggunakan manager
    return training_manager.validate_training_readiness()