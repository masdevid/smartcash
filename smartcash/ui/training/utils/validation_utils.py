"""
File: smartcash/ui/training/utils/validation_utils.py
Deskripsi: Utilities untuk model dan training validation
"""

import torch
from pathlib import Path
from typing import Dict, Any


def check_model_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check model manager dan training service status"""
    model_manager = ui_components.get('model_manager')
    training_service = ui_components.get('training_service')
    
    result = {
        'model_manager_available': model_manager is not None,
        'training_service_available': training_service is not None,
        'model_built': False,
        'model_type': None,
        'model_ready': False,
        'error': None
    }
    
    try:
        if model_manager:
            result['model_built'] = model_manager.model is not None
            result['model_type'] = model_manager.model_type
            result['model_ready'] = result['model_built'] and training_service is not None
        
    except Exception as e:
        result['error'] = f"Error checking model status: {str(e)}"
    
    return result


def check_model_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check model manager dan training service status"""
    model_manager = ui_components.get('model_manager')
    training_service = ui_components.get('training_service')
    
    result = {
        'model_manager_available': model_manager is not None,
        'training_service_available': training_service is not None,
        'model_built': False,
        'model_type': None,
        'model_ready': False,
        'error': None
    }
    
    try:
        if model_manager:
            result['model_built'] = model_manager.model is not None
            result['model_type'] = model_manager.model_type
            result['model_ready'] = result['model_built'] and training_service is not None
        
    except Exception as e:
        result['error'] = f"Error checking model status: {str(e)}"
    
    return result


def check_pretrained_models() -> Dict[str, Any]:
    """Check pre-trained models di Google Drive"""
    drive_models_path = Path('/content/drive/MyDrive/SmartCash/models')
    
    result = {
        'drive_mounted': Path('/content/drive/MyDrive').exists(),
        'models_dir_exists': drive_models_path.exists(),
        'available_models': [],
        'missing_models': [],
        'total_size': 0
    }
    
    expected_models = ['efficientnet_b4.pt', 'yolov5s.pt', 'model_metadata.json']
    
    if result['models_dir_exists']:
        for model_file in expected_models:
            model_path = drive_models_path / model_file
            if model_path.exists():
                result['available_models'].append({
                    'name': model_file,
                    'size': format_file_size(model_path.stat().st_size),
                    'path': str(model_path)
                })
                result['total_size'] += model_path.stat().st_size
            else:
                result['missing_models'].append(model_file)
    
    return result


def check_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check training configuration validity"""
    training_config = config.get('training', {})
    
    result = {
        'model_type': training_config.get('model_type', 'efficient_optimized'),
        'backbone': training_config.get('backbone', 'efficientnet_b4'),
        'epochs': training_config.get('epochs', 100),
        'batch_size': training_config.get('batch_size', 16),
        'learning_rate': training_config.get('learning_rate', 0.001),
        'optimizations': {
            'use_attention': config.get('model_optimization', {}).get('use_attention', True),
            'use_residual': config.get('model_optimization', {}).get('use_residual', True),
            'use_ciou': config.get('model_optimization', {}).get('use_ciou', True)
        },
        'valid': True,
        'warnings': []
    }
    
    # Validation checks
    if result['epochs'] <= 0:
        result['warnings'].append("Epochs harus > 0")
        result['valid'] = False
    
    if result['batch_size'] <= 0:
        result['warnings'].append("Batch size harus > 0")
        result['valid'] = False
    
    if result['learning_rate'] <= 0 or result['learning_rate'] > 1:
        result['warnings'].append("Learning rate harus antara 0 dan 1")
        result['valid'] = False
    
    return result


def check_gpu_status() -> Dict[str, Any]:
    """Check GPU status dan memory"""
    result = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'memory_info': {},
        'device_name': None
    }
    
    if result['cuda_available']:
        result['device_count'] = torch.cuda.device_count()
        result['current_device'] = torch.cuda.current_device()
        result['device_name'] = torch.cuda.get_device_name(0)
        result['memory_info'] = get_gpu_memory_info()
    
    return result


def check_detection_layers(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check detection layers configuration"""
    training_config = config.get('training', {})
    detection_layers = training_config.get('detection_layers', ['banknote'])
    
    result = {
        'configured_layers': detection_layers,
        'valid_layers': [],
        'invalid_layers': [],
        'total_classes': 0
    }
    
    try:
        from smartcash.common.layer_config import get_layer_config
        layer_config = get_layer_config()
        all_valid_layers = layer_config.get_layer_names()
        
        for layer in detection_layers:
            if layer in all_valid_layers:
                result['valid_layers'].append(layer)
                layer_data = layer_config.get_layer_config(layer)
                result['total_classes'] += len(layer_data.get('class_ids', []))
            else:
                result['invalid_layers'].append(layer)
                
    except Exception as e:
        result['error'] = f"Error checking layers: {str(e)}"
    
    return result


def get_gpu_memory_info() -> Dict[str, str]:
    """Get GPU memory information"""
    if torch.cuda.is_available():
        return {
            'allocated': format_memory_size(torch.cuda.memory_allocated()),
            'reserved': format_memory_size(torch.cuda.memory_reserved()),
            'max_allocated': format_memory_size(torch.cuda.max_memory_allocated())
        }
    return {'allocated': 'N/A', 'reserved': 'N/A', 'max_allocated': 'N/A'}


def format_file_size(size_bytes: int) -> str:
    """Format file size dalam human readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_memory_size(size_bytes: int) -> str:
    """Format memory size dalam human readable format"""
    return format_file_size(size_bytes)