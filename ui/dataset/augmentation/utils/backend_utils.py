"""
File: smartcash/ui/dataset/augmentation/utils/backend_utils.py
Deskripsi: Backend utilities untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

def execute_augmentation_pipeline(config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Execute augmentation pipeline dengan centralized error handling
    
    Args:
        config: Dictionary berisi konfigurasi augmentation
        progress_callback: Callback function untuk progress tracking
        
    Returns:
        Dictionary berisi hasil operasi
    """
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        
        # Create service dengan config
        service = create_augmentation_service(config)
        
        # Get target split
        target_split = config.get('augmentation', {}).get('target_split', 'train')
        
        # Execute pipeline
        result = service.run_augmentation_pipeline(
            target_split=target_split,
            progress_callback=progress_callback
        )
        
        # Ensure result uses 'status' key for consistency
        if 'success' in result and 'status' not in result:
            result['status'] = result.pop('success')
        
        return result
    except Exception as e:
        logger.error(f"Error executing augmentation pipeline: {str(e)}")
        return {
            'status': False,
            'message': f"Error executing augmentation pipeline: {str(e)}"
        }

def execute_dataset_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute dataset check dengan centralized error handling
    
    Args:
        config: Dictionary berisi konfigurasi augmentation
        
    Returns:
        Dictionary berisi hasil operasi
    """
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        
        # Create service dengan config
        service = create_augmentation_service(config)
        
        # Execute check
        result = service.get_augmentation_status()
        
        # Ensure result uses 'status' key for consistency
        if 'success' in result and 'status' not in result:
            result['status'] = result.pop('success')
        
        return result
    except Exception as e:
        logger.error(f"Error executing dataset check: {str(e)}")
        return {
            'status': False,
            'message': f"Error executing dataset check: {str(e)}",
            'service_ready': False
        }

def execute_cleanup(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute cleanup dengan centralized error handling
    
    Args:
        config: Dictionary berisi konfigurasi augmentation
        
    Returns:
        Dictionary berisi hasil operasi
    """
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        
        # Create service dengan config
        service = create_augmentation_service(config)
        
        # Get target split and cleanup target
        target_split = config.get('augmentation', {}).get('target_split', 'train')
        cleanup_target = config.get('cleanup', {}).get('default_target', 'both')
        
        # Execute cleanup
        result = service.cleanup_data(
            target_split=target_split,
            target=cleanup_target
        )
        
        # Ensure result uses 'status' key for consistency
        if 'success' in result and 'status' not in result:
            result['status'] = result.pop('success')
        
        return result
    except Exception as e:
        logger.error(f"Error executing cleanup: {str(e)}")
        return {
            'status': False,
            'message': f"Error executing cleanup: {str(e)}"
        }

def create_live_preview(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create live preview dengan centralized error handling
    
    Args:
        config: Dictionary berisi konfigurasi augmentation
        
    Returns:
        Dictionary berisi hasil operasi
    """
    try:
        from smartcash.dataset.augmentor.service import create_augmentation_service
        
        # Create service dengan config
        service = create_augmentation_service(config)
        
        # Get target split
        target_split = config.get('augmentation', {}).get('target_split', 'train')
        
        # Create preview
        result = service.create_live_preview(target_split=target_split)
        
        # Ensure result uses 'status' key for consistency
        if 'success' in result and 'status' not in result:
            result['status'] = result.pop('success')
        
        return result
    except Exception as e:
        logger.error(f"Error creating live preview: {str(e)}")
        return {
            'status': False,
            'message': f"Error creating live preview: {str(e)}"
        }
