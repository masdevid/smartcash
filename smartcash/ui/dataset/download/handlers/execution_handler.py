"""
File: smartcash/ui/dataset/download/handlers/execution_handler.py
Deskripsi: Fixed execution handler dengan integrasi latest progress_tracking dan enhanced callbacks
"""

import time
from typing import Dict, Any, Callable
from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
from smartcash.components.observer.manager_observer import get_observer_manager

def execute_download_process(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute download process dengan latest progress tracking integration.
    
    Args:
        ui_components: Dictionary komponen UI dengan latest progress methods
        params: Parameter download yang sudah divalidasi
        
    Returns:
        Dictionary berisi hasil download
    """
    logger = ui_components.get('logger')
    start_time = time.time()
    
    try:
        # Pastikan observer_manager tersedia di ui_components
        if 'observer_manager' not in ui_components:
            ui_components['observer_manager'] = get_observer_manager()
            if logger and hasattr(logger, 'debug'):
                logger.debug("🔄 Observer manager initialized")
        
        # Notify start via observer jika tersedia
        observer_manager = ui_components.get('observer_manager')
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DOWNLOAD_START', None, {
                    'message': "Memulai proses download dataset",
                    'timestamp': time.time(),
                    'params': {k: v for k, v in params.items() if k != 'api_key'}
                })
            except Exception as e:
                if logger and hasattr(logger, 'debug'):
                    logger.debug(f"⚠️ Observer notification error: {str(e)}")
        
        # Initialize download service
        _update_execution_progress(ui_components, 40, "🔧 Menginisialisasi service...")
        download_service = UIDownloadService(ui_components)
        
        # Setup enhanced progress callback dengan latest integration
        progress_callback = _create_enhanced_progress_callback(ui_components)
        
        # Execute download dengan latest progress tracking
        _update_execution_progress(ui_components, 50, "📥 Memulai download...")
        result = _execute_with_enhanced_progress_tracking(download_service, params, progress_callback)
        
        # Add execution time
        result['duration'] = time.time() - start_time
        
        # Log hasil dengan success indicator
        if result.get('status') == 'success':
            # Notify complete via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DOWNLOAD_COMPLETE', None, {
                        'message': "Download dataset selesai",
                        'timestamp': time.time(),
                        'duration': result['duration'],
                        'stats': result.get('stats', {})
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
            
            if logger and hasattr(logger, 'info'):
                logger.info("🎯 Download service menyelesaikan proses dengan sukses")
                
                stats = result.get('stats', {})
                if stats.get('total_images', 0) > 0:
                    logger.info(f"📊 Berhasil mengorganisir {stats['total_images']} gambar")
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Notify error via observer
        observer_manager = ui_components.get('observer_manager')
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DOWNLOAD_ERROR', None, {
                    'message': f"Error: {str(e)}",
                    'timestamp': time.time(),
                    'error_details': str(e)
                })
            except Exception:
                pass  # Silent fail untuk observer notification
        
        if logger and hasattr(logger, 'error'):
            logger.error(f"❌ Download execution error: {str(e)}")
            
        return {
            'status': 'error',
            'message': f'Download execution error: {str(e)}',
            'duration': duration
        }

def _create_enhanced_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create enhanced progress callback dengan latest ProgressTracker integration."""
    
    def enhanced_progress_callback(stage: str, progress: int, message: str, **kwargs) -> None:
        """
        Enhanced progress callback dengan support untuk berbagai stage dan latest integration.
        
        Args:
            stage: Stage saat ini (download, extract, organize, verify)
            progress: Progress percentage (0-100)
            message: Progress message
            **kwargs: Additional context information
        """
        try:
            if stage == 'download':
                # Download progress: 50-80% overall
                base_progress = 50 + int((progress / 100) * 30)
                _update_execution_progress(ui_components, base_progress, f"📥 Download: {message}")
                
                # Step progress untuk download detail dengan latest integration
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('step', progress, f"📥 {message}")
                elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                    ui_components['tracker'].update('step', progress, f"📥 {message}")
            
            elif stage == 'extract':
                # Extract progress: masih dalam range download dengan latest integration
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('step', progress, f"📦 Extract: {message}")
                elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                    ui_components['tracker'].update('step', progress, f"📦 {message}")
                
                # Overall progress: 80-85%
                base_progress = 80 + int((progress / 100) * 5)
                _update_execution_progress(ui_components, base_progress, f"📦 Extract: {message}")
                
            elif stage == 'organize':
                # Organize progress: 85-95%
                base_progress = 85 + int((progress / 100) * 10)
                _update_execution_progress(ui_components, base_progress, f"🗂️ Organize: {message}")
                
                # Current progress untuk organize detail
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('current', progress, f"🗂️ {message}")
                elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                    ui_components['tracker'].update('current', progress, f"🗂️ {message}")
                    
            elif stage == 'verify':
                # Verify progress: 95-100%
                base_progress = 95 + int((progress / 100) * 5)
                _update_execution_progress(ui_components, base_progress, f"✅ Verify: {message}")
                
                # Step progress untuk verify detail
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('step', progress, f"✅ {message}")
                elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                    ui_components['tracker'].update('step', progress, f"✅ {message}")
            
            else:
                # Fallback untuk unknown stages
                _update_execution_progress(ui_components, progress, f"{stage}: {message}")
                
            # Notify via observer jika tersedia
            observer_manager = ui_components.get('observer_manager')
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DOWNLOAD_PROGRESS', None, {
                        'stage': stage,
                        'progress': progress,
                        'message': message,
                        'timestamp': time.time(),
                        **kwargs
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger and hasattr(logger, 'error'):
                logger.error(f"❌ Progress callback error: {str(e)}")
            # Silent fail untuk progress callback agar tidak mengganggu proses utama
            pass
    
    return enhanced_progress_callback

def _execute_with_enhanced_progress_tracking(download_service: UIDownloadService, 
                                           params: Dict[str, Any], 
                                           progress_callback: Callable) -> Dict[str, Any]:
    """Execute download dengan enhanced progress tracking dan latest integration."""
    
    try:
        # Setup progress callback ke service
        if hasattr(download_service, 'set_progress_callback'):
            download_service.set_progress_callback(progress_callback)
        
        # Setup additional progress hooks untuk detailed tracking
        if hasattr(download_service, 'set_stage_callback'):
            stage_callback = _create_stage_callback(progress_callback)
            download_service.set_stage_callback(stage_callback)
        
        # Execute download dengan comprehensive progress tracking
        result = download_service.download_dataset(params)
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Enhanced execution error: {str(e)}'
        }

def _create_stage_callback(progress_callback: Callable) -> Callable:
    """Create stage-specific callback untuk detailed progress updates."""
    
    def stage_callback(stage: str, stage_progress: int, stage_message: str, **kwargs):
        """Stage-specific callback dengan enhanced context."""
        
        # Map stage ke progress ranges
        stage_ranges = {
            'init': (40, 50),
            'metadata': (50, 55),
            'download': (55, 80),
            'extract': (80, 85),
            'organize': (85, 95),
            'verify': (95, 100)
        }
        
        if stage in stage_ranges:
            start_range, end_range = stage_ranges[stage]
            mapped_progress = start_range + int((stage_progress / 100) * (end_range - start_range))
            
            # Call main progress callback dengan mapped progress
            progress_callback(stage, mapped_progress, stage_message, **kwargs)
    
    return stage_callback

def _update_execution_progress(ui_components: Dict[str, Any], progress: int, message: str, color: str = None) -> None:
    """Update execution progress dengan latest ProgressTracker integration."""
    try:
        # Use latest progress tracking methods dengan fallback support
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('overall', progress, message, color or 'info')
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
            ui_components['tracker'].update('overall', progress, message, color)
        else:
            # Fallback untuk legacy progress widgets
            _update_legacy_progress(ui_components, progress, message)
            
        # Log progress jika logger tersedia
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'debug'):
            logger.debug(f"📊 Progress update: {progress}% - {message}")
    except Exception as e:
        # Silent fail untuk mencegah error progress mengganggu proses utama
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'debug'):
            logger.debug(f"⚠️ Progress update error: {str(e)}")

def _update_legacy_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Fallback update untuk legacy progress widgets."""
    try:
        # Update legacy progress bar jika tersedia
        if 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                progress_bar.value = progress
            if hasattr(progress_bar, 'description'):
                progress_bar.description = f'Progress: {progress}%'
        
        # Update legacy progress label jika tersedia
        if 'overall_label' in ui_components:
            label = ui_components['overall_label']
            if hasattr(label, 'value'):
                label.value = f"<div style='color: #007bff;'>{message}</div>"
        
        # Update step label jika tersedia
        if 'step_label' in ui_components:
            step_label = ui_components['step_label']
            if hasattr(step_label, 'value'):
                step_label.value = f"<div style='color: #6c757d;'>{message}</div>"
    
    except Exception as e:
        # Silent fail untuk legacy updates
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'debug'):
            logger.debug(f"⚠️ Legacy progress update error: {str(e)}")

def get_execution_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status execution handler untuk debugging."""
    
    status = {
        'progress_integration': 'unknown',
        'available_methods': [],
        'tracker_available': False,
        'legacy_widgets': []
    }
    
    # Check latest progress integration
    latest_methods = ['update_progress', 'show_for_operation', 'complete_operation', 'error_operation']
    available_latest = [method for method in latest_methods if method in ui_components]
    status['available_methods'] = available_latest
    
    if len(available_latest) == len(latest_methods):
        status['progress_integration'] = 'latest'
    elif len(available_latest) > 0:
        status['progress_integration'] = 'partial'
    else:
        status['progress_integration'] = 'legacy'
    
    # Check tracker availability
    status['tracker_available'] = 'tracker' in ui_components and ui_components['tracker'] is not None
    
    # Check legacy widgets
    legacy_widgets = ['progress_bar', 'overall_label', 'step_label']
    available_legacy = [widget for widget in legacy_widgets if widget in ui_components]
    status['legacy_widgets'] = available_legacy
    
    return status