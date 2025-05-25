"""
File: smartcash/ui/dataset/download/handlers/execution_handler.py
Deskripsi: Fixed execution handler dengan integrasi latest progress_tracking dan enhanced callbacks
"""

import time
from typing import Dict, Any, Callable
from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService

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
        if result.get('status') == 'success' and logger:
            logger.info("🎯 Download service menyelesaikan proses dengan sukses")
            
            stats = result.get('stats', {})
            if stats.get('total_images', 0) > 0:
                logger.info(f"📊 Berhasil mengorganisir {stats['total_images']} gambar")
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
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
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📥 {message}")
                elif 'tracker' in ui_components:
                    ui_components['tracker'].update('step', progress, f"📥 {message}")
            
            elif stage == 'extract':
                # Extract progress: masih dalam range download dengan latest integration
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📦 Extract: {message}")
                elif 'tracker' in ui_components:
                    ui_components['tracker'].update('step', progress, f"📦 Extract: {message}")
            
            elif stage == 'organize':
                # Organize progress: 80-95% overall
                base_progress = 80 + int((progress / 100) * 15)
                _update_execution_progress(ui_components, base_progress, f"📁 Organisir: {message}")
                
                # Step progress untuk organize dengan latest integration
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📁 {message}")
                elif 'tracker' in ui_components:
                    ui_components['tracker'].update('step', progress, f"📁 {message}")
                
                # Current progress untuk detail per split jika ada dengan latest integration
                split_info = kwargs.get('split_info', {})
                if split_info and 'current_split' in split_info:
                    split_name = split_info['current_split']
                    split_progress = split_info.get('split_progress', 0)
                    if 'update_progress' in ui_components:
                        ui_components['update_progress']('current', split_progress, f"📂 Memindahkan {split_name}")
                    elif 'tracker' in ui_components:
                        ui_components['tracker'].update('current', split_progress, f"📂 Memindahkan {split_name}")
            
            elif stage == 'verify':
                # Verification stage: 95-100% overall
                base_progress = 95 + int((progress / 100) * 5)
                _update_execution_progress(ui_components, base_progress, f"✅ Verifikasi: {message}")
                
                # Step progress untuk verification dengan latest integration
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"✅ {message}")
                elif 'tracker' in ui_components:
                    ui_components['tracker'].update('step', progress, f"✅ {message}")
            
        except Exception:
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
    # Use latest progress tracking methods dengan fallback support
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, message, color or 'info')
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', progress, message, color)
    else:
        # Fallback untuk legacy progress widgets
        _update_legacy_progress(ui_components, progress, message)

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
    
    except Exception:
        # Silent fail untuk legacy updates
        pass

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