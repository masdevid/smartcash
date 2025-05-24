"""
File: smartcash/ui/dataset/download/handlers/execution_handler.py
Deskripsi: Handler untuk eksekusi download dengan progress tracking yang detail dan persistent
"""

import time
from typing import Dict, Any, Callable
from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService

def execute_download_process(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute download process dengan tqdm progress tracking yang persistent.
    
    Args:
        ui_components: Dictionary komponen UI
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
        
        # Setup enhanced progress callback
        progress_callback = _create_progress_callback(ui_components)
        
        # Execute download dengan progress tracking
        _update_execution_progress(ui_components, 50, "📥 Memulai download...")
        result = _execute_with_progress_tracking(download_service, params, progress_callback)
        
        # Add execution time
        result['duration'] = time.time() - start_time
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'status': 'error',
            'message': f'Download execution error: {str(e)}',
            'duration': duration
        }

def _create_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create enhanced progress callback untuk detailed tracking."""
    
    def progress_callback(stage: str, progress: int, message: str, **kwargs) -> None:
        """
        Enhanced progress callback dengan support untuk berbagai stage.
        
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
                
                # Step progress untuk download detail
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📥 {message}")
            
            elif stage == 'extract':
                # Extract progress: masih dalam range download
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📦 Extract: {message}")
            
            elif stage == 'organize':
                # Organize progress: 80-95% overall
                base_progress = 80 + int((progress / 100) * 15)
                _update_execution_progress(ui_components, base_progress, f"📁 Organisir: {message}")
                
                # Step progress untuk organize
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"📁 {message}")
                
                # Current progress untuk detail per split jika ada
                split_info = kwargs.get('split_info', {})
                if split_info and 'current_split' in split_info:
                    split_name = split_info['current_split']
                    split_progress = split_info.get('split_progress', 0)
                    if 'update_progress' in ui_components:
                        ui_components['update_progress']('current', split_progress, 
                                                       f"📂 Memindahkan {split_name}")
            
            elif stage == 'verify':
                # Verify progress: 95-99% overall
                base_progress = 95 + int((progress / 100) * 4)
                _update_execution_progress(ui_components, base_progress, f"✅ Verifikasi: {message}")
                
                if 'update_progress' in ui_components:
                    ui_components['update_progress']('step', progress, f"✅ {message}")
            
        except Exception:
            # Silent fail untuk progress callback agar tidak mengganggu proses utama
            pass
    
    return progress_callback

def _execute_with_progress_tracking(download_service: UIDownloadService, 
                                  params: Dict[str, Any], 
                                  progress_callback: Callable) -> Dict[str, Any]:
    """Execute download dengan progress tracking yang comprehensive."""
    
    try:
        # Setup progress callback ke service
        if hasattr(download_service, 'set_progress_callback'):
            download_service.set_progress_callback(progress_callback)
        
        # Execute download dengan enhanced monitoring
        result = download_service.download_dataset(params)
        
        if result.get('status') == 'success':
            # Validate hasil download
            validation_result = _validate_download_result(result)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'message': f"Validasi hasil gagal: {validation_result['message']}"
                }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Execution error: {str(e)}'
        }

def _validate_download_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi hasil download untuk memastikan integritas."""
    try:
        stats = result.get('stats', {})
        
        # Check basic stats
        total_images = stats.get('total_images', 0)
        if total_images == 0:
            return {
                'valid': False,
                'message': "Tidak ada gambar yang berhasil didownload"
            }
        
        # Check splits
        splits_with_images = []
        for split in ['train', 'valid', 'test']:
            split_key = f'{split}_images'
            if stats.get(split_key, 0) > 0:
                splits_with_images.append(split)
        
        if not splits_with_images:
            return {
                'valid': False,
                'message': "Tidak ada split yang berhasil didownload"
            }
        
        # Minimal requirement: harus ada train split
        if 'train' not in splits_with_images:
            return {
                'valid': False,
                'message': "Split training tidak ditemukan atau kosong"
            }
        
        return {
            'valid': True,
            'message': f"Validasi berhasil: {total_images} gambar dalam {len(splits_with_images)} split"
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error validasi: {str(e)}"
        }

def _update_execution_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update execution progress dengan tqdm."""
    if 'update_progress' in ui_components:
        color = 'success' if progress >= 100 else 'info' if progress > 0 else ''
        ui_components['update_progress']('overall', progress, message, color)