"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Fixed download action dengan explicit progress bar updates
"""

from typing import Dict, Any
import traceback
from pathlib import Path
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager
from IPython.display import display

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi download dengan explicit progress updates."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('download'):
        try:
            if logger:
                logger.info("🚀 Memulai proses download")
            
            _clear_ui_outputs(ui_components)
            _force_show_progress_bars(ui_components)  # Force show progress bars
            
            # Progress: 10% - Validation
            _update_download_progress(ui_components, 10, "Memvalidasi parameter...")
            
            validation_result = _robust_validate_params(ui_components, logger)
            if not validation_result['valid']:
                if logger:
                    logger.error(f"❌ Validasi gagal: {validation_result['message']}")
                raise Exception(validation_result['message'])
            
            # Progress: 20% - Check existing
            _update_download_progress(ui_components, 20, "Memeriksa dataset yang ada...")
            
            params = validation_result['params']
            existing_check = _safe_check_existing_dataset(ui_components)
            
            if existing_check['exists']:
                _show_organized_dataset_confirmation(ui_components, params, existing_check, button_manager)
            else:
                _execute_download_confirmed(ui_components, params)
            
        except Exception as e:
            _update_download_progress(ui_components, 0, f"❌ Error: {str(e)}")
            if logger:
                logger.error(f"❌ Error download: {str(e)}")
            raise

def _force_show_progress_bars(ui_components: Dict[str, Any]) -> None:
    """Force show semua progress bars dengan explicit updates."""
    try:
        # Show container
        if 'progress_container' in ui_components:
            container = ui_components['progress_container']
            if isinstance(container, dict) and 'show_container' in container:
                container['show_container']()
            elif hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'block'
        
        # Force show progress widgets
        progress_widgets = ['overall_progress', 'progress_bar', 'step_progress']
        for widget_key in progress_widgets:
            if widget_key in ui_components and ui_components[widget_key]:
                widget = ui_components[widget_key]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
                    widget.layout.display = 'block'
                    widget.layout.width = '100%'
                    widget.layout.height = '25px' if 'overall' in widget_key or widget_key == 'progress_bar' else '20px'
                
                if hasattr(widget, 'value'):
                    widget.value = 0
                
                if hasattr(widget, 'bar_style'):
                    widget.bar_style = 'info'
                    
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"📊 Error showing progress bars: {str(e)}")

def _update_download_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update download progress dengan explicit widget updates."""
    try:
        # Update overall progress bar
        if 'overall_progress' in ui_components and ui_components['overall_progress']:
            widget = ui_components['overall_progress']
            widget.value = progress
            widget.layout.visibility = 'visible'
            widget.layout.display = 'block'
            if hasattr(widget, 'bar_style'):
                if progress >= 100:
                    widget.bar_style = 'success'
                elif progress > 0:
                    widget.bar_style = 'info'
        
        # Update progress_bar alias
        if 'progress_bar' in ui_components and ui_components['progress_bar']:
            widget = ui_components['progress_bar']
            widget.value = progress
            widget.layout.visibility = 'visible'
            widget.layout.display = 'block'
            if hasattr(widget, 'bar_style'):
                if progress >= 100:
                    widget.bar_style = 'success'
                elif progress > 0:
                    widget.bar_style = 'info'
        
        # Update label
        if 'overall_label' in ui_components and ui_components['overall_label']:
            ui_components['overall_label'].value = f"<div style='color: #495057; font-weight: bold;'>📊 {message} ({progress}%)</div>"
            
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"📊 Error updating progress: {str(e)}")

def _robust_validate_params(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Robust parameter validation."""
    try:
        params = {}
        required_fields = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        
        for field in required_fields:
            try:
                value = _ultra_safe_get_value(ui_components, field, logger)
                params[field] = value
            except Exception as e:
                params[field] = ''
        
        if not params['api_key']:
            try:
                params['api_key'] = _get_api_key_from_sources()
            except Exception:
                pass
        
        missing_fields = [field for field in required_fields if not params[field]]
        
        if missing_fields:
            return {
                'valid': False,
                'message': f"Parameter tidak lengkap: {', '.join(missing_fields)}",
                'params': params
            }
        
        output_validation = _safe_validate_output_directory(params['output_dir'], logger)
        if not output_validation['valid']:
            return {
                'valid': False,
                'message': output_validation['message'],
                'params': params
            }
        
        params['output_dir'] = output_validation['path']
        
        return {
            'valid': True,
            'message': f"Parameter valid - Storage: {output_validation['storage_type']}",
            'params': params
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Validation error: {str(e)}",
            'params': {},
        }

def _ultra_safe_get_value(ui_components: Dict[str, Any], key: str, logger, default: str = '') -> str:
    """Ultra safe value extraction."""
    try:
        if key not in ui_components or ui_components[key] is None:
            return default
        
        component = ui_components[key]
        if not hasattr(component, 'value'):
            return default
        
        value = getattr(component, 'value', default)
        if value is None:
            return default
        
        return str(value).strip()
        
    except Exception:
        return default

def _safe_check_existing_dataset(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Safe check existing dataset."""
    try:
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab,
            is_drive_mounted=env_manager.is_drive_mounted
        )
        
        result = {'exists': False, 'total_images': 0, 'splits': {}, 'paths': paths}
        
        for split in ['train', 'valid', 'test']:
            try:
                split_path = Path(paths[split])
                if split_path.exists():
                    images_dir = split_path / 'images'
                    if images_dir.exists():
                        img_files = list(images_dir.glob('*.*'))
                        if img_files:
                            result['splits'][split] = {'images': len(img_files), 'path': str(split_path)}
                            result['total_images'] += len(img_files)
            except Exception:
                continue
        
        result['exists'] = result['total_images'] > 0
        return result
        
    except Exception:
        return {'exists': False, 'total_images': 0, 'splits': {}}

def _safe_validate_output_directory(output_dir: str, logger) -> Dict[str, Any]:
    """Safe output directory validation."""
    try:
        env_manager = get_environment_manager()
        output_path = Path(output_dir)
        
        if env_manager.is_colab and env_manager.is_drive_mounted:
            if not output_path.is_absolute():
                drive_output = env_manager.drive_path / 'downloads' / output_path
                drive_output.mkdir(parents=True, exist_ok=True)
                return {'valid': True, 'path': str(drive_output), 'storage_type': 'Drive'}
            
            if str(output_path).startswith('/content/drive/MyDrive'):
                output_path.mkdir(parents=True, exist_ok=True)
                return {'valid': True, 'path': str(output_path), 'storage_type': 'Drive'}
            
            drive_output = env_manager.drive_path / 'downloads' / output_path.name
            drive_output.mkdir(parents=True, exist_ok=True)
            return {'valid': True, 'path': str(drive_output), 'storage_type': 'Drive (redirected)'}
        
        output_path.mkdir(parents=True, exist_ok=True)
        return {'valid': True, 'path': str(output_path), 'storage_type': 'Local'}
        
    except Exception as e:
        return {'valid': False, 'message': f"Error output directory: {str(e)}", 'storage_type': 'Unknown'}

def _get_api_key_from_sources() -> str:
    """Get API key dari environment atau Colab secrets."""
    import os
    
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except:
        return ''

def _show_organized_dataset_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], 
                                       existing_info: Dict[str, Any], button_manager) -> None:
    """Show confirmation dengan progress reset."""
    # Reset progress untuk confirmation
    _update_download_progress(ui_components, 0, "Menunggu konfirmasi...")
    
    env_manager = ui_components.get('env_manager')
    storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})" if env_manager and env_manager.is_drive_mounted else "📁 Storage: Local"
    
    split_info = [f"• {split}: {stats['images']} gambar" for split, stats in existing_info['splits'].items()]
    
    message = (
        f"⚠️ Dataset sudah ada!\n\n"
        f"📊 Dataset yang ada:\n" + '\n'.join(split_info) + f"\n• Total: {existing_info['total_images']} gambar\n"
        f"• {storage_info}\n\n"
        f"📥 Dataset baru:\n• Workspace: {params['workspace']}\n• Project: {params['project']}\n• Version: {params['version']}\n\n"
        f"Dataset yang ada akan diganti. Lanjutkan?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        try:
            _execute_download_confirmed(ui_components, params)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download error: {str(e)}")
            raise
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _update_download_progress(ui_components, 0, "Download dibatalkan")
        logger = ui_components.get('logger')
        if logger:
            logger.info("❌ Download dibatalkan")
        raise Exception("Download dibatalkan oleh user")
    
    dialog = create_confirmation_dialog(
        title="⚠️ Konfirmasi Replace Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Replace Dataset",
        cancel_text="Batal"
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _execute_download_confirmed(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Execute download dengan progress tracking."""
    logger = ui_components.get('logger')
    
    # Progress: 30% - Starting download
    _update_download_progress(ui_components, 30, "Memulai download...")
    
    if logger:
        logger.info("✅ Parameter valid - memulai download:")
        for key, value in params.items():
            if key != 'api_key':
                logger.info(f"   • {key}: {value}")
    
    # Execute download dengan progress callback
    result = _execute_enhanced_download_with_progress(ui_components, params)
    
    if result.get('status') == 'success':
        _update_download_progress(ui_components, 100, "Download selesai!")
        if logger:
            stats = result.get('stats', {})
            duration = result.get('duration', 0)
            storage_type = "Drive" if result.get('drive_storage', False) else "Local"
            
            logger.success(f"✅ Download berhasil ({duration:.1f}s)")
            logger.info(f"📁 Storage: {storage_type}")
            logger.info(f"📊 Total gambar: {stats.get('total_images', 0)}")
    else:
        error_msg = result.get('message', 'Unknown error')
        _update_download_progress(ui_components, 0, f"❌ Error: {error_msg}")
        if logger:
            logger.error(f"❌ Download gagal: {error_msg}")
        raise Exception(error_msg)

def _execute_enhanced_download_with_progress(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute download dengan progress updates."""
    try:
        # Progress: 40% - Initialize service
        _update_download_progress(ui_components, 40, "Menginisialisasi service...")
        
        from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
        download_service = UIDownloadService(ui_components)
        
        # Progress: 50% - Start download
        _update_download_progress(ui_components, 50, "Mengunduh dataset...")
        
        # Setup progress callback untuk service
        def progress_callback(stage: str, progress: int, message: str):
            if stage == 'download':
                base_progress = 50 + int((progress / 100) * 30)  # 50-80%
                _update_download_progress(ui_components, base_progress, f"Download: {message}")
            elif stage == 'organize':
                base_progress = 80 + int((progress / 100) * 15)  # 80-95%
                _update_download_progress(ui_components, base_progress, f"Organisir: {message}")
        
        # Execute dengan callback
        result = download_service.download_dataset(params)
        
        return result
        
    except Exception as e:
        return {'status': 'error', 'message': f'Download service error: {str(e)}'}

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass