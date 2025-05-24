"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Download action dengan tqdm progress tracking - FIXED
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager
from IPython.display import display

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi download dengan tqdm progress."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('download'):
        try:
            logger and logger.info("🚀 Memulai proses download")
            
            # PERBAIKAN: Inisialisasi progress tracking untuk download operation
            _initialize_download_progress(ui_components)
            
            _clear_ui_outputs(ui_components)
            _update_download_progress(ui_components, 10, "Memvalidasi parameter...")
            
            validation_result = _robust_validate_params(ui_components, logger)
            if not validation_result['valid']:
                logger and logger.error(f"❌ Validasi gagal: {validation_result['message']}")
                _error_download_progress(ui_components, f"Validasi gagal: {validation_result['message']}")
                raise Exception(validation_result['message'])
            
            _update_download_progress(ui_components, 20, "Memeriksa dataset yang ada...")
            
            params = validation_result['params']
            existing_check = _safe_check_existing_dataset(ui_components)
            
            if existing_check['exists']:
                _show_organized_dataset_confirmation(ui_components, params, existing_check, button_manager)
            else:
                _execute_download_confirmed(ui_components, params)
            
        except Exception as e:
            logger and logger.error(f"❌ Error download: {str(e)}")
            _error_download_progress(ui_components, f"Error download: {str(e)}")
            raise

def _initialize_download_progress(ui_components: Dict[str, Any]) -> None:
    """Inisialisasi progress tracking untuk download operation."""
    try:
        # Show progress container untuk download operation
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('download')
        elif 'show_container' in ui_components:
            ui_components['show_container']()
        
        # Update status awal
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 0, "Memulai download...")
            ui_components['update_progress']('step', 0, "Inisialisasi...")
            
    except Exception as e:
        print(f"Warning: Could not initialize progress tracking: {e}")

def _update_download_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update download progress dengan tqdm."""
    try:
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', progress, message)
        else:
            print(f"Progress {progress}%: {message}")
    except Exception as e:
        print(f"Progress update error: {e}")

def _update_step_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update step progress dengan tqdm."""
    try:
        if 'update_progress' in ui_components:
            ui_components['update_progress']('step', progress, message)
        else:
            print(f"Step {progress}%: {message}")
    except Exception as e:
        print(f"Step progress update error: {e}")

def _complete_download_progress(ui_components: Dict[str, Any], message: str = "Download selesai") -> None:
    """Complete download progress."""
    try:
        if 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
        else:
            print(f"✅ {message}")
    except Exception as e:
        print(f"Complete progress error: {e}")

def _error_download_progress(ui_components: Dict[str, Any], message: str = "Download error") -> None:
    """Set error state untuk download progress."""
    try:
        if 'error_operation' in ui_components:
            ui_components['error_operation'](message)
        else:
            print(f"❌ {message}")
    except Exception as e:
        print(f"Error progress error: {e}")

def _robust_validate_params(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Robust parameter validation."""
    try:
        params = {}
        required_fields = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        
        for field in required_fields:
            value = _ultra_safe_get_value(ui_components, field, logger)
            params[field] = value
        
        if not params['api_key']:
            params['api_key'] = _get_api_key_from_sources()
        
        missing_fields = [field for field in required_fields if not params[field]]
        
        if missing_fields:
            return {'valid': False, 'message': f"Parameter tidak lengkap: {', '.join(missing_fields)}", 'params': params}
        
        output_validation = _safe_validate_output_directory(params['output_dir'], logger)
        if not output_validation['valid']:
            return {'valid': False, 'message': output_validation['message'], 'params': params}
        
        params['output_dir'] = output_validation['path']
        
        return {'valid': True, 'message': f"Parameter valid - Storage: {output_validation['storage_type']}", 'params': params}
        
    except Exception as e:
        return {'valid': False, 'message': f"Validation error: {str(e)}", 'params': {}}

def _ultra_safe_get_value(ui_components: Dict[str, Any], key: str, logger, default: str = '') -> str:
    """Ultra safe value extraction."""
    try:
        if key not in ui_components or ui_components[key] is None:
            return default
        
        component = ui_components[key]
        if not hasattr(component, 'value'):
            return default
        
        value = getattr(component, 'value', default)
        return str(value).strip() if value is not None else default
        
    except Exception:
        return default

def _safe_check_existing_dataset(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Safe check existing dataset."""
    try:
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(is_colab=env_manager.is_colab, is_drive_mounted=env_manager.is_drive_mounted)
        
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
    _update_download_progress(ui_components, 25, "Menunggu konfirmasi user...")
    _update_step_progress(ui_components, 0, "Konfirmasi replace dataset")
    
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
            logger and logger.error(f"❌ Download error: {str(e)}")
            _error_download_progress(ui_components, f"Download error: {str(e)}")
            raise
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        _update_download_progress(ui_components, 0, "Download dibatalkan")
        _error_download_progress(ui_components, "Download dibatalkan oleh user")
        logger = ui_components.get('logger')
        logger and logger.info("❌ Download dibatalkan")
        raise Exception("Download dibatalkan oleh user")
    
    dialog = create_confirmation_dialog(
        title="⚠️ Konfirmasi Replace Dataset",
        message=message, on_confirm=on_confirm, on_cancel=on_cancel,
        confirm_text="Ya, Replace Dataset", cancel_text="Batal"
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _execute_download_confirmed(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Execute download dengan tqdm progress tracking."""
    logger = ui_components.get('logger')
    
    _update_download_progress(ui_components, 30, "Memulai download...")
    _update_step_progress(ui_components, 0, "Inisialisasi download service")
    
    if logger:
        logger.info("✅ Parameter valid - memulai download:")
        for key, value in params.items():
            if key != 'api_key':
                logger.info(f"   • {key}: {value}")
    
    result = _execute_enhanced_download_with_progress(ui_components, params)
    
    if result.get('status') == 'success':
        if logger:
            stats = result.get('stats', {})
            duration = result.get('duration', 0)
            storage_type = "Drive" if result.get('drive_storage', False) else "Local"
            
            logger.success(f"✅ Download berhasil ({duration:.1f}s)")
            logger.info(f"📁 Storage: {storage_type}")
            logger.info(f"📊 Total gambar: {stats.get('total_images', 0)}")
        
        # Complete progress dengan success message
        _complete_download_progress(ui_components, f"Download berhasil! Total: {stats.get('total_images', 0)} gambar")
    else:
        error_msg = result.get('message', 'Unknown error')
        if logger:
            logger.error(f"❌ Download gagal: {error_msg}")
        _error_download_progress(ui_components, f"Download gagal: {error_msg}")
        raise Exception(error_msg)

def _execute_enhanced_download_with_progress(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute download dengan tqdm progress updates."""
    try:
        _update_download_progress(ui_components, 40, "Menginisialisasi service...")
        _update_step_progress(ui_components, 10, "Loading download service")
        
        from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
        download_service = UIDownloadService(ui_components)
        
        _update_download_progress(ui_components, 50, "Mengunduh dataset...")
        _update_step_progress(ui_components, 20, "Memulai download dari Roboflow")
        
        # Setup progress monitoring melalui observer pattern yang sudah ada di UIDownloadService
        _setup_progress_monitoring(ui_components, download_service)
        
        # Jalankan download tanpa progress_callback parameter
        result = download_service.download_dataset(params)
        
        # Update final progress berdasarkan hasil
        if result.get('status') == 'success':
            _update_download_progress(ui_components, 100, "Download selesai!")
            _update_step_progress(ui_components, 100, "Download completed")
        
        return result
        
    except Exception as e:
        return {'status': 'error', 'message': f'Download service error: {str(e)}'}

def _setup_progress_monitoring(ui_components: Dict[str, Any], download_service) -> None:
    """Setup progress monitoring melalui observer pattern atau direct callback."""
    try:
        # Cek apakah ada progress_bridge di download_service
        if hasattr(download_service, 'progress_bridge') and download_service.progress_bridge:
            bridge = download_service.progress_bridge
            
            # Override progress_bridge notifications untuk update UI
            original_notify_step_progress = bridge.notify_step_progress
            original_notify_step_start = bridge.notify_step_start
            original_notify_step_complete = bridge.notify_step_complete
            
            def enhanced_step_progress(progress: int, message: str):
                # Call original method
                original_notify_step_progress(progress, message)
                
                # Update UI progress berdasarkan current step
                current_step = getattr(bridge, '_current_step', 'unknown')
                
                if current_step == 'download':
                    # Overall progress: 50-80% untuk download
                    base_progress = 50 + int((progress / 100) * 30)
                    _update_download_progress(ui_components, base_progress, f"Download: {message}")
                    _update_step_progress(ui_components, progress, f"📥 {message}")
                    
                elif current_step == 'organize':
                    # Overall progress: 80-95% untuk organize
                    base_progress = 80 + int((progress / 100) * 15)
                    _update_download_progress(ui_components, base_progress, f"Organisir: {message}")
                    _update_step_progress(ui_components, progress, f"📁 {message}")
                    
                elif current_step in ['validate', 'metadata']:
                    # Early stages: 50% base progress
                    base_progress = 30 + int((progress / 100) * 20)
                    _update_download_progress(ui_components, base_progress, message)
                    _update_step_progress(ui_components, progress, message)
                    
                elif current_step == 'verify':
                    # Final verification: 95-100%
                    base_progress = 95 + int((progress / 100) * 5)
                    _update_download_progress(ui_components, base_progress, f"Verifikasi: {message}")
                    _update_step_progress(ui_components, progress, f"✅ {message}")
            
            def enhanced_step_start(step_name: str, message: str):
                # Store current step untuk progress mapping
                bridge._current_step = step_name
                
                # Call original method
                original_notify_step_start(step_name, message)
                
                # Update UI berdasarkan step yang dimulai
                if step_name == 'validate':
                    _update_download_progress(ui_components, 30, message)
                    _update_step_progress(ui_components, 0, "🔍 Validasi parameter")
                elif step_name == 'metadata':
                    _update_download_progress(ui_components, 35, message)
                    _update_step_progress(ui_components, 0, "📊 Mengambil metadata")
                elif step_name == 'download':
                    _update_download_progress(ui_components, 50, message)
                    _update_step_progress(ui_components, 0, "📥 Memulai download")
                elif step_name == 'organize':
                    _update_download_progress(ui_components, 80, message)
                    _update_step_progress(ui_components, 0, "📁 Mengorganisir dataset")
                elif step_name == 'verify':
                    _update_download_progress(ui_components, 95, message)
                    _update_step_progress(ui_components, 0, "✅ Verifikasi hasil")
            
            def enhanced_step_complete(message: str):
                # Call original method
                original_notify_step_complete(message)
                
                # Update step progress ke 100% untuk step yang selesai
                current_step = getattr(bridge, '_current_step', 'unknown')
                _update_step_progress(ui_components, 100, f"✅ {message}")
            
            # Replace methods dengan enhanced versions
            bridge.notify_step_progress = enhanced_step_progress
            bridge.notify_step_start = enhanced_step_start
            bridge.notify_step_complete = enhanced_step_complete
            
    except Exception as e:
        print(f"Warning: Could not setup progress monitoring: {e}")
        # Fallback: Update progress secara manual tanpa monitoring detail

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass