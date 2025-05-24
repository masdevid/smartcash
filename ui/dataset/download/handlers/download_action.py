"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Fixed download action dengan proper button state management dan dual progress tracking
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
    """Eksekusi download dengan proper state management dan dual progress tracking."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Use context manager untuk download operation
    with button_manager.operation_context('download'):
        try:
            if logger:
                logger.info("🚀 Memulai proses download dan organisasi dataset")
            
            _clear_ui_outputs(ui_components)
            
            if logger:
                logger.info("📋 Memvalidasi parameter download...")
            
            # Validation
            validation_result = _robust_validate_params(ui_components, logger)
            if not validation_result['valid']:
                if logger:
                    logger.error(f"❌ Validasi gagal: {validation_result['message']}")
                raise Exception(validation_result['message'])
            
            params = validation_result['params']
            existing_check = _safe_check_existing_dataset(ui_components)
            
            if existing_check['exists']:
                _show_organized_dataset_confirmation(ui_components, params, existing_check, button_manager)
            else:
                _execute_download_confirmed(ui_components, params)
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error persiapan download: {str(e)}")
                logger.debug(f"🔧 Full traceback: {traceback.format_exc()}")
            raise

def _robust_validate_params(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Robust parameter validation dengan detailed error tracking."""
    try:
        params = {}
        required_fields = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        
        for field in required_fields:
            try:
                value = _ultra_safe_get_value(ui_components, field, logger)
                params[field] = value
            except Exception as e:
                params[field] = ''
        
        # Get API key dari sources jika kosong
        if not params['api_key']:
            try:
                params['api_key'] = _get_api_key_from_sources()
            except Exception:
                pass
        
        # Validasi field wajib
        missing_fields = [field for field in required_fields if not params[field]]
        
        if missing_fields:
            return {
                'valid': False,
                'message': f"Parameter tidak lengkap: {', '.join(missing_fields)}",
                'params': params
            }
        
        # Validasi output directory
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
            'traceback': traceback.format_exc()
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
    """Safe check existing dataset dengan error handling."""
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
    """Show confirmation untuk overwrite existing organized dataset."""
    env_manager = ui_components.get('env_manager')
    storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})" if env_manager and env_manager.is_drive_mounted else "📁 Storage: Local (akan hilang saat restart)"
    
    split_info = [f"• {split}: {stats['images']} gambar" for split, stats in existing_info['splits'].items()]
    
    message = (
        f"⚠️ Dataset sudah ada di struktur final!\n\n"
        f"📊 Dataset yang ada:\n" + '\n'.join(split_info) + f"\n• Total: {existing_info['total_images']} gambar\n"
        f"• {storage_info}\n\n"
        f"📥 Dataset baru:\n• Workspace: {params['workspace']}\n• Project: {params['project']}\n• Version: {params['version']}\n\n"
        f"⚠️ Dataset yang ada akan diganti dengan yang baru.\nLanjutkan download?\n\n"
        f"💡 Tips: Gunakan 'Cleanup Dataset' terlebih dahulu jika ingin backup."
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        # Continue dengan context manager yang sudah ada
        try:
            _execute_download_confirmed(ui_components, params)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Download error: {str(e)}")
            # Re-raise untuk ditangkap context manager
            raise
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        logger = ui_components.get('logger')
        if logger:
            logger.info("❌ Download dibatalkan - dataset yang ada tidak akan diganti")
        # Raise exception untuk exit dari context manager
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
    """Execute download dengan error handling."""
    logger = ui_components.get('logger')
    
    if logger:
        logger.info("✅ Parameter valid - memulai download dan organisasi:")
        for key, value in params.items():
            if key != 'api_key':
                logger.info(f"   • {key}: {value}")
        logger.info("🚀 Memulai download dengan organisasi otomatis...")
    
    result = _execute_enhanced_download(ui_components, params)
    
    if result.get('status') == 'success':
        if logger:
            stats = result.get('stats', {})
            duration = result.get('duration', 0)
            storage_type = "Drive" if result.get('drive_storage', False) else "Local"
            
            logger.success(f"✅ Download dan organisasi berhasil ({duration:.1f}s)")
            logger.info(f"📁 Storage: {storage_type}")
            logger.info(f"📊 Total gambar: {stats.get('total_images', 0)}")
            
            if 'splits' in stats:
                logger.info("📁 Struktur dataset:")
                for split, split_stats in stats['splits'].items():
                    if split_stats.get('images', 0) > 0:
                        logger.info(f"   • {split}: {split_stats['images']} gambar")
    else:
        error_msg = result.get('message', 'Unknown error')
        if logger:
            logger.error(f"❌ Download gagal: {error_msg}")
        raise Exception(error_msg)

def _execute_enhanced_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute download dengan enhanced service."""
    try:
        from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
        download_service = UIDownloadService(ui_components)
        return download_service.download_dataset(params)
    except Exception as e:
        return {'status': 'error', 'message': f'Enhanced download service error: {str(e)}'}

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear semua UI output sebelum mulai download."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass