"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Fixed download handler dengan UI component validation dan error handling
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params, validate_ui_components_structure
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager
from IPython.display import display

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi download dengan comprehensive error handling."""
    logger = ui_components.get('logger')
    
    if logger:
        logger.info("🚀 Memulai proses download dan organisasi dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        _clear_ui_outputs(ui_components)
        
        # 🔍 Validate UI components structure terlebih dahulu
        structure_validation = validate_ui_components_structure(ui_components)
        if not structure_validation['valid']:
            error_msg = f"UI components tidak lengkap: {'; '.join(structure_validation['issues'])}"
            if logger:
                logger.error(f"❌ {error_msg}")
                logger.debug("💡 Coba refresh cell dan inisialisasi ulang UI")
            return
        
        if logger:
            logger.info("📋 Memvalidasi parameter download...")
        
        # 🔍 Validate download parameters
        validation_result = validate_download_params(ui_components)
        if not validation_result['valid']:
            if logger:
                logger.error(f"❌ Validasi gagal: {validation_result['message']}")
                _suggest_parameter_fixes(logger, validation_result)
            return
        
        params = validation_result['params']
        existing_check = _check_existing_organized_dataset(ui_components)
        
        if existing_check['exists']:
            _show_organized_dataset_confirmation(ui_components, params, existing_check)
        else:
            _execute_download_confirmed(ui_components, params)
        
    except Exception as e:
        error_msg = f"Error persiapan download: {str(e)}"
        if logger:
            logger.error(f"❌ {error_msg}")
            logger.debug("🔧 Debug info: Pastikan semua field UI telah diinisialisasi dengan benar")
        
        # Try to provide helpful debugging info
        _debug_ui_components(ui_components, logger)
        
    finally:
        disable_download_buttons(ui_components, False)

def _suggest_parameter_fixes(logger, validation_result: Dict[str, Any]) -> None:
    """Berikan saran perbaikan untuk parameter yang gagal validasi."""
    params = validation_result.get('params', {})
    
    logger.info("💡 Saran perbaikan:")
    
    if not params.get('workspace'):
        logger.info("   • Workspace: Masukkan workspace ID Roboflow (contoh: 'smartcash-wo2us')")
    
    if not params.get('project'):
        logger.info("   • Project: Masukkan project ID Roboflow (contoh: 'rupiah-emisi-2022')")
    
    if not params.get('version'):
        logger.info("   • Version: Masukkan version dataset (contoh: '3')")
    
    if not params.get('api_key'):
        logger.info("   • API Key: Masukkan API key atau set environment variable ROBOFLOW_API_KEY")
        try:
            from google.colab import userdata
            logger.info("   • Atau gunakan Colab Secrets untuk menyimpan ROBOFLOW_API_KEY")
        except ImportError:
            pass

def _debug_ui_components(ui_components: Dict[str, Any], logger) -> None:
    """Debug info untuk UI components."""
    if not logger:
        return
        
    logger.debug("🔧 Debug UI Components:")
    
    required_keys = ['workspace', 'project', 'version', 'api_key', 'output_dir']
    for key in required_keys:
        if key not in ui_components:
            logger.debug(f"   • {key}: MISSING")
        elif ui_components[key] is None:
            logger.debug(f"   • {key}: NULL")
        elif not hasattr(ui_components[key], 'value'):
            logger.debug(f"   • {key}: NO VALUE ATTRIBUTE")
        else:
            try:
                value = ui_components[key].value
                display_value = "***" if key == 'api_key' and value else str(value)[:20]
                logger.debug(f"   • {key}: '{display_value}' (OK)")
            except Exception as e:
                logger.debug(f"   • {key}: ERROR accessing value - {str(e)}")

def _check_existing_organized_dataset(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing organized dataset dengan error handling."""
    logger = ui_components.get('logger')
    
    try:
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab,
            is_drive_mounted=env_manager.is_drive_mounted
        )
        
        result = {'exists': False, 'total_images': 0, 'splits': {}, 'paths': paths}
        
        for split in ['train', 'valid', 'test']:
            split_path = Path(paths[split])
            if split_path.exists():
                images_dir = split_path / 'images'
                if images_dir.exists():
                    img_files = list(images_dir.glob('*.*'))
                    if img_files:
                        result['splits'][split] = {'images': len(img_files), 'path': str(split_path)}
                        result['total_images'] += len(img_files)
        
        result['exists'] = result['total_images'] > 0
        
        if result['exists'] and logger:
            logger.warning(f"⚠️ Dataset sudah ada: {result['total_images']} gambar")
            for split, stats in result['splits'].items():
                logger.info(f"   • {split}: {stats['images']} gambar di {stats['path']}")
        
        return result
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error checking existing dataset: {str(e)}")
        return {'exists': False, 'total_images': 0, 'splits': {}}

def _show_organized_dataset_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], existing_info: Dict[str, Any]) -> None:
    """Show confirmation untuk overwrite existing organized dataset."""
    
    env_manager = ui_components.get('env_manager')
    if env_manager and env_manager.is_drive_mounted:
        storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})"
    else:
        storage_info = "📁 Storage: Local (akan hilang saat restart)"
    
    split_info = []
    for split, stats in existing_info['splits'].items():
        split_info.append(f"• {split}: {stats['images']} gambar")
    
    message = (
        f"⚠️ Dataset sudah ada di struktur final!\n\n"
        f"📊 Dataset yang ada:\n"
        f"{''.join([info + chr(10) for info in split_info])}"
        f"• Total: {existing_info['total_images']} gambar\n"
        f"• {storage_info}\n\n"
        f"📥 Dataset baru:\n"
        f"• Workspace: {params['workspace']}\n"
        f"• Project: {params['project']}\n"
        f"• Version: {params['version']}\n\n"
        f"⚠️ Dataset yang ada akan diganti dengan yang baru.\n"
        f"Lanjutkan download?\n\n"
        f"💡 Tips: Gunakan 'Cleanup Dataset' terlebih dahulu jika ingin backup."
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        _execute_download_confirmed(ui_components, params)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        disable_download_buttons(ui_components, False)
        logger = ui_components.get('logger')
        if logger:
            logger.info("❌ Download dibatalkan - dataset yang ada tidak akan diganti")
            logger.info("💡 Gunakan 'Cleanup Dataset' untuk menghapus dataset lama")
    
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
    """Execute download dengan enhanced error handling."""
    logger = ui_components.get('logger')
    
    try:
        if logger:
            logger.info("✅ Parameter valid - memulai download dan organisasi:")
            logger.info(f"   • Workspace: {params['workspace']}")
            logger.info(f"   • Project: {params['project']}")
            logger.info(f"   • Version: {params['version']}")
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
                
    except Exception as e:
        if logger:
            logger.error(f"❌ Error download: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)

def _execute_enhanced_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute download dengan enhanced service."""
    try:
        from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
        
        download_service = UIDownloadService(ui_components)
        result = download_service.download_dataset(params)
        
        return result
        
    except Exception as e:
        return {'status': 'error', 'message': f'Enhanced download service error: {str(e)}'}

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear semua UI output sebelum mulai download."""
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    _reset_progress_indicators(ui_components)

def _reset_progress_indicators(ui_components: Dict[str, Any]) -> None:
    """Reset dual progress indicators ke state awal."""
    # Reset both progress bars
    for widget_key in ['progress_bar', 'current_progress']:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            if widget_key == 'progress_bar':
                ui_components[widget_key].description = "Overall: 0%"
            else:
                ui_components[widget_key].description = "Step: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'visible'
    
    # Reset labels
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = "Siap memulai"
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'visible'
    
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'