"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Enhanced download handler dengan dataset check dan confirmation untuk prevent overwrite
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
from smartcash.ui.dataset.download.utils.download_executor import execute_roboflow_download
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from IPython.display import display

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi download dengan dataset check dan confirmation dialog."""
    logger = ui_components.get('logger')
    
    # 🚀 Start download process
    if logger:
        logger.info("🚀 Memulai proses download dataset")
    
    # Disable semua buttons untuk mencegah double click
    disable_download_buttons(ui_components, True)
    
    try:
        # 🧹 Clear outputs sebelum mulai
        _clear_ui_outputs(ui_components)
        
        # ✅ Validasi parameter
        if logger:
            logger.info("📋 Memvalidasi parameter download...")
        
        validation_result = validate_download_params(ui_components)
        if not validation_result['valid']:
            if logger:
                logger.error(f"❌ Validasi gagal: {validation_result['message']}")
            return
        
        params = validation_result['params']
        
        # 🔍 Check existing dataset
        existing_check = _check_existing_dataset(ui_components, params['output_dir'])
        
        if existing_check['exists']:
            # Show confirmation dialog untuk overwrite
            _show_overwrite_confirmation(ui_components, params, existing_check)
        else:
            # Langsung execute download
            _execute_download_confirmed(ui_components, params)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan download: {str(e)}")
        # Re-enable buttons jika error
        disable_download_buttons(ui_components, False)

def _check_existing_dataset(ui_components: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Check apakah dataset sudah ada di output directory."""
    output_path = Path(output_dir)
    logger = ui_components.get('logger')
    
    result = {
        'exists': False,
        'file_count': 0,
        'splits': [],
        'total_size_mb': 0
    }
    
    if not output_path.exists():
        return result
    
    try:
        # Check untuk split directories
        for split in ['train', 'valid', 'test']:
            split_dir = output_path / split
            if split_dir.exists():
                images_dir = split_dir / 'images'
                if images_dir.exists():
                    img_files = list(images_dir.glob('*.*'))
                    if img_files:
                        result['splits'].append(split)
                        result['file_count'] += len(img_files)
        
        # Calculate total size
        try:
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            result['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        except Exception:
            result['total_size_mb'] = 0
        
        # Dataset exists jika ada minimal 1 split dengan gambar
        result['exists'] = len(result['splits']) > 0
        
        if result['exists'] and logger:
            logger.warning(f"⚠️ Dataset sudah ada: {result['file_count']} gambar, {result['total_size_mb']} MB")
            logger.info(f"📁 Splits ditemukan: {', '.join(result['splits'])}")
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error checking existing dataset: {str(e)}")
    
    return result

def _show_overwrite_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], existing_info: Dict[str, Any]) -> None:
    """Show confirmation dialog untuk overwrite existing dataset."""
    
    # Determine storage type
    env_manager = ui_components.get('env_manager')
    if env_manager and env_manager.is_drive_mounted:
        storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})"
    else:
        storage_info = "📁 Storage: Local (akan hilang saat restart)"
    
    message = (
        f"⚠️ Dataset sudah ada di lokasi target!\n\n"
        f"📊 Dataset yang ada:\n"
        f"• File: {existing_info['file_count']} gambar\n"
        f"• Ukuran: {existing_info['total_size_mb']} MB\n"
        f"• Splits: {', '.join(existing_info['splits'])}\n"
        f"• Lokasi: {params['output_dir']}\n\n"
        f"📥 Dataset baru:\n"
        f"• Workspace: {params['workspace']}\n"
        f"• Project: {params['project']}\n"
        f"• Version: {params['version']}\n"
        f"• {storage_info}\n\n"
        f"⚠️ Dataset yang ada akan diganti. Lanjutkan download?\n\n"
        f"💡 Tips: Gunakan 'Cleanup Dataset' terlebih dahulu atau gunakan output directory berbeda."
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
            logger.info("💡 Gunakan 'Cleanup Dataset' untuk menghapus dataset lama atau ganti output directory")
    
    dialog = create_confirmation_dialog(
        title="⚠️ Konfirmasi Overwrite Dataset",
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
    """Execute download setelah confirmation."""
    logger = ui_components.get('logger')
    
    try:
        # 📊 Log parameter yang akan digunakan
        if logger:
            logger.info("✅ Parameter valid - memulai download:")
            logger.info(f"   • Workspace: {params['workspace']}")
            logger.info(f"   • Project: {params['project']}")
            logger.info(f"   • Version: {params['version']}")
            logger.info(f"   • Output: {params['output_dir']}")
            logger.info("🚀 Memulai download dataset...")
        
        # Execute download dengan enhanced progress tracking
        result = execute_roboflow_download(ui_components, params)
        
        # Handle hasil download
        if result.get('status') == 'success':
            if logger:
                stats = result.get('stats', {})
                duration = result.get('duration', 0)
                storage_type = "Drive" if result.get('drive_storage', False) else "Local"
                
                logger.success(f"✅ Download berhasil ({duration:.1f}s)")
                logger.info(f"📁 Storage: {storage_type}")
                logger.info(f"📊 Dataset: {stats.get('total_images', 0)} gambar")
                logger.info(f"📍 Lokasi: {result.get('output_dir', '')}")
                
                # Log breakdown per split
                for split in ['train', 'valid', 'test']:
                    split_key = f'{split}_images'
                    if split_key in stats and stats[split_key] > 0:
                        logger.info(f"   • {split}: {stats[split_key]} gambar")
        else:
            error_msg = result.get('message', 'Unknown error')
            if logger:
                logger.error(f"❌ Download gagal: {error_msg}")
                
    except Exception as e:
        if logger:
            logger.error(f"❌ Error download: {str(e)}")
    finally:
        # Re-enable buttons setelah selesai
        disable_download_buttons(ui_components, False)

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear semua UI output sebelum mulai download."""
    # Clear log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    # Clear confirmation area
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Reset progress indicators
    _reset_progress_indicators(ui_components)

def _reset_progress_indicators(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicators ke state awal."""
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'visible'
    
    # Reset labels
    label_widgets = ['overall_label', 'step_label']
    for label_key in label_widgets:
        if label_key in ui_components:
            ui_components[label_key].value = "Siap memulai"
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'visible'
    
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'