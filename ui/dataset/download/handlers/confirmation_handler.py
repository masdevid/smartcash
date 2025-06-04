"""
File: smartcash/ui/dataset/download/handlers/confirmation_handler.py
Deskripsi: Handler untuk konfirmasi download dengan existing dataset detection
"""

from typing import Dict, Any, Callable
from pathlib import Path
from IPython.display import display
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

def handle_download_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], 
                                execute_callback: Callable) -> None:
    """
    Handle konfirmasi download dengan check existing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter download yang sudah divalidasi
        execute_callback: Callback untuk execute download setelah konfirmasi
    """
    logger = ui_components.get('logger')
    
    try:
        # Check existing dataset
        existing_check = _check_existing_dataset(ui_components)
        
        if existing_check['exists']:
            # Show replacement confirmation
            _show_replacement_confirmation(ui_components, params, existing_check, execute_callback)
        else:
            # Show standard download confirmation
            _show_standard_confirmation(ui_components, params, execute_callback)
            
    except Exception as e:
        logger and logger.error(f"❌ Error konfirmasi: {str(e)}")
        raise

def _check_existing_dataset(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check apakah dataset sudah ada di struktur final."""
    try:
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab,
            is_drive_mounted=env_manager.is_drive_mounted
        )
        
        result = {
            'exists': False,
            'total_images': 0,
            'total_labels': 0,
            'splits': {},
            'paths': paths,
            'storage_type': 'Google Drive' if env_manager.is_drive_mounted else 'Local Storage'
        }
        
        # Check setiap split
        for split in ['train', 'valid', 'test']:
            split_info = _check_split_directory(paths[split])
            result['splits'][split] = split_info
            result['total_images'] += split_info['images']
            result['total_labels'] += split_info['labels']
        
        result['exists'] = result['total_images'] > 0
        return result
        
    except Exception:
        return {'exists': False, 'total_images': 0, 'splits': {}}

def _check_split_directory(split_path_str: str) -> Dict[str, Any]:
    """Check directory split dan hitung file."""
    split_info = {
        'exists': False,
        'images': 0,
        'labels': 0,
        'path': split_path_str,
        'images_path': '',
        'labels_path': ''
    }
    
    try:
        split_path = Path(split_path_str)
        if not split_path.exists():
            return split_info
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        split_info['images_path'] = str(images_dir)
        split_info['labels_path'] = str(labels_dir)
        
        if images_dir.exists():
            img_files = list(images_dir.glob('*.*'))
            split_info['images'] = len(img_files)
            split_info['exists'] = len(img_files) > 0
        
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            split_info['labels'] = len(label_files)
            
    except Exception:
        pass
    
    return split_info

def _show_replacement_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], 
                                 existing_info: Dict[str, Any], execute_callback: Callable) -> None:
    """Show confirmation untuk replace existing dataset."""
    
    # Build split info string
    split_info_lines = []
    for split, stats in existing_info['splits'].items():
        if stats['exists'] and stats['images'] > 0:
            split_info_lines.append(f"• {split}: {stats['images']} gambar, {stats['labels']} label")
    
    env_manager = ui_components.get('env_manager')
    storage_info = f"📁 Storage: {existing_info['storage_type']}"
    if env_manager and env_manager.is_drive_mounted:
        storage_info += f" ({env_manager.drive_path})"
    
    message = (
        f"⚠️ Dataset sudah ada di lokasi target!\n\n"
        f"📊 Dataset yang ada:\n" + '\n'.join(split_info_lines) + 
        f"\n• Total: {existing_info['total_images']} gambar, {existing_info['total_labels']} label\n"
        f"• {storage_info}\n\n"
        f"📥 Dataset baru:\n"
        f"• Workspace: {params['workspace']}\n"
        f"• Project: {params['project']}\n"
        f"• Version: {params['version']}\n\n"
        f"🔄 Dataset yang ada akan diganti dengan yang baru.\n"
        f"Lanjutkan download?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        try:
            execute_callback(ui_components, params)
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Error execute download: {str(e)}")
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        logger = ui_components.get('logger')
        logger and logger.info("❌ Download dibatalkan oleh user")
        
        # Reset progress
        if 'error_operation' in ui_components:
            ui_components['error_operation']("Download dibatalkan")
    
    dialog = create_confirmation_dialog(
        title="⚠️ Konfirmasi Replace Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Replace Dataset",
        cancel_text="Batal",
        danger_mode=True
    )
    
    # Show dialog
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _show_standard_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], 
                               execute_callback: Callable) -> None:
    """Show standard download confirmation."""
    
    env_manager = ui_components.get('env_manager')
    if env_manager and env_manager.is_drive_mounted:
        storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})"
        storage_note = "💾 Dataset akan tersimpan permanen di Google Drive"
    else:
        storage_info = "📁 Storage: Local Storage"
        storage_note = "⚠️ Dataset akan hilang saat runtime restart (gunakan Drive untuk penyimpanan permanen)"
    
    message = (
        f"📥 Konfirmasi Download Dataset\n\n"
        f"🎯 Dataset yang akan didownload:\n"
        f"• Workspace: {params['workspace']}\n"
        f"• Project: {params['project']}\n"
        f"• Version: {params['version']}\n"
        f"• Output: {params['output_dir']}\n\n"
        f"💾 Storage Info:\n"
        f"• {storage_info}\n"
        f"• {storage_note}\n\n"
        f"🚀 Proses akan:\n"
        f"1. Download dataset dari Roboflow\n"
        f"2. Ekstrak dan organisir ke struktur final\n"
        f"3. Validasi hasil download\n\n"
        f"Lanjutkan download?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        try:
            execute_callback(ui_components, params)
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Error execute download: {str(e)}")
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        logger = ui_components.get('logger')
        logger and logger.info("❌ Download dibatalkan oleh user")
        
        # Reset progress
        if 'error_operation' in ui_components:
            ui_components['error_operation']("Download dibatalkan")
    
    dialog = create_confirmation_dialog(
        title="📥 Konfirmasi Download Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="Ya, Download Dataset",
        cancel_text="Batal"
    )
    
    # Show dialog
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)