"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler pembersihan untuk augmentasi dataset
"""

import os
import glob
import shutil
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

def cleanup_augmentation_results(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bersihkan hasil augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil pembersihan dengan status dan pesan
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan path dari ui_components
        paths = ui_components.get('augmentation_paths', {})
        
        if not paths:
            # Inisialisasi direktori jika belum ada
            from smartcash.ui.dataset.augmentation.handlers.initialization_handler import initialize_augmentation_directories
            init_result = initialize_augmentation_directories(ui_components)
            
            if init_result['status'] == 'error':
                return {
                    'status': 'error',
                    'message': init_result['message']
                }
            
            paths = init_result['paths']
        
        # Dapatkan direktori output
        output_dir = paths.get('output_dir')
        images_output_dir = paths.get('images_output_dir')
        labels_output_dir = paths.get('labels_output_dir')
        
        # Dapatkan konfigurasi backup
        backup_enabled = paths.get('backup_enabled', True)
        backup_dir = paths.get('backup_dir')
        
        # Cek apakah direktori output ada
        if not os.path.exists(output_dir):
            return {
                'status': 'warning',
                'message': f'Direktori output tidak ditemukan: {output_dir}'
            }
        
        # Hitung jumlah file yang akan dihapus
        num_images = len(os.listdir(images_output_dir)) if os.path.exists(images_output_dir) else 0
        num_labels = len(os.listdir(labels_output_dir)) if os.path.exists(labels_output_dir) else 0
        
        # Jika tidak ada file, tidak perlu membersihkan
        if num_images == 0 and num_labels == 0:
            return {
                'status': 'warning',
                'message': 'Tidak ada file yang perlu dibersihkan'
            }
        
        # Backup file jika diperlukan
        if backup_enabled and backup_dir:
            # Buat direktori backup jika belum ada
            os.makedirs(os.path.join(backup_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(backup_dir, 'labels'), exist_ok=True)
            
            # Backup gambar
            if os.path.exists(images_output_dir):
                for file in os.listdir(images_output_dir):
                    src = os.path.join(images_output_dir, file)
                    dst = os.path.join(backup_dir, 'images', file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
            
            # Backup label
            if os.path.exists(labels_output_dir):
                for file in os.listdir(labels_output_dir):
                    src = os.path.join(labels_output_dir, file)
                    dst = os.path.join(backup_dir, 'labels', file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
            
            # Log backup dinonaktifkan untuk mengurangi output log
            pass
        
        # Hapus file di direktori output dengan progress bar
        total_files = 0
        files_to_remove = []
        
        # Kumpulkan semua file yang akan dihapus
        if os.path.exists(images_output_dir):
            image_files = [f for f in glob.glob(os.path.join(images_output_dir, '*')) if os.path.isfile(f)]
            files_to_remove.extend(image_files)
            total_files += len(image_files)
        
        if os.path.exists(labels_output_dir):
            label_files = [f for f in glob.glob(os.path.join(labels_output_dir, '*')) if os.path.isfile(f)]
            files_to_remove.extend(label_files)
            total_files += len(label_files)
        
        # Hapus file dengan progress bar
        if files_to_remove:
            with tqdm(total=total_files, desc="🗑️ Menghapus file sementara", unit="file", colour="red") as pbar:
                for f in files_to_remove:
                    if os.path.isfile(f):
                        os.remove(f)
                        pbar.update(1)
        
        # Log info
        logger.info(f"✅ Berhasil menghapus {num_images} gambar dan {num_labels} label dari {output_dir}")
        
        return {
            'status': 'success',
            'message': f'Berhasil menghapus {num_images} gambar dan {num_labels} label' + 
                      (f' (backup di {backup_dir})' if backup_enabled else ''),
            'num_images': num_images,
            'num_labels': num_labels,
            'backup_dir': backup_dir if backup_enabled else None
        }
    except Exception as e:
        logger.error(f"❌ Error saat membersihkan hasil augmentasi: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat membersihkan hasil augmentasi: {str(e)}'
        }

def cleanup_augmented_files(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bersihkan file hasil augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil pembersihan dengan status dan pesan
    """
    # Panggil fungsi cleanup_augmentation_results untuk kompatibilitas dengan pengujian
    return cleanup_augmentation_results(ui_components)

def remove_augmentation_results(ui_components: Dict[str, Any], prefix: str = 'aug') -> Dict[str, Any]:
    """
    Hapus hasil augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        prefix: Prefix file augmentasi
        
    Returns:
        Dictionary hasil penghapusan dengan status dan pesan
    """
    # Panggil fungsi remove_augmented_files_from_preprocessed untuk kompatibilitas dengan pengujian
    return remove_augmented_files_from_preprocessed(ui_components, prefix)

def remove_augmented_files_from_preprocessed(ui_components: Dict[str, Any], prefix: str = 'aug') -> Dict[str, Any]:
    """
    Hapus file hasil augmentasi dari direktori preprocessed.
    
    Args:
        ui_components: Dictionary komponen UI
        prefix: Prefix file augmentasi
        
    Returns:
        Dictionary hasil penghapusan dengan status dan pesan
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan path dari ui_components
        paths = ui_components.get('augmentation_paths', {})
        
        if not paths:
            # Inisialisasi direktori jika belum ada
            from smartcash.ui.dataset.augmentation.handlers.initialization_handler import initialize_augmentation_directories
            init_result = initialize_augmentation_directories(ui_components)
            
            if init_result['status'] == 'error':
                return {
                    'status': 'error',
                    'message': init_result['message']
                }
            
            paths = init_result['paths']
        
        # Dapatkan direktori preprocessed
        final_output_dir = paths.get('final_output_dir')
        split = paths.get('split', 'train')
        
        # Cek apakah direktori preprocessed ada
        if not os.path.exists(final_output_dir):
            return {
                'status': 'warning',
                'message': f'Direktori preprocessed tidak ditemukan: {final_output_dir}'
            }
        
        # Direktori gambar dan label
        images_dir = os.path.join(final_output_dir, 'images')
        labels_dir = os.path.join(final_output_dir, 'labels')
        
        # Cari file dengan prefix dan hapus dengan progress bar
        num_images = 0
        num_labels = 0
        files_to_remove = []
        
        # Cari gambar untuk dihapus
        images_dir = os.path.join(final_output_dir, 'images')
        if os.path.exists(images_dir):
            image_files = [img_file for img_file in glob.glob(os.path.join(images_dir, f"{prefix}*.*")) 
                         if os.path.isfile(img_file) and any(img_file.lower().endswith(ext) 
                                                              for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
            files_to_remove.extend(image_files)
            num_images = len(image_files)
        
        # Cari label untuk dihapus
        labels_dir = os.path.join(final_output_dir, 'labels')
        if os.path.exists(labels_dir):
            label_files = [label_file for label_file in glob.glob(os.path.join(labels_dir, f"{prefix}*.txt")) 
                          if os.path.isfile(label_file)]
            files_to_remove.extend(label_files)
            num_labels = len(label_files)
        
        # Hapus file dengan progress bar
        if files_to_remove:
            with tqdm(total=len(files_to_remove), desc=f"🗑️ Menghapus file dengan prefix '{prefix}'", 
                     unit="file", colour="red") as pbar:
                for f in files_to_remove:
                    if os.path.isfile(f):
                        os.remove(f)
                        pbar.update(1)
        
        # Jika tidak ada file yang dihapus
        if num_images == 0 and num_labels == 0:
            return {
                'status': 'warning',
                'message': f'Tidak ada file dengan prefix "{prefix}" di direktori preprocessed'
            }
        
        # Log penghapusan dinonaktifkan untuk mengurangi output log
        pass
        
        return {
            'status': 'success',
            'message': f'Berhasil menghapus {num_images} gambar dan {num_labels} label dengan prefix "{prefix}" dari direktori preprocessed',
            'num_images': num_images,
            'num_labels': num_labels
        }
    except Exception as e:
        logger.error(f"❌ Error saat menghapus file augmentasi dari preprocessed: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat menghapus file augmentasi dari preprocessed: {str(e)}'
        }

def setup_cleanup_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk pembersihan hasil augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI dengan handler terpasang
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    logger.info("🧹 Setup cleanup handler untuk augmentasi...")
    
    try:
        # Daftarkan handler untuk tombol cleanup
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].on_click(
                lambda b: on_cleanup_augmentation(b, ui_components)
            )
            
        return ui_components
    except Exception as e:
        logger.error(f"❌ Error saat setup cleanup handler: {str(e)}")
        return ui_components

def on_cleanup_augmentation(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol cleanup augmentasi.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    logger.info("🧹 Memulai pembersihan hasil augmentasi...")
    
    # Tampilkan status
    status_output = ui_components.get('status_panel', None)
    if status_output:
        with status_output:
            status_output.clear_output()
            display(create_status_indicator(
                "🧹 Membersihkan hasil augmentasi...", 
                "info"
            ))
    
    # Jalankan pembersihan
    result = cleanup_augmentation_results(ui_components)
    
    # Tampilkan hasil
    if status_output:
        with status_output:
            status_output.clear_output()
            if result['status'] == 'success':
                display(create_status_indicator(
                    f"✅ {result['message']}", 
                    "success"
                ))
            elif result['status'] == 'warning':
                display(create_status_indicator(
                    f"⚠️ {result['message']}", 
                    "warning"
                ))
            else:
                display(create_status_indicator(
                    f"❌ {result['message']}", 
                    "error"
                ))
    
    logger.info(f"🧹 Pembersihan selesai dengan status: {result['status']}")

