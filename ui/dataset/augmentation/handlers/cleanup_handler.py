"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler pembersihan untuk augmentasi dataset
"""

import os
import shutil
from typing import Dict, Any, List, Optional
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
            
            logger.info(f"✅ Backup {num_images} gambar dan {num_labels} label ke {backup_dir}")
        
        # Hapus file di direktori output
        if os.path.exists(images_output_dir):
            for file in os.listdir(images_output_dir):
                file_path = os.path.join(images_output_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        if os.path.exists(labels_output_dir):
            for file in os.listdir(labels_output_dir):
                file_path = os.path.join(labels_output_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
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
        
        # Hitung jumlah file yang akan dihapus
        num_images = 0
        num_labels = 0
        
        # Hapus gambar yang dimulai dengan prefix
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                if file.startswith(prefix):
                    file_path = os.path.join(images_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        num_images += 1
        
        # Hapus label yang dimulai dengan prefix
        if os.path.exists(labels_dir):
            for file in os.listdir(labels_dir):
                if file.startswith(prefix):
                    file_path = os.path.join(labels_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        num_labels += 1
        
        # Jika tidak ada file yang dihapus
        if num_images == 0 and num_labels == 0:
            return {
                'status': 'warning',
                'message': f'Tidak ada file dengan prefix "{prefix}" di direktori preprocessed'
            }
        
        # Log info
        logger.info(f"✅ Berhasil menghapus {num_images} gambar dan {num_labels} label dengan prefix '{prefix}' dari {final_output_dir}")
        
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
