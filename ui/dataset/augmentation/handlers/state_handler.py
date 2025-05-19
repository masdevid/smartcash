"""
File: smartcash/ui/dataset/augmentation/handlers/state_handler.py
Deskripsi: Handler state untuk mendeteksi dan mengelola state augmentasi dataset
"""

from typing import Dict, Any
from pathlib import Path
import os
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def detect_augmentation_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deteksi state augmentasi dan update UI sesuai keadaan dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Get paths dari UI components
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Cek apakah augmented_dir sudah ada
        augmented_path = Path(augmented_dir)
        if not augmented_path.exists():
            if logger: logger.info(f"ℹ️ Direktori augmentasi belum ada: {augmented_dir}")
            return ui_components
        
        # Cari file di split train (augmentasi biasanya hanya untuk training data)
        has_augmented_data = False
        image_count = 0
        stats = {'total': {'images': 0}, 'splits': {}}
        
        # Cek split train
        train_dir = augmented_path / 'train'
        has_split = train_dir.exists()
        
        images_dir = train_dir / 'images'
        
        # Count image files di split
        train_images = len(list(images_dir.glob('*.jpg'))) if has_split and images_dir.exists() else 0
        
        # Update statistik
        image_count += train_images
        stats['total']['images'] += train_images
        stats['splits']['train'] = {
            'exists': has_split,
            'images': train_images,
            'complete': train_images > 0
        }
        
        # Cek hasil augmentasi
        if train_images > 0:
            has_augmented_data = True
        
        # Update UI status jika augmentasi sudah dilakukan
        if has_augmented_data:
            # Tampilkan status informasi
            ui_components['update_status_panel'](
                ui_components, 
                "info", 
                f"{ICONS['info']} Ditemukan {image_count} gambar hasil augmentasi di {augmented_dir}"
            )
            
            # Tampilkan tombol-tombol terkait data
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
                
            if 'summary_container' in ui_components:
                ui_components['summary_container'].layout.display = 'block'
            
            if logger:
                logger.info(f"{ICONS['success']} Dataset augmentasi terdeteksi dengan {image_count} gambar")
        else:
            # Tampilkan status jika belum ada data
            ui_components['update_status_panel'](
                ui_components, 
                "warning", 
                f"{ICONS['warning']} Belum ada data hasil augmentasi di {augmented_dir}"
            )
            
            if logger:
                logger.info(f"{ICONS['info']} Belum ada dataset augmentasi")
    
    except Exception as e:
        # Handle error
        if logger:
            logger.error(f"{ICONS['error']} Error saat mendeteksi state augmentasi: {str(e)}")
        
        # Update status panel
        ui_components['update_status_panel'](
            ui_components, 
            "error", 
            f"{ICONS['error']} Error saat mendeteksi state augmentasi: {str(e)}"
        )
    
    return ui_components

def get_augmentation_stats(ui_components: Dict[str, Any], augmented_dir: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        augmented_dir: Direktori dataset augmented
        
    Returns:
        Dictionary statistik augmentasi
    """
    # Inisialisasi struktur statistik
    stats = {
        'splits': {},
        'total': {'images': 0, 'classes': {}}
    }
    
    # Cek split train (augmentasi biasanya hanya untuk training data)
    train_dir = Path(augmented_dir) / 'train'
    if not train_dir.exists():
        stats['splits']['train'] = {'exists': False, 'images': 0}
        return stats
            
    # Count images dengan one-liner dan support multi-format
    images_dir = train_dir / 'images'
    image_extensions = ['.jpg', '.png', '.jpeg']
    num_images = sum([len(list(images_dir.glob(f"*{ext}"))) for ext in image_extensions]) if images_dir.exists() else 0
    
    # Update statistik
    stats['splits']['train'] = {
        'exists': True,
        'images': num_images,
        'complete': num_images > 0
    }
    
    # Update total
    stats['total']['images'] += num_images
    
    # Dataset dianggap valid jika ada gambar hasil augmentasi
    stats['valid'] = num_images > 0
    
    return stats

def setup_state_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup state handler untuk augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan augmented_dir dari config atau default
    augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented') if config else 'data/augmented'
    ui_components['augmented_dir'] = augmented_dir
    
    # Tambahkan fungsi ke ui_components
    ui_components['detect_augmentation_state'] = detect_augmentation_state
    ui_components['get_augmentation_stats'] = get_augmentation_stats
    
    # Deteksi state augmentasi
    ui_components = detect_augmentation_state(ui_components)
    
    return ui_components
