"""
File: smartcash/ui/dataset/visualization/handlers/dashboard_handler.py
Deskripsi: Handler untuk dashboard visualisasi dataset
"""

import os
from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
from tqdm.notebook import tqdm

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_split_cards, create_preprocessing_cards, create_augmentation_cards
)

logger = get_logger(__name__)


def get_dataset_stats() -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset dari direktori dataset.
    
    Returns:
        Dictionary berisi statistik dataset
    """
    try:
        # Dapatkan path dataset
        config_manager = get_config_manager()
        dataset_path = config_manager.get('dataset_path', None)
        
        if not dataset_path or not os.path.exists(dataset_path):
            logger.error(f"{ICONS.get('error', '❌')} Path dataset tidak valid: {dataset_path}")
            return {
                'split_stats': {},
                'preprocessing_stats': {},
                'augmentation_stats': {}
            }
        
        # Inisialisasi statistik
        split_stats = {
            'train': {'images': 0, 'labels': 0, 'objects': 0},
            'val': {'images': 0, 'labels': 0, 'objects': 0},
            'test': {'images': 0, 'labels': 0, 'objects': 0}
        }
        
        # Path untuk images dan labels
        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')
        
        # Cek apakah direktori ada
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Direktori images atau labels tidak ditemukan")
            return {
                'split_stats': split_stats,
                'preprocessing_stats': {},
                'augmentation_stats': {}
            }
        
        # Hitung jumlah gambar dan label per split
        for split in ['train', 'val', 'test']:
            # Hitung jumlah gambar
            split_images_path = os.path.join(images_path, split)
            if os.path.exists(split_images_path):
                image_files = [f for f in os.listdir(split_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                split_stats[split]['images'] = len(image_files)
            
            # Hitung jumlah label dan objek
            split_labels_path = os.path.join(labels_path, split)
            if os.path.exists(split_labels_path):
                label_files = [f for f in os.listdir(split_labels_path) if f.endswith('.txt')]
                split_stats[split]['labels'] = len(label_files)
                
                # Hitung jumlah objek
                for label_file in label_files:
                    with open(os.path.join(split_labels_path, label_file), 'r') as f:
                        split_stats[split]['objects'] += len(f.readlines())
        
        # Dapatkan statistik preprocessing
        preprocessing_stats = get_preprocessing_stats(dataset_path)
        
        # Dapatkan statistik augmentasi
        augmentation_stats = get_augmentation_stats(dataset_path)
        
        return {
            'split_stats': split_stats,
            'preprocessing_stats': preprocessing_stats,
            'augmentation_stats': augmentation_stats
        }
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mendapatkan statistik dataset: {str(e)}")
        return {
            'split_stats': {},
            'preprocessing_stats': {},
            'augmentation_stats': {}
        }


def get_preprocessing_stats(dataset_path: str) -> Dict[str, int]:
    """
    Mendapatkan statistik preprocessing dari direktori dataset.
    
    Args:
        dataset_path: Path ke direktori dataset
        
    Returns:
        Dictionary berisi statistik preprocessing
    """
    try:
        # Inisialisasi statistik preprocessing
        preprocessing_stats = {
            'processed_images': 0,
            'filtered_images': 0,
            'normalized_images': 0
        }
        
        # Path untuk metadata preprocessing
        preprocessing_meta_path = os.path.join(dataset_path, 'metadata', 'preprocessing')
        
        # Cek apakah direktori metadata ada
        if not os.path.exists(preprocessing_meta_path):
            return preprocessing_stats
        
        # Cek file metadata preprocessing
        processed_file = os.path.join(preprocessing_meta_path, 'processed_images.txt')
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                preprocessing_stats['processed_images'] = len(f.readlines())
        
        # Cek file metadata filtered
        filtered_file = os.path.join(preprocessing_meta_path, 'filtered_images.txt')
        if os.path.exists(filtered_file):
            with open(filtered_file, 'r') as f:
                preprocessing_stats['filtered_images'] = len(f.readlines())
        
        # Cek file metadata normalized
        normalized_file = os.path.join(preprocessing_meta_path, 'normalized_images.txt')
        if os.path.exists(normalized_file):
            with open(normalized_file, 'r') as f:
                preprocessing_stats['normalized_images'] = len(f.readlines())
        
        return preprocessing_stats
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mendapatkan statistik preprocessing: {str(e)}")
        return {
            'processed_images': 0,
            'filtered_images': 0,
            'normalized_images': 0
        }


def get_augmentation_stats(dataset_path: str) -> Dict[str, int]:
    """
    Mendapatkan statistik augmentasi dari direktori dataset.
    
    Args:
        dataset_path: Path ke direktori dataset
        
    Returns:
        Dictionary berisi statistik augmentasi
    """
    try:
        # Inisialisasi statistik augmentasi
        augmentation_stats = {
            'augmented_images': 0,
            'generated_images': 0,
            'augmentation_types': 0
        }
        
        # Path untuk metadata augmentasi
        augmentation_meta_path = os.path.join(dataset_path, 'metadata', 'augmentation')
        
        # Cek apakah direktori metadata ada
        if not os.path.exists(augmentation_meta_path):
            return augmentation_stats
        
        # Cek file metadata augmentasi
        augmented_file = os.path.join(augmentation_meta_path, 'augmented_images.txt')
        if os.path.exists(augmented_file):
            with open(augmented_file, 'r') as f:
                augmentation_stats['augmented_images'] = len(f.readlines())
        
        # Cek file metadata generated
        generated_file = os.path.join(augmentation_meta_path, 'generated_images.txt')
        if os.path.exists(generated_file):
            with open(generated_file, 'r') as f:
                augmentation_stats['generated_images'] = len(f.readlines())
        
        # Cek file metadata tipe augmentasi
        types_file = os.path.join(augmentation_meta_path, 'augmentation_types.txt')
        if os.path.exists(types_file):
            with open(types_file, 'r') as f:
                augmentation_stats['augmentation_types'] = len(f.readlines())
        
        return augmentation_stats
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mendapatkan statistik augmentasi: {str(e)}")
        return {
            'augmented_images': 0,
            'generated_images': 0,
            'augmentation_types': 0
        }


def get_processing_status(dataset_path: str) -> Dict[str, Dict[str, bool]]:
    """
    Mendapatkan status preprocessing dan augmentasi untuk setiap split.
    
    Args:
        dataset_path: Path ke direktori dataset
        
    Returns:
        Dictionary berisi status preprocessing dan augmentasi per split
    """
    try:
        # Inisialisasi status
        preprocessing_status = {
            'train': False,
            'val': False,
            'test': False
        }
        
        augmentation_status = {
            'train': False,
            'val': False,
            'test': False
        }
        
        # Path untuk metadata
        metadata_path = os.path.join(dataset_path, 'metadata')
        
        # Cek status preprocessing
        preprocessing_meta_path = os.path.join(metadata_path, 'preprocessing')
        if os.path.exists(preprocessing_meta_path):
            for split in ['train', 'val', 'test']:
                split_file = os.path.join(preprocessing_meta_path, f'{split}_processed.txt')
                if os.path.exists(split_file):
                    preprocessing_status[split] = True
        
        # Cek status augmentasi
        augmentation_meta_path = os.path.join(metadata_path, 'augmentation')
        if os.path.exists(augmentation_meta_path):
            for split in ['train', 'val', 'test']:
                split_file = os.path.join(augmentation_meta_path, f'{split}_augmented.txt')
                if os.path.exists(split_file):
                    augmentation_status[split] = True
        
        return {
            'preprocessing_status': preprocessing_status,
            'augmentation_status': augmentation_status
        }
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mendapatkan status processing: {str(e)}")
        return {
            'preprocessing_status': {'train': False, 'val': False, 'test': False},
            'augmentation_status': {'train': False, 'val': False, 'test': False}
        }


def update_dashboard_cards(ui_components: Dict[str, Any]) -> None:
    """
    Memperbarui dashboard cards dengan data terbaru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Tampilkan status loading
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat data dashboard..."))
        
        # Tampilkan progress bar
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 10
        
        # Dapatkan path dataset
        config_manager = get_config_manager()
        dataset_path = config_manager.get('dataset_path', None)
        
        if not dataset_path or not os.path.exists(dataset_path):
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
            ui_components['progress_bar'].layout.visibility = 'hidden'
            return
        
        # Update progress
        ui_components['progress_bar'].value = 30
        
        # Dapatkan statistik dataset
        stats = get_dataset_stats()
        split_stats = stats['split_stats']
        preprocessing_stats = stats['preprocessing_stats']
        augmentation_stats = stats['augmentation_stats']
        
        # Update progress
        ui_components['progress_bar'].value = 60
        
        # Dapatkan status processing
        processing_status = get_processing_status(dataset_path)
        preprocessing_status = processing_status['preprocessing_status']
        augmentation_status = processing_status['augmentation_status']
        
        # Update progress
        ui_components['progress_bar'].value = 80
        
        # Update split cards
        with ui_components['split_cards_container']:
            clear_output(wait=True)
            display(create_split_cards(split_stats, preprocessing_status, augmentation_status))
        
        # Update preprocessing cards
        with ui_components['preprocessing_cards']:
            clear_output(wait=True)
            display(create_preprocessing_cards(preprocessing_stats))
        
        # Update augmentation cards
        with ui_components['augmentation_cards']:
            clear_output(wait=True)
            display(create_augmentation_cards(augmentation_stats))
        
        # Update progress
        ui_components['progress_bar'].value = 100
        
        # Tampilkan status sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Dashboard berhasil diperbarui"))
        
        # Sembunyikan progress bar
        ui_components['progress_bar'].layout.visibility = 'hidden'
        
    except Exception as e:
        # Tampilkan error
        with ui_components['status']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui dashboard: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
        
        # Sembunyikan progress bar
        ui_components['progress_bar'].layout.visibility = 'hidden'


def setup_dashboard_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk dashboard visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol refresh
    ui_components['refresh_button'].on_click(
        lambda b: update_dashboard_cards(ui_components)
    )
    
    # Update dashboard cards saat pertama kali
    threading.Thread(target=update_dashboard_cards, args=(ui_components,)).start()
    
    return ui_components
