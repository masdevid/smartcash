"""
File: smartcash/ui/dataset/visualization/handlers/dashboard_handlers.py
Deskripsi: Handler untuk dashboard visualisasi dataset
"""

import os
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import ConfigManager, get_config_manager
from smartcash.common.environment import get_default_base_dir
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_preprocessing_cards, create_augmentation_cards
)
from smartcash.ui.dataset.visualization.components.split_stats_cards import create_split_stats_cards
from smartcash.ui.dataset.visualization.components.comparison_cards import create_comparison_cards
from smartcash.ui.utils.loading_indicator import create_loading_indicator, LoadingIndicator
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_success_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def setup_dashboard_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk dashboard visualisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol refresh
    if 'refresh_button' in ui_components:
        ui_components['refresh_button'].on_click(
            lambda b: on_refresh_click(b, ui_components)
        )
    
    return ui_components

def on_refresh_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol refresh.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memperbarui data dashboard...")
    
    # Update dashboard cards
    update_dashboard_cards(ui_components)

def get_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """
    Dapatkan statistik dataset dari service.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Dictionary berisi statistik dataset
    """
    try:
        # Ambil dataset service
        dataset_service = get_dataset_service(service_name='visualization')
        
        # Dapatkan statistik dataset
        stats = dataset_service.get_dataset_statistics(dataset_path)
        return stats
    except Exception as e:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat mengakses dataset: {str(e)}")
        return {}

def get_empty_statistics() -> Dict[str, Any]:
    """
    Dapatkan statistik kosong untuk inisialisasi awal.
    
    Returns:
        Dictionary berisi statistik kosong
    """
    return {
        'split': {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
        },
        'preprocessing': {
            'processed_images': 0,
            'filtered_images': 0,
            'normalized_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0
        },
        'augmentation': {
            'augmented_images': 0,
            'augmentations_created': 0,
            'augmentation_types': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0
        }
    }

def find_valid_dataset_path() -> str:
    """
    Cari path dataset yang valid dari berbagai sumber.
    
    Returns:
        Path dataset yang valid atau path default
    """
    # Daftar kemungkinan lokasi dataset
    possible_paths = [
        'data',                     # Relative to current directory
        'data/dataset',             # Common subdirectory
        '/content/data',            # Colab path
        '/content/data/dataset',    # Colab subdirectory
        os.path.join(get_default_base_dir(), 'data'),  # Base dir
        os.path.join(get_default_base_dir(), 'data/dataset')  # Base dir subdirectory
    ]
    
    # Coba dapatkan dari config manager
    try:
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        config_path = dataset_config.get('dataset_path', None)
        if config_path and os.path.exists(config_path):
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan path dataset dari konfigurasi: {config_path}")
            return config_path
    except Exception as e:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat mengakses konfigurasi dataset: {str(e)}")
    
    # Coba setiap kemungkinan path
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan path dataset yang ditemukan: {path}")
            return path
    
    # Gunakan default jika tidak ada yang valid
    default_path = 'data'
    logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak menemukan path dataset valid, menggunakan default: {default_path}")
    return default_path

def update_dashboard_cards(ui_components: Dict[str, Any]) -> None:
    """
    Update dashboard cards dengan data terbaru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Ambil komponen UI
        preprocessing_cards = ui_components.get('preprocessing_cards')
        augmentation_cards = ui_components.get('augmentation_cards')
        split_cards_container = ui_components.get('split_cards_container')
        comparison_cards_container = ui_components.get('comparison_cards_container')
        progress_bar = ui_components.get('progress_bar')
        
        # Tampilkan progress bar jika ada
        if progress_bar:
            progress_bar.value = 0
            progress_bar.layout.visibility = 'visible'
        
        # Inisialisasi dengan statistik kosong
        stats = get_empty_statistics()
        
        # Flag untuk menandai apakah data valid
        data_valid = False
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = 20
        
        # Cari dataset path yang valid
        dataset_path = find_valid_dataset_path()
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = 40
            
        # Dapatkan statistik dataset jika path valid
        if os.path.exists(dataset_path):
            # Dapatkan statistik dataset
            dataset_stats = get_dataset_statistics(dataset_path)
            
            # Jika statistik tidak kosong, gunakan data tersebut
            if dataset_stats:
                stats = dataset_stats
                data_valid = True
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = 60
        
        # Update split cards jika container tersedia
        if split_cards_container is not None:
            if hasattr(split_cards_container, 'clear_output'):
                split_cards_container.clear_output(wait=True)
                with split_cards_container:
                    display(create_split_stats_cards(stats.get('split', {})))
        
        # Update comparison cards jika container tersedia
        if comparison_cards_container is not None:
            if hasattr(comparison_cards_container, 'clear_output'):
                comparison_cards_container.clear_output(wait=True)
                with comparison_cards_container:
                    # Dapatkan statistik preprocessing dan augmentation
                    preprocessing_stats = {
                        'train': {'images': stats.get('preprocessing', {}).get('train_images', 0), 'labels': 0},
                        'val': {'images': stats.get('preprocessing', {}).get('val_images', 0), 'labels': 0},
                        'test': {'images': stats.get('preprocessing', {}).get('test_images', 0), 'labels': 0}
                    }
                    
                    augmentation_stats = {
                        'train': {'images': stats.get('augmentation', {}).get('train_images', 0), 'labels': 0},
                        'val': {'images': stats.get('augmentation', {}).get('val_images', 0), 'labels': 0},
                        'test': {'images': stats.get('augmentation', {}).get('test_images', 0), 'labels': 0}
                    }
                    
                    display(create_comparison_cards(
                        stats.get('split', {}),
                        preprocessing_stats,
                        augmentation_stats
                    ))
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = 80
        
        # Update preprocessing cards jika container tersedia
        if preprocessing_cards is not None:
            if hasattr(preprocessing_cards, 'clear_output'):
                preprocessing_cards.clear_output(wait=True)
                with preprocessing_cards:
                    display(create_preprocessing_cards(stats.get('preprocessing', {})))
        
        # Update augmentation cards jika container tersedia
        if augmentation_cards is not None:
            if hasattr(augmentation_cards, 'clear_output'):
                augmentation_cards.clear_output(wait=True)
                with augmentation_cards:
                    display(create_augmentation_cards(stats.get('augmentation', {})))
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = 100
            # Sembunyikan progress bar setelah selesai
            progress_bar.layout.visibility = 'hidden'
        
        # Tampilkan pesan sukses atau warning
        if data_valid:
            show_success_status(ui_components, f"Dashboard berhasil diperbarui dengan data dari {dataset_path}")
        else:
            show_warning_status(ui_components, f"Dataset tidak tersedia di {dataset_path}. Menampilkan data kosong.")
        
        # Log dengan timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        success_message = f"Dashboard berhasil diperbarui" + (f" dengan data valid dari {dataset_path}" if data_valid else f" dengan data kosong (path: {dataset_path})")
        logger.info(f"[{current_time}] {ICONS.get('info', 'ℹ️')} 📊 {success_message}")
    
    except Exception as e:
        error_message = f"Error saat memperbarui dashboard: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        
        # Tampilkan pesan error
        show_error_status(ui_components, error_message)
        
        # Sembunyikan progress bar jika ada
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'hidden' 