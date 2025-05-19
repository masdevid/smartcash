"""
File: smartcash/ui/dataset/visualization/handlers/dashboard_handler.py
Deskripsi: Handler untuk dashboard visualisasi dataset dengan pendekatan minimalis
"""

import os
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
from datetime import datetime

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config import ConfigManager, get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_preprocessing_cards, create_augmentation_cards
)
from smartcash.ui.dataset.visualization.components.split_stats_cards import create_split_stats_cards
from smartcash.ui.utils.loading_indicator import create_loading_indicator, LoadingIndicator

logger = get_logger(__name__)


# Fungsi create_dashboard_handler telah dihapus karena tidak lagi digunakan
# Komponen dashboard sekarang dibuat langsung di visualization_initializer.py


def update_dashboard_cards(ui_components: Dict[str, Any]) -> None:
    """
    Update dashboard cards dengan data terbaru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Ambil komponen UI
        status_panel = ui_components.get('status_panel')
        preprocessing_cards = ui_components.get('preprocessing_cards')
        augmentation_cards = ui_components.get('augmentation_cards')
        split_stats_cards = ui_components.get('split_stats_cards')
        loading_indicator = ui_components.get('loading_indicator')
        
        # Tampilkan loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(25, "Memperbarui data dashboard...")
        elif status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memperbarui dashboard..."))
        
        # Inisialisasi variabel untuk menandai apakah menggunakan data dummy
        using_dummy_data = False
        error_message = ""
        stats = {}
        
        # Coba dapatkan dataset path dari konfigurasi
        try:
            # Ambil dataset service
            dataset_service = get_dataset_service(service_name='visualization')
            
            config_manager = ConfigManager()
            dataset_config = config_manager.get_config('dataset_config')
            dataset_path = dataset_config.get('dataset_path', None)
            
            if not dataset_path:
                error_message = "Dataset path tidak ditemukan dalam konfigurasi"
                logger.warning(f"{ICONS.get('warning', '⚠️')} {error_message}")
                using_dummy_data = True
            else:
                # Perbarui loading indicator jika ada
                if loading_indicator:
                    loading_indicator.update(40, "Mengambil statistik dataset...")
                    
                # Dapatkan statistik dataset
                stats = dataset_service.get_dataset_statistics(dataset_path)
        except Exception as e:
            error_message = str(e)
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat mengakses dataset: {error_message}")
            using_dummy_data = True
        
        # Jika menggunakan data dummy
        if using_dummy_data:
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(40, "Menyiapkan data dummy...")
            
            # Gunakan data dummy
            stats = {
                'split': {
                    'train': {'images': 1400, 'labels': 1400},
                    'val': {'images': 300, 'labels': 300},
                    'test': {'images': 300, 'labels': 300}
                },
                'preprocessing': {
                    'processed_images': 2000,
                    'filtered_images': 2000,
                    'normalized_images': 2000
                },
                'augmentation': {
                    'augmented_images': 2000,
                    'augmentations_created': 2000,
                    'augmentation_types': 5
                }
            }
            
            # Tampilkan pesan warning di status panel
            if status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(widgets.HTML(f"""
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #ffc107;">
                        <p style="margin: 0;"><strong>{ICONS.get('warning', '⚠️')} Status Data:</strong> Menggunakan data dummy untuk visualisasi. Silakan inisialisasi dataset untuk melihat data aktual.</p>
                    </div>
                    """))
        
        # Perbarui loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(60, "Memperbarui statistik split...")
        
        # Update split stats cards
        if split_stats_cards:
            with split_stats_cards:
                clear_output(wait=True)
                display(create_split_stats_cards(stats))
        
        # Perbarui loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(75, "Memperbarui kartu preprocessing dan augmentasi...")
        
        # Update preprocessing cards
        if preprocessing_cards:
            with preprocessing_cards:
                clear_output(wait=True)
                display(create_preprocessing_cards(stats.get('preprocessing', {})))
        
        # Update augmentation cards
        if augmentation_cards:
            with augmentation_cards:
                clear_output(wait=True)
                display(create_augmentation_cards(stats.get('augmentation', {})))
        
        # Tampilkan pesan sukses
        success_message = "Dashboard berhasil diperbarui" + (" dengan data dummy" if using_dummy_data else "")
        if loading_indicator:
            loading_indicator.complete(success_message)
        elif status_panel and not using_dummy_data:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(create_status_indicator("success", f"{ICONS.get('success', '✅')} {success_message}"))
        
        # Log dengan timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"{ICONS.get('info', 'ℹ️')} [{current_time}] 📊 {success_message}")
    
    except Exception as e:
        error_message = f"Error saat memperbarui dashboard: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        
        # Tampilkan pesan error
        if loading_indicator:
            loading_indicator.error(error_message)
        elif status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} {error_message}"))
