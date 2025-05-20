"""
File: smartcash/ui/dataset/visualization/handlers/tabs/preprocessing_tab.py
Deskripsi: Handler untuk tab sampel preprocessing
"""

import os
from typing import Dict, Any, List
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.common.config.manager import get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_success_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def on_preprocessing_samples_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol sampel preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat sampel preprocessing...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    preprocessing_samples_tab = visualization_components.get('preprocessing_samples_tab', {})
    output = preprocessing_samples_tab.get('output')
    
    if output is None:
        show_error_status(ui_components, "Komponen output tidak ditemukan")
        return
    
    try:
        # Dapatkan dataset path dari konfigurasi
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', '/content/data')
        
        if not dataset_path or not os.path.exists(dataset_path):
            show_warning_status(ui_components, "Dataset path tidak valid. Menampilkan data dummy.")
            _show_dummy_preprocessing_samples(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Dapatkan sampel preprocessing
            try:
                # Coba dapatkan sampel preprocessing dari service
                preprocessing_samples = explorer_service.get_preprocessing_samples(3)
                _show_preprocessing_samples(output, preprocessing_samples)
            except (AttributeError, NotImplementedError):
                # Fallback ke dummy jika metode tidak tersedia
                logger.warning("Metode get_preprocessing_samples tidak tersedia. Menampilkan data dummy.")
                _show_dummy_preprocessing_samples(output)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Sampel preprocessing berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan sampel preprocessing: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_preprocessing_samples(output: widgets.Output) -> None:
    """
    Tampilkan sampel preprocessing dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat grid 1x3 untuk menampilkan 3 sampel preprocessing
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Buat sampel dummy
        np.random.seed(42)  # Untuk hasil yang konsisten
        
        # Original image (noisy)
        original = np.random.rand(100, 100, 3)
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Preprocessed image (normalized)
        normalized = (original - original.min()) / (original.max() - original.min())
        axes[1].imshow(normalized)
        axes[1].set_title("Normalized")
        axes[1].axis('off')
        
        # Preprocessed image (grayscale)
        grayscale = np.mean(original, axis=2)
        axes[2].imshow(grayscale, cmap='gray')
        axes[2].set_title("Grayscale")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan deskripsi preprocessing
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Tahapan Preprocessing</h3>
            <ol>
                <li><strong>Normalisasi:</strong> Menyesuaikan nilai piksel ke rentang 0-1 untuk memudahkan proses training</li>
                <li><strong>Resize:</strong> Mengubah ukuran gambar menjadi 640x640 piksel sesuai input YOLOv5</li>
                <li><strong>Augmentasi Warna:</strong> Menyesuaikan brightness, contrast, dan saturation untuk meningkatkan robustness model</li>
            </ol>
            <h3>Manfaat Preprocessing</h3>
            <ul>
                <li><strong>Standarisasi Input:</strong> Memastikan semua gambar memiliki format yang sama untuk model</li>
                <li><strong>Peningkatan Kualitas:</strong> Mengurangi noise dan meningkatkan fitur penting dalam gambar</li>
                <li><strong>Optimasi Performa:</strong> Membantu model konvergen lebih cepat dan mencapai akurasi lebih tinggi</li>
            </ul>
        </div>
        """))

def _show_preprocessing_samples(output: widgets.Output, preprocessing_samples: List[Dict[str, Any]]) -> None:
    """
    Tampilkan sampel preprocessing dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        preprocessing_samples: List sampel preprocessing
    """
    with output:
        clear_output(wait=True)
        
        if preprocessing_samples and len(preprocessing_samples) > 0:
            # Buat grid untuk menampilkan sampel preprocessing
            n_samples = min(len(preprocessing_samples), 3)
            fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
            
            # Tampilkan sampel
            for i, sample in enumerate(preprocessing_samples[:n_samples]):
                img = sample.get('image', np.zeros((100, 100, 3)))
                title = sample.get('title', f"Sampel {i+1}")
                
                if n_samples == 1:
                    axes.imshow(img)
                    axes.set_title(title)
                    axes.axis('off')
                else:
                    axes[i].imshow(img)
                    axes[i].set_title(title)
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Tampilkan deskripsi preprocessing steps jika tersedia
            steps = []
            for sample in preprocessing_samples:
                if 'steps' in sample:
                    steps.extend(sample['steps'])
            
            if steps:
                steps_html = ""
                for step in steps:
                    steps_html += f"<li><strong>{step['name']}:</strong> {step['description']}</li>"
                
                display(widgets.HTML(f"""
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3>Tahapan Preprocessing</h3>
                    <ol>
                        {steps_html}
                    </ol>
                </div>
                """))
            else:
                # Tampilkan deskripsi preprocessing default
                display(widgets.HTML("""
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3>Tahapan Preprocessing</h3>
                    <ol>
                        <li><strong>Normalisasi:</strong> Menyesuaikan nilai piksel ke rentang 0-1 untuk memudahkan proses training</li>
                        <li><strong>Resize:</strong> Mengubah ukuran gambar menjadi 640x640 piksel sesuai input YOLOv5</li>
                        <li><strong>Augmentasi Warna:</strong> Menyesuaikan brightness, contrast, dan saturation untuk meningkatkan robustness model</li>
                    </ol>
                    <h3>Manfaat Preprocessing</h3>
                    <ul>
                        <li><strong>Standarisasi Input:</strong> Memastikan semua gambar memiliki format yang sama untuk model</li>
                        <li><strong>Peningkatan Kualitas:</strong> Mengurangi noise dan meningkatkan fitur penting dalam gambar</li>
                        <li><strong>Optimasi Performa:</strong> Membantu model konvergen lebih cepat dan mencapai akurasi lebih tinggi</li>
                    </ul>
                </div>
                """))
        else:
            display(widgets.HTML("""
            <div style="margin-top: 20px; padding: 10px; background-color: #fff3cd; border-radius: 5px; color: #856404;">
                <h3>Tidak ada sampel preprocessing yang tersedia</h3>
                <p>Pastikan dataset telah melalui proses preprocessing.</p>
            </div>
            """)) 