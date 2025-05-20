"""
File: smartcash/ui/dataset/visualization/handlers/tabs/augmentation_tab.py
Deskripsi: Handler untuk tab perbandingan augmentasi
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

def on_augmentation_comparison_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol perbandingan augmentasi.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat perbandingan augmentasi...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    augmentation_comparison_tab = visualization_components.get('augmentation_comparison_tab', {})
    output = augmentation_comparison_tab.get('output')
    
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
            _show_dummy_augmentation_comparison(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Dapatkan sampel augmentasi
            try:
                # Coba dapatkan sampel augmentasi dari service
                augmentation_samples = explorer_service.get_augmentation_samples(3)
                _show_augmentation_comparison(output, augmentation_samples)
            except (AttributeError, NotImplementedError):
                # Fallback ke dummy jika metode tidak tersedia
                logger.warning("Metode get_augmentation_samples tidak tersedia. Menampilkan data dummy.")
                _show_dummy_augmentation_comparison(output)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Perbandingan augmentasi berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan perbandingan augmentasi: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_augmentation_comparison(output: widgets.Output) -> None:
    """
    Tampilkan perbandingan augmentasi dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat grid 3x2 untuk menampilkan 3 pasang gambar (original dan augmented)
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        # Buat sampel dummy
        np.random.seed(42)  # Untuk hasil yang konsisten
        
        # Augmentation types
        aug_types = ["Rotasi", "Flip Horizontal", "Brightness"]
        
        for i in range(3):
            # Buat gambar original dummy
            original_img = np.random.rand(100, 100, 3)
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')
            
            # Buat gambar augmented dummy (sedikit berbeda dari original)
            if i == 0:  # Rotasi
                augmented_img = np.rot90(original_img)
            elif i == 1:  # Flip horizontal
                augmented_img = np.fliplr(original_img)
            else:  # Brightness
                augmented_img = np.clip(original_img * 1.5, 0, 1)
                
            axes[i, 1].imshow(augmented_img)
            axes[i, 1].set_title(f"Augmented ({aug_types[i]})")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan deskripsi augmentasi
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Teknik Augmentasi yang Digunakan</h3>
            <ul>
                <li><strong>Rotasi:</strong> Memutar gambar dengan sudut acak antara -15° hingga 15°</li>
                <li><strong>Flip Horizontal:</strong> Membalik gambar secara horizontal</li>
                <li><strong>Brightness & Contrast:</strong> Mengubah kecerahan dan kontras gambar secara acak</li>
                <li><strong>Noise:</strong> Menambahkan noise acak pada gambar</li>
                <li><strong>Blur:</strong> Menerapkan blur Gaussian ringan pada gambar</li>
            </ul>
            <h3>Manfaat Augmentasi</h3>
            <ul>
                <li><strong>Peningkatan Dataset:</strong> Memperbanyak jumlah data training tanpa perlu mengumpulkan data baru</li>
                <li><strong>Generalisasi Model:</strong> Membantu model belajar fitur yang lebih robust dan invariant terhadap transformasi</li>
                <li><strong>Mencegah Overfitting:</strong> Mengurangi risiko model terlalu menyesuaikan dengan data training</li>
                <li><strong>Meningkatkan Performa:</strong> Meningkatkan akurasi dan performa model pada data yang belum pernah dilihat</li>
            </ul>
        </div>
        """))

def _show_augmentation_comparison(output: widgets.Output, augmentation_samples: List[Dict[str, Any]]) -> None:
    """
    Tampilkan perbandingan augmentasi dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        augmentation_samples: List sampel augmentasi
    """
    with output:
        clear_output(wait=True)
        
        if augmentation_samples and len(augmentation_samples) > 0:
            # Buat grid untuk menampilkan sampel augmentasi
            n_samples = min(len(augmentation_samples), 3)
            fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
            
            # Tampilkan sampel
            for i, sample in enumerate(augmentation_samples[:n_samples]):
                original_img = sample.get('original', np.zeros((100, 100, 3)))
                augmented_img = sample.get('augmented', np.zeros((100, 100, 3)))
                aug_type = sample.get('augmentation_type', 'Unknown')
                
                if n_samples == 1:
                    axes[0].imshow(original_img)
                    axes[0].set_title(f"Original {i+1}")
                    axes[0].axis('off')
                    
                    axes[1].imshow(augmented_img)
                    axes[1].set_title(f"Augmented ({aug_type})")
                    axes[1].axis('off')
                else:
                    axes[i, 0].imshow(original_img)
                    axes[i, 0].set_title(f"Original {i+1}")
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(augmented_img)
                    axes[i, 1].set_title(f"Augmented ({aug_type})")
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Tampilkan deskripsi augmentasi jika tersedia
            aug_types = []
            for sample in augmentation_samples:
                if 'augmentation_type' in sample and sample['augmentation_type'] not in aug_types:
                    aug_types.append(sample['augmentation_type'])
            
            if aug_types:
                aug_types_html = ""
                for aug_type in aug_types:
                    aug_types_html += f"<li><strong>{aug_type}:</strong> {_get_augmentation_description(aug_type)}</li>"
                
                display(widgets.HTML(f"""
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3>Teknik Augmentasi yang Digunakan</h3>
                    <ul>
                        {aug_types_html}
                    </ul>
                    <h3>Manfaat Augmentasi</h3>
                    <ul>
                        <li><strong>Peningkatan Dataset:</strong> Memperbanyak jumlah data training tanpa perlu mengumpulkan data baru</li>
                        <li><strong>Generalisasi Model:</strong> Membantu model belajar fitur yang lebih robust dan invariant terhadap transformasi</li>
                        <li><strong>Mencegah Overfitting:</strong> Mengurangi risiko model terlalu menyesuaikan dengan data training</li>
                        <li><strong>Meningkatkan Performa:</strong> Meningkatkan akurasi dan performa model pada data yang belum pernah dilihat</li>
                    </ul>
                </div>
                """))
            else:
                # Tampilkan deskripsi augmentasi default
                display(widgets.HTML("""
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3>Teknik Augmentasi yang Digunakan</h3>
                    <ul>
                        <li><strong>Rotasi:</strong> Memutar gambar dengan sudut acak antara -15° hingga 15°</li>
                        <li><strong>Flip Horizontal:</strong> Membalik gambar secara horizontal</li>
                        <li><strong>Brightness & Contrast:</strong> Mengubah kecerahan dan kontras gambar secara acak</li>
                        <li><strong>Noise:</strong> Menambahkan noise acak pada gambar</li>
                        <li><strong>Blur:</strong> Menerapkan blur Gaussian ringan pada gambar</li>
                    </ul>
                    <h3>Manfaat Augmentasi</h3>
                    <ul>
                        <li><strong>Peningkatan Dataset:</strong> Memperbanyak jumlah data training tanpa perlu mengumpulkan data baru</li>
                        <li><strong>Generalisasi Model:</strong> Membantu model belajar fitur yang lebih robust dan invariant terhadap transformasi</li>
                        <li><strong>Mencegah Overfitting:</strong> Mengurangi risiko model terlalu menyesuaikan dengan data training</li>
                        <li><strong>Meningkatkan Performa:</strong> Meningkatkan akurasi dan performa model pada data yang belum pernah dilihat</li>
                    </ul>
                </div>
                """))
        else:
            display(widgets.HTML("""
            <div style="margin-top: 20px; padding: 10px; background-color: #fff3cd; border-radius: 5px; color: #856404;">
                <h3>Tidak ada sampel augmentasi yang tersedia</h3>
                <p>Pastikan dataset telah melalui proses augmentasi.</p>
            </div>
            """))

def _get_augmentation_description(aug_type: str) -> str:
    """
    Dapatkan deskripsi untuk tipe augmentasi.
    
    Args:
        aug_type: Tipe augmentasi
        
    Returns:
        Deskripsi augmentasi
    """
    descriptions = {
        'rotation': 'Memutar gambar dengan sudut acak antara -15° hingga 15°',
        'flip': 'Membalik gambar secara horizontal',
        'brightness': 'Mengubah kecerahan gambar secara acak',
        'contrast': 'Mengubah kontras gambar secara acak',
        'noise': 'Menambahkan noise acak pada gambar',
        'blur': 'Menerapkan blur Gaussian ringan pada gambar',
        'scale': 'Mengubah skala gambar secara acak',
        'translate': 'Menggeser posisi gambar secara acak',
        'shear': 'Mengubah bentuk gambar dengan transformasi shear',
        'hsv': 'Mengubah nilai HSV (Hue, Saturation, Value) gambar secara acak'
    }
    
    # Coba dapatkan deskripsi yang cocok
    for key, desc in descriptions.items():
        if key.lower() in aug_type.lower():
            return desc
    
    # Default description
    return 'Transformasi gambar untuk meningkatkan variasi dataset' 