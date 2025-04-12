"""
File: smartcash/ui/dataset/augmentation/handlers/visualization_handlers.py
Deskripsi: Handler visualisasi untuk augmentasi dataset
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi hasil augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel augmentasi."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi sampel..."))
        
        try:
            # Gunakan shared utility dari module shared jika tersedia
            try:
                from smartcash.ui.dataset.shared.visualization_handler import _visualize_augmented_samples
                _visualize_augmented_samples(output_widget, 5)
                return
            except ImportError:
                pass
            
            # Fallback jika shared utility tidak tersedia
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
            
            # Cari lokasi yang memiliki file augmentasi (prioritas ke preprocessed)
            primary_path = Path(preprocessed_dir) / 'train' / 'images'
            secondary_path = Path(augmented_dir) / 'images'
            
            # Dapatkan prefix dari UI
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components else 'aug'
            
            # Cek mana yang memiliki file augmentasi
            active_path = None
            if primary_path.exists() and list(primary_path.glob(f"{aug_prefix}_*.*")):
                active_path = primary_path
            elif secondary_path.exists() and list(secondary_path.glob(f"{aug_prefix}_*.*")):
                active_path = secondary_path
            else:
                with output_widget:
                    display(create_status_indicator('warning', 
                        f"{ICONS['warning']} Tidak ditemukan file augmentasi dengan prefix '{aug_prefix}'"))
                return
            
            # Import visualisasi sesuai kebutuhan
            from smartcash.ui.charts.visualize_augmented_samples import visualize_augmented_samples
            
            # Tampilkan visualisasi
            with output_widget:
                visualize_augmented_samples(active_path, output_widget, ui_components)
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi sampel: {str(e)}")
    
    # Handler untuk tombol komparasi
    def on_compare_click(b):
        """Handler untuk komparasi dataset original vs augmented."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi dataset..."))
        
        try:
            # Gunakan shared utility dari module shared jika tersedia
            try:
                from smartcash.ui.dataset.shared.visualization_handler import _compare_original_vs_augmented
                _compare_original_vs_augmented(output_widget)
                return
            except ImportError:
                pass
            
            # Fallback jika shared utility tidak tersedia
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
            
            # Original path (preprocessed train split)
            original_path = Path(preprocessed_dir) / 'train' / 'images'
            
            # Cek lokasi augmentasi (prioritas ke preprocessed/train)
            primary_path = Path(preprocessed_dir) / 'train' / 'images'
            secondary_path = Path(augmented_dir) / 'images'
            
            if not original_path.exists():
                with output_widget:
                    display(create_status_indicator('warning', 
                        f"{ICONS['warning']} Direktori original tidak ditemukan: {original_path}"))
                return
            
            # Cari lokasi aktif
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components else 'aug'
            active_path = primary_path if primary_path.exists() and list(primary_path.glob(f"{aug_prefix}_*.*")) else secondary_path
            
            # Import komparasi
            from smartcash.ui.charts.compare_original_vs_augmented import compare_original_vs_augmented
            
            # Tampilkan komparasi
            with output_widget:
                compare_original_vs_augmented(original_path, active_path, output_widget, ui_components)
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Handler untuk tombol distribusi
    def on_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Gunakan shared utility dari module shared jika tersedia
            try:
                from smartcash.ui.dataset.shared.visualization_handler import _visualize_class_distribution
                _visualize_class_distribution(output_widget)
                return
            except ImportError:
                pass
            
            # Fallback jika shared utility tidak tersedia
            from smartcash.ui.charts.visualization_integrator import create_distribution_visualizations
            
            # Dataset path
            dataset_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Verifikasi direktori
            if not Path(dataset_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', 
                        f"{ICONS['warning']} Direktori dataset tidak ditemukan: {dataset_dir}"))
                return
            
            # Build parameter untuk visualisasi
            viz_params = {
                "visualization_container": output_widget,
                "logger": logger,
                "status": output_widget,
                "data_dir": ui_components.get('data_dir', 'data')
            }
            
            # Dapatkan prefix augmentasi
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components else 'aug'
            
            # Tampilkan visualisasi distribusi
            with output_widget:
                create_distribution_visualizations(
                    ui_components=viz_params,
                    dataset_dir=dataset_dir,
                    split_name='train',
                    aug_prefix=aug_prefix,
                    orig_prefix='rp',
                    target_count=1000
                )
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi distribusi kelas: {str(e)}")
    
    # Register handlers ke tombol visualisasi
    ui_components['visualize_button'].on_click(on_visualize_click)
    ui_components['compare_button'].on_click(on_compare_click)
    ui_components['distribution_button'].on_click(on_distribution_click)
    
    # Tambahkan handlers ke UI components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'on_distribution_click': on_distribution_click
    })
    
    return ui_components