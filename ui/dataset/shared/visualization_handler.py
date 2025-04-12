"""
File: smartcash/ui/dataset/shared/visualization_handler.py
Deskripsi: Utilitas bersama untuk visualisasi dataset dengan komponen atomik
untuk preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from IPython.display import display, clear_output, HTML
from pathlib import Path
import random
import re
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_shared_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None, 
                                      module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset dengan fungsi atomik yang dioptimasi.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (optional)
        config: Konfigurasi aplikasi (optional)
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Simpan cache hasil visualisasi untuk performa yang lebih baik
    visualization_cache = {'last_samples': {}, 'last_comparison': {}, 'last_distribution': {}}
    
    # Handler atomik untuk visualisasi sampel
    def on_visualize_click(b) -> None:
        """Handler untuk visualisasi sampel dataset dengan auto-detection format."""
        # Dapatkan output widget untuk tampilan visualisasi
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi sampel..."))
        
        try:
            # Tentukan parameter berdasarkan module_type
            if module_type == 'preprocessing':
                # Visualisasi sampel preprocessing
                _visualize_preprocessed_samples(output_widget)
            else:
                # Visualisasi sampel augmentasi
                _visualize_augmented_samples(output_widget)
                
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi: {str(e)}")
    
    # Handler atomik untuk komparasi
    def on_compare_click(b) -> None:
        """Handler untuk komparasi dataset original vs processed."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi dataset..."))
        
        try:
            # Tentukan parameter berdasarkan module_type
            if module_type == 'preprocessing':
                # Komparasi original vs preprocessed
                _compare_original_vs_processed(output_widget)
            else:
                # Komparasi original vs augmented
                _compare_original_vs_augmented(output_widget)
                
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Handler atomik untuk visualisasi distribusi
    def on_distribution_click(b) -> None:
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Visualisasi distribusi dengan fungsi atomik
            _visualize_class_distribution(output_widget)
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}")
    
    # ==== Fungsi atomik internal untuk implementasi visualisasi ====
    
    def _visualize_preprocessed_samples(output_widget, num_samples: int = 5) -> None:
        """
        Visualisasi sampel hasil preprocessing dengan optimasi performa.
        
        Args:
            output_widget: Widget untuk output visualisasi
            num_samples: Jumlah sampel yang ditampilkan
        """
        # Jika ada cache dan parameter sama, gunakan cache
        cache_key = f"preproc_{num_samples}"
        if cache_key in visualization_cache['last_samples']:
            with output_widget:
                display(visualization_cache['last_samples'][cache_key])
                display(create_status_indicator('info', f"{ICONS['info']} Menampilkan {num_samples} sampel preprocessing (dari cache)"))
            return
            
        # Impor visualisasi preprocessed dengan lazy loading
        from smartcash.ui.charts.visualize_preprocessed_samples import visualize_preprocessed_samples
        
        # Parameter preprocessing
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        original_dir = ui_components.get('data_dir', 'data')
        
        # Generate visualisasi
        with output_widget:
            # Cache hasil visualisasi untuk penggunaan berikutnya
            visualization_cache['last_samples'][cache_key] = visualize_preprocessed_samples(
                ui_components=ui_components,
                preprocessed_dir=preprocessed_dir,
                original_dir=original_dir,
                num_samples=num_samples,
                return_widget=True
            )
    
    def _visualize_augmented_samples(output_widget, num_samples: int = 5) -> None:
        """
        Visualisasi sampel hasil augmentasi dengan optimasi performa.
        
        Args:
            output_widget: Widget untuk output visualisasi
            num_samples: Jumlah sampel yang ditampilkan
        """
        # Jika ada cache dan parameter sama, gunakan cache
        cache_key = f"aug_{num_samples}"
        if cache_key in visualization_cache['last_samples']:
            with output_widget:
                display(visualization_cache['last_samples'][cache_key])
                display(create_status_indicator('info', f"{ICONS['info']} Menampilkan {num_samples} sampel augmentasi (dari cache)"))
            return
        
        # Multi-location detection untuk file augmentasi
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Cari lokasi yang memiliki file augmentasi (prioritas ke preprocessed)
        primary_path = Path(preprocessed_dir) / 'train' / 'images'
        secondary_path = Path(augmented_dir) / 'images'
        
        # Cek mana yang memiliki file augmentasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Deteksi otomatis active path
        active_path = None
        if primary_path.exists() and list(primary_path.glob(f"{aug_prefix}_*.*")):
            active_path = primary_path
        elif secondary_path.exists() and list(secondary_path.glob(f"{aug_prefix}_*.*")):
            active_path = secondary_path
        else:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file augmentasi dengan prefix '{aug_prefix}'"))
            return
        
        # Import visualisasi augmented dengan lazy loading
        from smartcash.ui.charts.visualize_augmented_samples import visualize_augmented_samples
        
        # Generate visualisasi
        with output_widget:
            try:
                # Cache hasil visualisasi untuk penggunaan berikutnya
                visualization_cache['last_samples'][cache_key] = visualize_augmented_samples(
                    active_path, output_widget, ui_components, num_samples, return_widget=True
                )
            except Exception as e:
                display(create_status_indicator('error', f"{ICONS['error']} Error visualisasi augmentasi: {str(e)}"))
                raise e
    
    def _compare_original_vs_processed(output_widget) -> None:
        """
        Komparasi dataset original vs preprocessed dengan lazy loading.
        
        Args:
            output_widget: Widget untuk output visualisasi
        """
        # Jika ada cache, gunakan cache
        if 'preproc_comparison' in visualization_cache['last_comparison']:
            with output_widget:
                display(visualization_cache['last_comparison']['preproc_comparison'])
                display(create_status_indicator('info', f"{ICONS['info']} Menampilkan komparasi preprocessing (dari cache)"))
            return
        
        # Impor komparasi dengan lazy loading
        from smartcash.ui.charts.compare_original_vs_preprocessed import compare_original_vs_preprocessed
        
        # Parameter preprocessing
        raw_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Generate komparasi
        with output_widget:
            try:
                # Cache hasil komparasi untuk penggunaan berikutnya
                visualization_cache['last_comparison']['preproc_comparison'] = compare_original_vs_preprocessed(
                    ui_components=ui_components,
                    raw_dir=raw_dir,
                    preprocessed_dir=preprocessed_dir,
                    return_widget=True
                )
            except Exception as e:
                display(create_status_indicator('error', f"{ICONS['error']} Error komparasi preprocessing: {str(e)}"))
                raise e
    
    def _compare_original_vs_augmented(output_widget) -> None:
        """
        Komparasi dataset original vs augmented dengan lazy loading.
        
        Args:
            output_widget: Widget untuk output visualisasi
        """
        # Jika ada cache, gunakan cache
        if 'aug_comparison' in visualization_cache['last_comparison']:
            with output_widget:
                display(visualization_cache['last_comparison']['aug_comparison'])
                display(create_status_indicator('info', f"{ICONS['info']} Menampilkan komparasi augmentasi (dari cache)"))
            return
        
        # Multi-location detection untuk file augmentasi
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Original path (preprocessed train split)
        original_path = Path(preprocessed_dir) / 'train' / 'images'
        
        # Cek lokasi augmentasi (prioritas ke preprocessed/train)
        primary_path = Path(preprocessed_dir) / 'train' / 'images'
        secondary_path = Path(augmented_dir) / 'images'
        
        if not original_path.exists():
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori original tidak ditemukan: {original_path}"))
            return
            
        # Impor komparasi dengan lazy loading
        from smartcash.ui.charts.compare_original_vs_augmented import compare_original_vs_augmented
        
        # Generate komparasi
        with output_widget:
            try:
                # Cari lokasi aktif
                aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
                active_path = primary_path if primary_path.exists() and list(primary_path.glob(f"{aug_prefix}_*.*")) else secondary_path
                
                # Cache hasil komparasi untuk penggunaan berikutnya
                visualization_cache['last_comparison']['aug_comparison'] = compare_original_vs_augmented(
                    original_path, active_path, output_widget, ui_components, return_widget=True
                )
            except Exception as e:
                display(create_status_indicator('error', f"{ICONS['error']} Error komparasi augmentasi: {str(e)}"))
                raise e
    
    def _visualize_class_distribution(output_widget) -> None:
        """
        Visualisasi distribusi kelas dengan fungsi atomik, optimize pandas loading.
        
        Args:
            output_widget: Widget untuk output visualisasi
        """
        # Jika ada cache, gunakan cache
        if 'class_distribution' in visualization_cache['last_distribution']:
            with output_widget:
                display(visualization_cache['last_distribution']['class_distribution'])
                display(create_status_indicator('info', f"{ICONS['info']} Menampilkan distribusi kelas (dari cache)"))
            return
        
        # Import helper untuk distribusi kelas dengan lazy loading
        from smartcash.ui.charts.visualization_integrator import create_distribution_visualizations
        
        # Dataset path berdasarkan module_type
        dataset_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Cek keberadaan direktori
        if not Path(dataset_dir).exists():
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset tidak ditemukan: {dataset_dir}"))
            return
        
        # Build parameter dinamis berdasarkan module_type
        vis_params = {
            "visualization_container": output_widget,
            "logger": logger,
            "status": output_widget,
            "data_dir": ui_components.get('data_dir', 'data')
        }
        
        # Tambahkan parameter khusus augmentasi jika relevan
        if module_type == 'augmentation' and 'aug_options' in ui_components:
            vis_params['aug_options'] = ui_components['aug_options']
            aug_prefix = ui_components['aug_options'].children[2].value if len(ui_components['aug_options'].children) > 2 else 'aug'
        else:
            aug_prefix = 'aug'
            
        # Prefix untuk original file tergantung module_type
        orig_prefix = 'rp' if module_type == 'preprocessing' else 'rp'
        
        # Generate visualisasi distribusi
        with output_widget:
            try:
                # Cache hasil visualisasi untuk penggunaan berikutnya
                visualization_cache['last_distribution']['class_distribution'] = create_distribution_visualizations(
                    ui_components=vis_params,
                    dataset_dir=dataset_dir,
                    split_name='train',
                    aug_prefix=aug_prefix,
                    orig_prefix=orig_prefix,
                    target_count=1000,
                    return_widget=True
                )
            except Exception as e:
                display(create_status_indicator('error', f"{ICONS['error']} Error visualisasi distribusi: {str(e)}"))
                raise e
    
    # Register handlers ke tombol-tombol visualisasi
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_click)
    
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_click)
    
    if 'distribution_button' in ui_components:
        ui_components['distribution_button'].on_click(on_distribution_click)
    
    # Tambahkan handlers dan cache ke UI components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'on_distribution_click': on_distribution_click,
        'visualization_cache': visualization_cache
    })
    
    return ui_components