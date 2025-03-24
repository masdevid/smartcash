"""
File: smartcash/ui/dataset/shared/visualization_handler.py
Deskripsi: Utilitas bersama untuk visualisasi dataset yang digunakan oleh preprocessing dan augmentasi
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_shared_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None, 
                                     module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset di kedua modul.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk visualisasi sampel dataset
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset dengan dynamic loading berdasarkan module_type."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi..."))
        
        try:
            # Dapatkan direktori dan parameter berdasarkan module_type
            if module_type == 'preprocessing':
                # Import dan parameter untuk preprocessing
                from smartcash.ui.visualization.visualize_preprocessed_samples import visualize_preprocessed_samples
                visualize_preprocessed_samples(
                    ui_components=ui_components,
                    preprocessed_dir=ui_components.get('preprocessed_dir', 'data/preprocessed'),
                    original_dir=ui_components.get('data_dir', 'data'),
                    num_samples=5
                )
            else:
                # Import dan parameter untuk augmentation
                from smartcash.ui.visualization.visualize_augmented_samples import visualize_augmented_samples
                # Cari lokasi yang memiliki data augmentasi (multiple locations)
                primary_path = Path(ui_components.get('preprocessed_dir', 'data/preprocessed')) / 'train' / 'images'
                secondary_path = Path(ui_components.get('augmented_dir', 'data/augmented')) / 'images'
                
                # Cek mana yang memiliki file augmentasi
                aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components else 'aug'
                
                active_path = primary_path if primary_path.exists() else secondary_path
                visualize_augmented_samples(active_path, output_widget, ui_components, 5)
                
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi dataset: {str(e)}")
    
    # Handler untuk komparasi dataset
    def on_compare_click(b):
        """Handler untuk komparasi dataset dengan dynamic loading berdasarkan module_type."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi..."))
        
        try:
            # Dapatkan direktori dataset dan parameter berdasarkan module_type
            if module_type == 'preprocessing':
                # Import dan parameter untuk preprocessing
                from smartcash.ui.visualization.compare_original_vs_preprocessed import compare_original_vs_preprocessed
                compare_original_vs_preprocessed(
                    ui_components=ui_components,
                    raw_dir=ui_components.get('data_dir', 'data'),
                    preprocessed_dir=ui_components.get('preprocessed_dir', 'data/preprocessed')
                )
            else:
                # Import dan parameter untuk augmentation
                from smartcash.ui.visualization.compare_original_vs_augmented import compare_original_vs_augmented
                # Dapatkan lokasi original dan augmented
                original_path = Path(ui_components.get('preprocessed_dir', 'data/preprocessed')) / 'train' / 'images'
                # Cari lokasi yang memiliki data augmentasi
                primary_path = Path(ui_components.get('preprocessed_dir', 'data/preprocessed')) / 'train' / 'images'
                secondary_path = Path(ui_components.get('augmented_dir', 'data/augmented')) / 'images'
                
                active_path = primary_path if primary_path.exists() else secondary_path
                compare_original_vs_augmented(original_path, active_path, output_widget, ui_components)
                
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Handler untuk distribusi kelas
    def on_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Dapatkan direktori dataset berdasarkan module_type
            dataset_dir = ui_components.get('preprocessed_dir' if module_type == 'preprocessing' else 'augmented_dir', 'data/preprocessed')
            
            # Cek apakah direktori tersedia
            if not Path(dataset_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori tidak ditemukan: {dataset_dir}"))
                return
            
            # Import helper untuk distribusi kelas
            from smartcash.ui.visualization.visualization_integrator import create_distribution_visualizations
            
            # Buat wrapper untuk ui_components untuk visualization_integrator
            vis_ui_components = {
                "visualization_container": output_widget,
                "logger": logger, 
                "status": output_widget,
                "data_dir": ui_components.get('data_dir')
            }
            
            # Tambahkan aug_options jika module_type adalah augmentation
            if module_type == 'augmentation' and 'aug_options' in ui_components:
                vis_ui_components['aug_options'] = ui_components['aug_options']
            
            # Gunakan function distribusi kelas standard
            create_distribution_visualizations(
                ui_components=vis_ui_components,
                dataset_dir=dataset_dir,
                split_name='train',
                aug_prefix='aug',
                orig_prefix='rp',
                target_count=1000
            )
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error visualisasi distribusi kelas: {str(e)}")
    
    # Register handlers untuk tombol visualisasi
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_click)
    
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_click)
    
    if 'distribution_button' in ui_components:
        ui_components['distribution_button'].on_click(on_distribution_click)
    
    # Tambahkan handlers ke UI components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'on_distribution_click': on_distribution_click
    })
    
    return ui_components