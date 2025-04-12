"""
File: smartcash/ui/dataset/preprocessing/handlers/visualization_handler.py
Deskripsi: Handler visualisasi untuk preprocessing dataset
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk visualisasi sampel dataset
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel preprocessing."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi..."))
        
        try:
            # Import dan parameter untuk preprocessing
            from smartcash.ui.charts.visualize_preprocessed_samples import visualize_preprocessed_samples
            
            # Dapatkan paths dari UI components
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            original_dir = ui_components.get('data_dir', 'data')
            
            # Jalankan visualisasi
            visualize_preprocessed_samples(
                ui_components=ui_components,
                preprocessed_dir=preprocessed_dir,
                original_dir=original_dir,
                num_samples=5
            )
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi dataset: {str(e)}")
    
    # Handler untuk komparasi dataset
    def on_compare_click(b):
        """Handler untuk komparasi dataset original vs preprocessing."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi..."))
        
        try:
            # Import dan parameter untuk preprocessing
            from smartcash.ui.charts.compare_original_vs_preprocessed import compare_original_vs_preprocessed
            
            # Dapatkan paths dari UI components
            raw_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Jalankan komparasi
            compare_original_vs_preprocessed(
                ui_components=ui_components,
                raw_dir=raw_dir,
                preprocessed_dir=preprocessed_dir
            )
            
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
            # Dapatkan direktori preprocessed
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek apakah direktori tersedia
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Import helper untuk distribusi kelas
            from smartcash.ui.charts.visualization_integrator import create_distribution_visualizations
            
            # Buat wrapper untuk ui_components untuk visualization_integrator
            vis_ui_components = {
                "visualization_container": output_widget,
                "logger": logger, 
                "status": output_widget,
                "data_dir": ui_components.get('data_dir')
            }
            
            # Gunakan function distribusi kelas standard
            create_distribution_visualizations(
                ui_components=vis_ui_components,
                dataset_dir=preprocessed_dir,
                split_name='train',
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
    visualization_handlers = {
        'visualize_button': on_visualize_click,
        'compare_button': on_compare_click,
        'distribution_button': on_distribution_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in visualization_handlers.items() if button in ui_components]
    
    # Tambahkan handlers ke UI components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'on_distribution_click': on_distribution_click
    })
    
    return ui_components