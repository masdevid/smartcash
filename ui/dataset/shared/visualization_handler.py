"""
File: smartcash/ui/dataset/shared/visualization_handler.py
Deskripsi: Handler standar untuk setup visualisasi dataset dengan integrasi UI helpers
"""

from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output
import ipywidgets as widgets

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset dengan integrasi UI helpers.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    # Handler untuk visualisasi sampel
    def on_visualize_samples_click(b):
        """Handler untuk visualisasi sampel dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan visualisasi sampel..."))
        
        try:
            # Dapatkan direktori dataset dari berbagai kemungkinan key
            dataset_dir = ui_components.get('dataset_dir', 
                           ui_components.get('preprocessed_dir', 
                           ui_components.get('data_dir', 'data')))
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Import dan panggil handler visualisasi yang sesuai (diprioritaskan)
            try:
                # Opsi 1: Coba gunakan fungsi dari preprocessing_visualization_handler
                from smartcash.ui.visualization.visualize_preprocessed_samples import visualize_preprocessed_samples
                visualize_preprocessed_samples(ui_components, dataset_dir, ui_components.get('data_dir', 'data'))
            except ImportError:
                try:
                    # Opsi 2: Coba gunakan fungsi dari augmentation_visualization_handler
                    from smartcash.ui.visualization.visualize_augmented_samples import visualize_augmented_samples
                    visualize_augmented_samples(Path(dataset_dir) / 'train' / 'images', output_widget, ui_components)
                except ImportError:
                    # Opsi 3: Coba gunakan helper visualisasi dataset yang sudah ada
                    from smartcash.ui.helpers.dataset_visualizer import visualize_dataset_samples
                    visualize_dataset_samples(dataset_dir, output_widget=output_widget, split='train', num_samples=6)
            
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error visualisasi sampel: {str(e)}")
    
    # Handler untuk visualisasi distribusi kelas
    def on_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Dapatkan direktori dataset dari berbagai kemungkinan key
            dataset_dir = ui_components.get('dataset_dir', 
                           ui_components.get('preprocessed_dir', 
                           ui_components.get('data_dir', 'data')))
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Coba gunakan helper distribusi kelas yang sudah ada
            try:
                from smartcash.ui.visualization.visualization_integrator import create_distribution_visualizations
                create_distribution_visualizations(
                    ui_components={"visualization_container": output_widget, "logger": logger, 
                                  "status": output_widget, "data_dir": ui_components.get('data_dir')},
                    dataset_dir=dataset_dir,
                    split_name='train',
                    aug_prefix='aug',
                    orig_prefix='rp',
                    target_count=1000
                )
            except ImportError:
                try:
                    # Fallback ke visualisasi plot_class_distribution
                    from smartcash.ui.visualization.plot_single import plot_class_distribution
                    from smartcash.ui.visualization.class_distribution_analyzer import analyze_class_distribution
                    
                    # Analisis distribusi kelas
                    class_counts = analyze_class_distribution(dataset_dir, split='train')
                    
                    # Plot distribusi kelas
                    plot_class_distribution(class_counts, title="Distribusi Kelas Dataset", sort_by='value')
                except ImportError:
                    # Message jika modul visualisasi tidak tersedia
                    with output_widget:
                        display(create_status_indicator("warning", 
                                f"{ICONS['warning']} Modul visualisasi distribusi kelas tidak tersedia"))
                    
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error visualisasi distribusi kelas: {str(e)}")
    
    # Handler untuk komparasi dataset
    def on_compare_click(b):
        """Handler untuk komparasi dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan komparasi dataset..."))
        
        try:
            # Dapatkan direktori dataset dari berbagai kemungkinan key
            processed_dir = ui_components.get('output_dir', 
                            ui_components.get('preprocessed_dir', 
                            ui_components.get('augmented_dir', 'data/preprocessed')))
            
            original_dir = ui_components.get('input_dir', 
                           ui_components.get('data_dir', 'data'))
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Import dan panggil handler komparasi yang sesuai (diprioritaskan)
            try:
                # Opsi 1: Coba gunakan fungsi dari preprocessing_visualization_handler
                from smartcash.ui.visualization.compare_original_vs_preprocessed import compare_original_vs_preprocessed
                compare_original_vs_preprocessed(ui_components, original_dir, processed_dir)
            except ImportError:
                try:
                    # Opsi 2: Coba gunakan fungsi dari augmentation_visualization_handler
                    from smartcash.ui.visualization.compare_original_vs_augmented import compare_original_vs_augmented
                    compare_original_vs_augmented(Path(original_dir) / 'train' / 'images', 
                                                Path(processed_dir) / 'train' / 'images', 
                                                output_widget, ui_components)
                except ImportError:
                    try:
                        # Opsi 3: Coba gunakan helper komparasi dataset yang sudah ada
                        from smartcash.ui.visualization.plot_comparison import plot_class_distribution_comparison
                        from smartcash.ui.visualization.class_distribution_analyzer import analyze_class_distribution
                        
                        # Analisis distribusi kelas untuk kedua dataset
                        original_counts = analyze_class_distribution(original_dir, split='train')
                        processed_counts = analyze_class_distribution(processed_dir, split='train')
                        
                        # Plot komparasi distribusi kelas
                        plot_class_distribution_comparison(
                            original_counts, 
                            processed_counts, 
                            title="Komparasi Distribusi Kelas", 
                            label1="Original", 
                            label2="Processed"
                        )
                    except ImportError:
                        # Message jika modul komparasi tidak tersedia
                        with output_widget:
                            display(create_status_indicator("warning", 
                                    f"{ICONS['warning']} Modul komparasi dataset tidak tersedia"))
            
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error komparasi dataset: {str(e)}")
    
    # Register handlers ke tombol jika tersedia
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_samples_click)
    
    if 'distribution_button' in ui_components:
        ui_components['distribution_button'].on_click(on_distribution_click)
    
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_click)
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_visualize_samples_click': on_visualize_samples_click,
        'on_distribution_click': on_distribution_click,
        'on_compare_click': on_compare_click
    })
    
    if logger: logger.debug(f"{ICONS['success']} Visualization handlers berhasil diinisialisasi")
    
    return ui_components