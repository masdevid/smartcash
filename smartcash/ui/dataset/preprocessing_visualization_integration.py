"""
File: smartcash/ui/dataset/preprocessing_visualization_integration.py
Deskripsi: Integrasi visualisasi untuk modul preprocessing dengan fokus pada distribusi kelas
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
from pathlib import Path
import time

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler visualisasi untuk preprocessing dengan integrasi yang ditingkatkan."""
    
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS, COLORS
    
    # Handler untuk visualisasi sampel
    def on_visualize_samples_click(b):
        """Visualisasikan sampel dataset yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan visualisasi sampel..."))
        
        try:
            # Dapatkan direktori preprocessed
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek apakah direktori tersedia
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Direktori preprocessed tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Import fungsi visualisasi
            from smartcash.ui.visualization.visualize_preprocessed_sample import visualize_preprocessed_sample
            
            # Visualisasikan sampel
            visualize_preprocessed_sample(
                ui_components={"status": output_widget, "logger": logger},
                preprocessed_dir=preprocessed_dir,
                original_dir=ui_components.get('data_dir', 'data'),
                num_samples=5
            )
            
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error visualisasi sampel: {str(e)}")
    
    # Handler untuk perbandingan dataset
    def on_compare_dataset_click(b):
        """Bandingkan sampel dataset mentah dengan yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan perbandingan dataset..."))
        
        try:
            # Dapatkan direktori
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek ketersediaan direktori
            if not Path(data_dir).exists() or not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Direktori dataset tidak lengkap untuk perbandingan"))
                return
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Import fungsi perbandingan
            from smartcash.ui.visualization.compare_raw_vs_preprocessed import compare_raw_vs_preprocessed
            
            # Visualisasikan perbandingan
            compare_raw_vs_preprocessed(
                ui_components={"status": output_widget, "logger": logger},
                raw_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
                num_samples=3
            )
            
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat perbandingan: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error perbandingan dataset: {str(e)}")
    
    # Handler untuk distribusi kelas
    def on_distribution_click(b):
        """Visualisasikan distribusi kelas dataset."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Dapatkan direktori preprocessed
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek apakah direktori tersedia
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Direktori preprocessed tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Tampilkan container visualisasi
            ui_components['visualization_container'].layout.display = 'block'
            
            # Import helper untuk distribusi kelas
            from smartcash.ui.visualization.visualization_integrator import create_distribution_visualizations
            
            # Gunakan function distribusi kelas standard dari visualization_integrator dengan wrapper yang terkonsolidasi
            create_distribution_visualizations(
                ui_components={"visualization_container": output_widget, "logger": logger, 
                               "status": output_widget, "data_dir": ui_components.get('data_dir')},
                dataset_dir=preprocessed_dir,
                split_name='train',
                aug_prefix='aug',
                orig_prefix='rp',
                target_count=1000
            )
            
        except Exception as e:
            with output_widget:
                display(create_status_indicator("error", f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error visualisasi distribusi kelas: {str(e)}")
    
    # Register handlers ke tombol
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_samples_click)
    
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_dataset_click)
    
    if 'distribution_button' in ui_components:
        ui_components['distribution_button'].on_click(on_distribution_click)
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_visualize_samples_click': on_visualize_samples_click,
        'on_compare_dataset_click': on_compare_dataset_click,
        'on_distribution_click': on_distribution_click
    })
    
    return ui_components

def analyze_preprocessing_distribution(preprocessed_dir: str, ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analisis distribusi kelas dataset preprocessed dengan pendekatan DRY.
    
    Args:
        preprocessed_dir: Direktori dataset preprocessed
        ui_components: Komponen UI untuk output
        
    Returns:
        Dictionary statistik distribusi kelas
    """
    try:
        from smartcash.ui.helpers.class_distribution_analyzer import analyze_class_distribution_by_prefix
        
        # Analisis dengan parameter default
        all_counts, orig_counts, aug_counts = analyze_class_distribution_by_prefix(
            preprocessed_dir, 'train', 'aug', 'rp')
        
        # Buat ringkasan
        total_instances = sum(all_counts.values()) if all_counts else 0
        class_count = len(all_counts) if all_counts else 0
        min_instances = min(all_counts.values()) if all_counts and all_counts.values() else 0
        max_instances = max(all_counts.values()) if all_counts and all_counts.values() else 0
        
        return {
            'all_counts': all_counts,
            'orig_counts': orig_counts,
            'aug_counts': aug_counts,
            'total_instances': total_instances,
            'class_count': class_count,
            'min_instances': min_instances,
            'max_instances': max_instances,
            'balance_ratio': min_instances / max_instances if max_instances > 0 else 0
        }
    except Exception as e:
        if ui_components and 'logger' in ui_components:
            ui_components['logger'].warning(f"{ICONS['warning']} Error analisis distribusi: {str(e)}")
        return {
            'error': str(e),
            'all_counts': {},
            'orig_counts': {},
            'aug_counts': {},
            'total_instances': 0
        }