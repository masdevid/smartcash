"""
File: smartcash/ui/dataset/visualization_integrator.py
Deskripsi: Modul integrasi visualisasi distribusi kelas
"""

from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output, HTML
from smartcash.ui.visualization.get_preprocessing_stats import get_preprocessing_stats
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import utils dengan pendekatan DRY
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

# Import fungsi analisis dari helpers
from smartcash.ui.visualization.class_distribution_analyzer import (
    analyze_class_distribution_by_prefix,
    count_files_by_prefix
)
from smartcash.ui.visualization.distribution_summary_display import (
    display_distribution_summary,
    display_file_summary
)
from smartcash.ui.visualization.plot_single import plot_class_distribution
from smartcash.ui.visualization.plot_comparison import plot_class_distribution_comparison
from smartcash.ui.visualization.plot_stacked import plot_class_distribution_stacked

# Gunakan throttling untuk fungsi logging dengan state closure
def create_throttled_function(interval: float = 1.0):
    """Buat fungsi throttling logger untuk mencegah flooding log."""
    last_call_time = {'value': 0}
    
    def throttled(func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call_time['value'] >= interval:
                last_call_time['value'] = current_time
                return func(*args, **kwargs)
            # Skip jika di-throttled
        return wrapper
    return throttled

def create_distribution_visualizations(
    ui_components: Dict[str, Any],
    dataset_dir: str,
    split_name: str = 'train',
    aug_prefix: str = 'aug',
    orig_prefix: str = 'rp',
    target_count: int = 1000
):
    """
    Buat dan tampilkan visualisasi distribusi kelas
    
    Args:
        ui_components: Dictionary berisi komponen UI
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        aug_prefix: Prefix untuk file augmentasi
        orig_prefix: Prefix untuk file original
        target_count: Target jumlah instance per kelas
    """
    logger = ui_components.get('logger')
    output_widget = ui_components.get('visualization_container', ui_components.get('status'))
    
    if not output_widget:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Output widget tidak tersedia untuk visualisasi")
        return
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS.get('chart', '📊')} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Analisis distribusi kelas dengan satu baris
            all_counts, orig_counts, aug_counts = analyze_class_distribution_by_prefix(
                dataset_dir, split_name, aug_prefix, orig_prefix)
            prefix_counts = count_files_by_prefix(dataset_dir, split_name)
            
            # Validasi dan tampilkan pesan error jika tidak ada data
            if not all_counts and not prefix_counts:
                display(create_info_alert(
                    f"Tidak ditemukan data distribusi kelas di {dataset_dir}/{split_name}",
                    "warning"))
                return
            
            # Tampilkan judul dengan HTML semantic styling
            display(HTML(f"""
            <h3 style="color: {COLORS.get('dark', '#2F58CD')};">{ICONS.get('chart', '📊')} Visualisasi Distribusi Kelas</h3>
            <p>Analisis distribusi kelas pada split <strong>{split_name}</strong> dengan target balancing <strong>{target_count}</strong> instance per kelas.</p>
            """))
            
            # Konsolidasi visualisasi data dengan conditional logic
            if prefix_counts: display_file_summary(prefix_counts)
            
            # Tampilkan ringkasan kelas jika ada data
            if all_counts:
                # Tampilkan ringkasan dan semua visualisasi dalam urutan logis
                display_distribution_summary(all_counts, orig_counts, aug_counts, target_count)
                
                # Gunakan pipeline visualisasi dengan conditional execution untuk setiap jenis plot
                visualization_pipeline = [
                    # (condition, plot_function, title, kwargs)
                    (bool(orig_counts), plot_class_distribution, 
                     f"Distribusi Kelas pada Data Asli (total: {sum(orig_counts.values() or [0])} instance)",
                     {"color": COLORS.get('primary', '#3498db')}),
                    
                    (bool(aug_counts), plot_class_distribution,
                     f"Distribusi Kelas pada Data Augmentasi (total: {sum(aug_counts.values() or [0])} instance)",
                     {"color": COLORS.get('warning', '#f39c12')}),
                    
                    (bool(orig_counts and aug_counts), plot_class_distribution_comparison,
                     None, {}),
                    
                    (bool(orig_counts or aug_counts), plot_class_distribution_stacked,
                     f"Total Distribusi Kelas Setelah Augmentasi (total: {sum(all_counts.values() or [0])} instance)",
                     {})
                ]
                
                # Execute pipeline dengan data yang tepat
                for condition, plot_func, title, kwargs in visualization_pipeline:
                    if not condition: continue
                    if title: kwargs['title'] = title
                    
                    # Pilih data yang tepat untuk plot dan tampilkan
                    if plot_func == plot_class_distribution_comparison:
                        fig = plot_func(orig_counts, aug_counts, **kwargs)
                    elif plot_func == plot_class_distribution_stacked:
                        fig = plot_func(orig_counts or {}, aug_counts or {}, **kwargs)
                    elif "Data Asli" in (title or ""): 
                        fig = plot_func(orig_counts, **kwargs)
                    elif "Augmentasi" in (title or ""):
                        fig = plot_func(aug_counts, **kwargs)
                    else:
                        fig = plot_func(all_counts, **kwargs)
                    
                    plt.show()
            
            # Tampilkan container dan tombol visualisasi
            output_widget.layout.display = 'block'
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
                
        except Exception as e:
            if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi: {str(e)}")
            display(create_info_alert(f"Error saat membuat visualisasi: {str(e)}", "error"))

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None, context: str = "") -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi distribusi kelas dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        context: Konteks visualisasi ('preprocessing', 'augmentation', dll)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi common untuk mendapatkan informasi visualisasi dari context
    def get_context_info(context_name: str) -> Tuple[str, str, int]:
        """Dapatkan info berdasarkan context untuk code konsolidasi."""
        # Default values
        dataset_dir, target_count = 'data/preprocessed', 1000
        
        # Dapatkan data dir berdasarkan context
        context_map = {
            'preprocessing': ('data/preprocessed', f"Distribusi Kelas setelah Preprocessing", 1000),
            'augmentation': ('data/augmented', f"Distribusi Kelas setelah Augmentasi", 1500),
            'training': ('data', f"Distribusi Kelas Dataset Training", 1000),
        }
        
        # Ekstrak dari config jika tersedia
        if config:
            if context_name == 'preprocessing':
                dataset_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            elif context_name == 'augmentation':
                dataset_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
        
        # Return tuple (dir, title, target)
        return context_map.get(context_name, (dataset_dir, 'Distribusi Kelas Dataset', target_count))
    
    # Consolidated handler untuk visualisasi
    def on_show_distribution_click(b):
        """Handler terkonsolidasi untuk visualisasi distribusi kelas."""
        # Dapatkan info berdasarkan context
        dataset_dir, title, target_count = get_context_info(context)
        
        # Ekstrak prefix dari ui_components atau config
        aug_prefix, orig_prefix = "aug", "rp"
        
        # Dapatkan prefix dari UI komponen jika tersedia dengan conditional one-liner
        if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 2:
            aug_prefix = ui_components['aug_options'].children[2].value
        
        # Tampilkan visualisasi
        create_distribution_visualizations(
            ui_components=ui_components,
            dataset_dir=dataset_dir,
            split_name='train',
            aug_prefix=aug_prefix,
            orig_prefix=orig_prefix,
            target_count=target_count
        )
    
    # Tambahkan tombol distribusi dengan pendekatan DRY
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'children'):
        # Dapatkan atau buat tombol distribusi dengan inline conditional
        distribution_button = next((child for child in visualization_buttons.children 
                                   if hasattr(child, 'description') and 'Distribusi Kelas' in child.description), None)
        
        # Buat tombol baru jika tidak ditemukan
        if not distribution_button:
            # Import dan buat tombol dalam satu blok
            from ipywidgets import Button, Layout
            distribution_button = Button(
                description='Distribusi Kelas',
                button_style='info',
                icon='bar-chart',
                tooltip='Tampilkan distribusi kelas dataset',
                layout=Layout(margin='5px 0')
            )
            
            # Tambahkan ke container dengan extended children
            visualization_buttons.children = list(visualization_buttons.children) + [distribution_button]
        
        # Register handler
        distribution_button.on_click(on_show_distribution_click)
        ui_components.update({
            'distribution_button': distribution_button,
            'on_show_distribution_click': on_show_distribution_click
        })
        
        if logger: logger.info(f"{ICONS.get('success', '✅')} Handler visualisasi distribusi kelas berhasil ditambahkan")
    
    return ui_components

def analyze_preprocessing_effect(ui_components: Dict[str, Any], dataset_dir: str) -> None:
    """
    Analisis singkat efek preprocessing terhadap distribusi kelas.
    
    Args:
        ui_components: Dictionary komponen UI
        dataset_dir: Path direktori dataset
    """
    # Panggil visualisasi standar
    create_distribution_visualizations(
        ui_components=ui_components,
        dataset_dir=dataset_dir,
        split_name='train',
        aug_prefix='aug',
        orig_prefix='rp',
        target_count=1000
    )