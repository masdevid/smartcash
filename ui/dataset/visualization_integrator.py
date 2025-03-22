"""
File: smartcash/ui/dataset/visualization_integrator.py
Deskripsi: Modul utama untuk integrasi visualisasi distribusi kelas dalam UI
"""

from typing import Dict, Any
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

from smartcash.ui.helpers.class_distribution_analyzer import (
    analyze_class_distribution_by_prefix,
    count_files_by_prefix
)
from smartcash.ui.helpers.distribution_summary_display import (
    display_distribution_summary,
    display_file_summary
)
from smartcash.ui.helpers.plot_single import plot_class_distribution
from smartcash.ui.helpers.plot_comparison import plot_class_distribution_comparison
from smartcash.ui.helpers.plot_stacked import plot_class_distribution_stacked

def create_distribution_visualizations(
    ui_components: Dict[str, Any],
    dataset_dir: str,
    split_name: str = 'train',
    aug_prefix: str = 'aug',
    orig_prefix: str = 'rp',
    target_count: int = 1000
):
    """
    Buat dan tampilkan semua visualisasi distribusi kelas.
    
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
            # Analisis distribusi kelas
            all_counts, orig_counts, aug_counts = analyze_class_distribution_by_prefix(
                dataset_dir, split_name, aug_prefix, orig_prefix
            )
            
            # Analisis jumlah file berdasarkan prefix
            prefix_counts = count_files_by_prefix(dataset_dir, split_name)
            
            if not all_counts and not prefix_counts:
                display(create_info_alert(
                    f"Tidak ditemukan data distribusi kelas di {dataset_dir}/{split_name}",
                    "warning"
                ))
                return
            
            # Tampilkan judul
            display(HTML(f"""
            <h3 style="color: {COLORS.get('dark', '#2F58CD')};">{ICONS.get('chart', '📊')} Visualisasi Distribusi Kelas</h3>
            <p>Analisis distribusi kelas pada split <strong>{split_name}</strong> dengan target balancing <strong>{target_count}</strong> instance per kelas.</p>
            """))
            
            # Tampilkan ringkasan file
            if prefix_counts:
                display_file_summary(prefix_counts)
            
            # Tampilkan ringkasan kelas
            if all_counts:
                display_distribution_summary(all_counts, orig_counts, aug_counts, target_count)
                
                # Plot distribusi asli
                if orig_counts:
                    fig1 = plot_class_distribution(
                        orig_counts,
                        title=f"Distribusi Kelas pada Data Asli (total: {sum(orig_counts.values())} instance)",
                        color=COLORS.get('primary', '#3498db')
                    )
                    plt.show()
                
                # Plot distribusi augmentasi (jika ada)
                if aug_counts:
                    fig2 = plot_class_distribution(
                        aug_counts,
                        title=f"Distribusi Kelas pada Data Augmentasi (total: {sum(aug_counts.values())} instance)",
                        color=COLORS.get('warning', '#f39c12')
                    )
                    plt.show()
                
                # Plot perbandingan side-by-side
                if orig_counts and aug_counts:
                    fig3 = plot_class_distribution_comparison(orig_counts, aug_counts)
                    plt.show()
                
                # Plot distribusi total (stacked)
                if orig_counts or aug_counts:
                    fig4 = plot_class_distribution_stacked(
                        orig_counts, aug_counts,
                        title=f"Total Distribusi Kelas Setelah Augmentasi (total: {sum(all_counts.values())} instance)"
                    )
                    plt.show()
            
            # Tampilkan container
            output_widget.layout.display = 'block'
            
            # Tambahkan tombol tampilkan visualisasi
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
                
        except Exception as e:
            if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi: {str(e)}")
            display(create_info_alert(f"Error saat membuat visualisasi: {str(e)}", "error"))

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk menampilkan visualisasi distribusi kelas dalam UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi distribusi kelas
    def on_show_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas."""
        # Dapatkan lokasi data dari config atau UI
        config_obj = config or ui_components.get('config', {})
        dataset_dir = config_obj.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        
        # Dapatkan prefix untuk file
        aug_prefix = "aug"
        orig_prefix = "rp"
        
        if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
            aug_prefix = ui_components['aug_options'].children[2].value
            
        # Dapatkan target count
        target_count = 1000
        
        # Tampilkan visualisasi
        create_distribution_visualizations(
            ui_components=ui_components,
            dataset_dir=dataset_dir,
            split_name='train',
            aug_prefix=aug_prefix,
            orig_prefix=orig_prefix,
            target_count=target_count
        )
    
    # Register handler jika visualization buttons tersedia
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'children') and len(visualization_buttons.children) > 0:
        # Connect to the first button or create a new one if needed
        if len(visualization_buttons.children) >= 3:
            # If there's already a distribution button, use it
            distribution_button = visualization_buttons.children[2]
        else:
            # Create a new button and add it to visualization_buttons
            from ipywidgets import Button
            distribution_button = Button(
                description='Distribusi Kelas',
                button_style='info',
                icon='bar-chart',
                tooltip='Tampilkan distribusi kelas dataset'
            )
            
            # Add button to container
            import ipywidgets as widgets
            new_children = list(visualization_buttons.children) + [distribution_button]
            visualization_buttons.children = tuple(new_children)
        
        # Register click handler
        distribution_button.on_click(on_show_distribution_click)
        
        # Add reference to UI components
        ui_components['distribution_button'] = distribution_button
        ui_components['on_show_distribution_click'] = on_show_distribution_click
        
        if logger: logger.info(f"{ICONS.get('success', '✅')} Handler visualisasi distribusi kelas berhasil ditambahkan")
    
    return ui_components