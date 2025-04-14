"""
File: smartcash/ui/dataset/augmentation/handlers/visualization_handler.py
Deskripsi: Handler visualisasi untuk augmentasi dataset
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset augmentasi.
    
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
        """Handler untuk visualisasi sampel augmentasi."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi sampel augmentasi..."))
        
        try:
            # Dapatkan parameter untuk augmentasi
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            
            # Cek lokasi sampel augmentasi (prioritas ke preprocessed)
            train_images_dir = Path(preprocessed_dir) / 'train' / 'images'
            augmented_images_dir = Path(augmented_dir) / 'images'
            
            # Pilih lokasi yang memiliki sampel
            if train_images_dir.exists() and list(train_images_dir.glob(f"{aug_prefix}_*.jpg")):
                target_dir = train_images_dir
            elif augmented_images_dir.exists() and list(augmented_images_dir.glob(f"{aug_prefix}_*.jpg")):
                target_dir = augmented_images_dir
            else:
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file augmentasi dengan prefix '{aug_prefix}'"))
                return
            
            # Import dan visualisasi
            try:
                from smartcash.ui.charts.visualize_augmented_samples import visualize_augmented_samples
                visualize_augmented_samples(target_dir, output_widget, ui_components, 5)
            except ImportError:
                # Fallback: Gunakan visualisasi standar dari shared
                from smartcash.ui.dataset.shared.visualization_handler import _visualize_augmented_samples
                _visualize_augmented_samples(output_widget, 5)
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi sampel augmentasi: {str(e)}")
    
    # Handler untuk komparasi dataset
    def on_compare_click(b):
        """Handler untuk komparasi dataset original vs augmentasi."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi original vs augmentasi..."))
        
        try:
            # Dapatkan parameter untuk augmentasi
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            
            # Cek lokasi sampel (original dari train, augmented dari train atau augmented_dir)
            original_dir = Path(preprocessed_dir) / 'train' / 'images'
            
            # Cek lokasi file augmentasi (prioritas ke preprocessed/train)
            train_images_dir = Path(preprocessed_dir) / 'train' / 'images'
            augmented_images_dir = Path(augmented_dir) / 'images'
            
            # Pilih lokasi file augmentasi yang memiliki sampel
            if train_images_dir.exists() and list(train_images_dir.glob(f"{aug_prefix}_*.jpg")):
                augmented_dir = train_images_dir
            elif augmented_images_dir.exists() and list(augmented_images_dir.glob(f"{aug_prefix}_*.jpg")):
                augmented_dir = augmented_images_dir
            else:
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file augmentasi dengan prefix '{aug_prefix}'"))
                return
            
            # Cek juga file original
            if not original_dir.exists() or not list(original_dir.glob("rp_*.jpg")):
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file original (rp_*) di {original_dir}"))
                return
            
            # Import dan visualisasi
            try:
                from smartcash.ui.charts.compare_original_vs_augmented import compare_original_vs_augmented
                compare_original_vs_augmented(original_dir, augmented_dir, output_widget, ui_components)
            except ImportError:
                # Fallback: Gunakan visualisasi standar dari shared
                from smartcash.ui.dataset.shared.visualization_handler import _compare_original_vs_augmented
                _compare_original_vs_augmented(output_widget)
            
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
            # Dapatkan parameter
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            
            # Cek direktori
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Import helper untuk distribusi kelas
            try:
                from smartcash.ui.charts.visualization_integrator import create_distribution_visualizations
                
                # Buat wrapper untuk ui_components
                vis_ui_components = {
                    "visualization_container": output_widget,
                    "logger": logger, 
                    "status": output_widget,
                    "data_dir": ui_components.get('data_dir'),
                    "aug_options": ui_components.get('aug_options')
                }
                
                # Visualisasi distribusi kelas
                create_distribution_visualizations(
                    ui_components=vis_ui_components,
                    dataset_dir=preprocessed_dir,
                    split_name='train',
                    aug_prefix=aug_prefix,
                    orig_prefix='rp',
                    target_count=1000
                )
            except ImportError:
                # Fallback: Gunakan visualisasi standar dari shared
                from smartcash.ui.dataset.shared.visualization_handler import _visualize_class_distribution
                _visualize_class_distribution(output_widget)
            
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