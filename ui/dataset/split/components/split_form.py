"""
File: smartcash/ui/dataset/split/components/split_form.py
Deskripsi: Komponen form untuk konfigurasi split dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons


def create_split_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen form untuk split dataset.
    
    Args:
        config: Konfigurasi untuk mengisi nilai default
        
    Returns:
        Dict berisi komponen form
    """
    # Ekstrak nilai dari config
    split_ratios = config.get('data', {}).get('split_ratios', {})
    split_settings = config.get('split_settings', {})
    
    # Slider untuk ratio
    train_slider = widgets.FloatSlider(
        value=split_ratios.get('train', 0.7),
        min=0.5, max=0.9, step=0.05,
        description='Train:',
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    
    valid_slider = widgets.FloatSlider(
        value=split_ratios.get('valid', 0.15),
        min=0.05, max=0.3, step=0.05,
        description='Valid:',
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    
    test_slider = widgets.FloatSlider(
        value=split_ratios.get('test', 0.15),
        min=0.05, max=0.3, step=0.05,
        description='Test:',
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    
    # Label total
    total_label = widgets.HTML(
        value=f"<div style='padding: 10px; color: {COLORS['success']}; font-weight: bold;'>Total: 1.00</div>",
        layout=widgets.Layout(width='90%')
    )
    
    # Checkbox options
    stratified_checkbox = widgets.Checkbox(
        value=config.get('data', {}).get('stratified_split', True),
        description='Stratified Split',
        indent=False
    )
    
    backup_checkbox = widgets.Checkbox(
        value=split_settings.get('backup_before_split', True),
        description='Backup Sebelum Split',
        indent=False
    )
    
    # Input fields
    random_seed = widgets.IntText(
        value=config.get('data', {}).get('random_seed', 42),
        description='Random Seed:',
        layout=widgets.Layout(width='50%')
    )
    
    dataset_path = widgets.Text(
        value=split_settings.get('dataset_path', 'data'),
        description='Dataset Path:',
        layout=widgets.Layout(width='90%')
    )
    
    preprocessed_path = widgets.Text(
        value=split_settings.get('preprocessed_path', 'data/preprocessed'),
        description='Preprocessed:',
        layout=widgets.Layout(width='90%')
    )
    
    backup_dir = widgets.Text(
        value=split_settings.get('backup_dir', 'data/splits_backup'),
        description='Backup Dir:',
        layout=widgets.Layout(width='90%')
    )
    
    # Status panel
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS['info']} Status konfigurasi akan ditampilkan di sini</p>
        </div>"""
    )
    
    # Save/Reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset", 
        save_tooltip="Simpan konfigurasi split dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi ke nilai default",
        with_sync_info=True
    )
    
    return {
        'train_slider': train_slider,
        'valid_slider': valid_slider,
        'test_slider': test_slider,
        'total_label': total_label,
        'stratified_checkbox': stratified_checkbox,
        'backup_checkbox': backup_checkbox,
        'random_seed': random_seed,
        'dataset_path': dataset_path,
        'preprocessed_path': preprocessed_path,
        'backup_dir': backup_dir,
        'status_panel': status_panel,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_container': save_reset_buttons['container']
    }


def create_ratio_section(components: Dict[str, Any]) -> widgets.VBox:
    """Buat section untuk ratio split."""
    return widgets.VBox([
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['split']} Ratio Split Dataset</h4>"),
        components['train_slider'],
        components['valid_slider'],
        components['test_slider'],
        components['total_label'],
        components['stratified_checkbox'],
        components['random_seed']
    ], layout=widgets.Layout(width='100%', padding='5px'))


def create_path_section(components: Dict[str, Any]) -> widgets.VBox:
    """Buat section untuk path dan backup."""
    return widgets.VBox([
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['folder']} Path dan Backup</h4>"),
        components['dataset_path'],
        components['preprocessed_path'],
        components['backup_checkbox'],
        components['backup_dir']
    ], layout=widgets.Layout(width='100%', padding='5px'))