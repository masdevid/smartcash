"""
File: smartcash/ui/dataset/split/components/ui_form.py
Deskripsi: Komponen form untuk split dataset - refactored dengan DRY approach
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import create_save_reset_buttons, create_status_panel, create_section_title


def create_split_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat komponen form untuk split dataset dengan one-liner style"""
    split_ratios = config.get('data', {}).get('split_ratios', {})
    split_settings = config.get('split_settings', {})
    data_config = config.get('data', {})
    
    # One-liner slider creation
    create_ratio_slider = lambda value, min_val, max_val, desc: widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=0.05, description=desc,
        readout_format='.2f', layout=widgets.Layout(width='90%')
    )
    
    # One-liner input creation  
    create_text_input = lambda value, desc, width='90%': widgets.Text(
        value=value, description=desc, layout=widgets.Layout(width=width)
    )
    
    create_checkbox = lambda value, desc: widgets.Checkbox(value=value, description=desc, indent=False)
    create_int_input = lambda value, desc: widgets.IntText(value=value, description=desc, layout=widgets.Layout(width='50%'))
    
    # Form components dengan consolidated creation
    form_components = {
        'train_slider': create_ratio_slider(split_ratios.get('train', 0.7), 0.5, 0.9, 'Train:'),
        'valid_slider': create_ratio_slider(split_ratios.get('valid', 0.15), 0.05, 0.3, 'Valid:'),
        'test_slider': create_ratio_slider(split_ratios.get('test', 0.15), 0.05, 0.3, 'Test:'),
        'total_label': widgets.HTML(
            value=f"<div style='padding: 10px; color: {COLORS['success']}; font-weight: bold;'>Total: 1.00</div>",
            layout=widgets.Layout(width='90%')
        ),
        'stratified_checkbox': create_checkbox(data_config.get('stratified_split', True), 'Stratified Split'),
        'backup_checkbox': create_checkbox(split_settings.get('backup_before_split', True), 'Backup Sebelum Split'),
        'random_seed': create_int_input(data_config.get('random_seed', 42), 'Random Seed:'),
        'dataset_path': create_text_input(split_settings.get('dataset_path', 'data'), 'Dataset Path:'),
        'preprocessed_path': create_text_input(split_settings.get('preprocessed_path', 'data/preprocessed'), 'Preprocessed:'),
        'backup_dir': create_text_input(split_settings.get('backup_dir', 'data/splits_backup'), 'Backup Dir:')
    }
    
    # Reuse existing shared components
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi split dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi ke nilai default",
        with_sync_info=True
    )
    
    status_panel = create_status_panel("Status konfigurasi akan ditampilkan di sini", "info")
    
    # Merge all components
    form_components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'], 
        'save_reset_container': save_reset_buttons['container'],
        'status_panel': status_panel
    })
    
    return form_components


def create_ratio_section(components: Dict[str, Any]) -> widgets.VBox:
    """Buat section untuk ratio split dengan existing header"""
    
    return widgets.VBox([
        create_section_title("Ratio Split Dataset", ICONS.get('split', '✂️')),
        components['train_slider'], components['valid_slider'], 
        components['test_slider'], components['total_label'], 
        components['stratified_checkbox'], components['random_seed']
    ], layout=widgets.Layout(width='100%', padding='5px'))


def create_path_section(components: Dict[str, Any]) -> widgets.VBox:
    """Buat section untuk path dan backup dengan existing header"""
    
    return widgets.VBox([
        create_section_title("Path dan Backup", ICONS.get('folder', '📁')),
        components['dataset_path'], components['preprocessed_path'],
        components['backup_checkbox'], components['backup_dir']
    ], layout=widgets.Layout(width='100%', padding='5px'))