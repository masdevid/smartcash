"""
File: smartcash/ui/dataset/split/components/split_components.py
Deskripsi: Komponen UI untuk konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display

def create_split_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi split dataset.
    
    Args:
        config: Konfigurasi dataset
        
    Returns:
        Dict berisi komponen UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, create_divider
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.components.sync_info_message import create_sync_info_message
    
    # Inisialisasi komponen
    ui_components = {}
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi split dataset</p>
        </div>"""
    )
    ui_components['status_panel'] = status_panel
    
    # Buat header
    header = create_header(
        title="Konfigurasi Split Dataset",
        description="Konfigurasi pembagian dataset menjadi train, validation, dan test"
    )
    
    # Buat slider untuk train ratio
    train_slider = widgets.FloatSlider(
        value=0.7,
        min=0.5,
        max=0.9,
        step=0.05,
        description='Train:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    ui_components['train_slider'] = train_slider
    
    # Buat slider untuk validation ratio
    val_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Val:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    ui_components['val_slider'] = val_slider
    
    # Buat slider untuk test ratio
    test_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Test:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='90%')
    )
    ui_components['test_slider'] = test_slider
    
    # Buat label untuk total ratio
    total_label = widgets.HTML(
        value=f"<div style='padding: 10px; color: {COLORS['primary']}; font-weight: bold;'>Total: 1.00</div>",
        layout=widgets.Layout(width='90%')
    )
    ui_components['total_label'] = total_label
    
    # Buat checkbox untuk stratified split
    stratified_checkbox = widgets.Checkbox(
        value=True,
        description='Stratified Split',
        disabled=False,
        indent=False
    )
    ui_components['stratified_checkbox'] = stratified_checkbox
    
    # Buat input untuk random seed
    random_seed = widgets.IntText(
        value=42,
        description='Random Seed:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    ui_components['random_seed'] = random_seed
    
    # Buat checkbox untuk backup sebelum split
    backup_checkbox = widgets.Checkbox(
        value=True,
        description='Backup Sebelum Split',
        disabled=False,
        indent=False
    )
    ui_components['backup_checkbox'] = backup_checkbox
    
    # Buat input untuk direktori backup
    backup_dir = widgets.Text(
        value='data/splits_backup',
        description='Backup Dir:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    ui_components['backup_dir'] = backup_dir
    
    # Buat input untuk path dataset
    dataset_path = widgets.Text(
        value='data',
        description='Dataset Path:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    ui_components['dataset_path'] = dataset_path
    
    # Buat input untuk path preprocessed
    preprocessed_path = widgets.Text(
        value='data/preprocessed',
        description='Preprocessed:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    ui_components['preprocessed_path'] = preprocessed_path
    
    # Buat tombol save dan reset menggunakan shared component
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi split dataset dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi split dataset ke default",
        save_icon="save",
        reset_icon="reset",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        button_width="100px"
    )
    
    # Tambahkan referensi ke ui_components
    ui_components['save_button'] = save_reset_buttons['save_button']
    ui_components['reset_button'] = save_reset_buttons['reset_button']
    ui_components['save_reset_buttons'] = save_reset_buttons
    ui_components['sync_info'] = save_reset_buttons.get('sync_info', {})
    
    # Tambahkan handler untuk slider untuk memastikan total selalu 1.0
    def update_sliders(change):
        sender = change['owner']
        new_value = change['new']
        
        # Jika sender adalah train_slider
        if sender == train_slider:
            # Hitung sisa untuk val dan test dengan proporsi yang sama
            remaining = 1.0 - new_value
            # Bagi sisa secara merata antara val dan test
            val_value = round(remaining / 2, 2)
            test_value = round(remaining / 2, 2)
            # Update slider val dan test
            val_slider.value = val_value
            test_slider.value = test_value
        # Jika sender adalah val_slider
        elif sender == val_slider:
            # Hitung test berdasarkan train dan val
            test_value = round(1.0 - train_slider.value - new_value, 2)
            # Pastikan test tidak negatif
            if test_value < 0:
                test_value = 0
                # Recalculate val
                val_slider.value = round(1.0 - train_slider.value, 2)
            else:
                test_slider.value = test_value
        # Jika sender adalah test_slider
        elif sender == test_slider:
            # Hitung val berdasarkan train dan test
            val_value = round(1.0 - train_slider.value - new_value, 2)
            # Pastikan val tidak negatif
            if val_value < 0:
                val_value = 0
                # Recalculate test
                test_slider.value = round(1.0 - train_slider.value, 2)
            else:
                val_slider.value = val_value
        
        # Update total label
        total = round(train_slider.value + val_slider.value + test_slider.value, 2)
        color = COLORS['success'] if total == 1.0 else COLORS['danger']
        total_label.value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
    
    # Register handler untuk slider
    train_slider.observe(update_sliders, names='value')
    val_slider.observe(update_sliders, names='value')
    test_slider.observe(update_sliders, names='value')
    
    # Buat layout UI
    split_config_box = widgets.VBox([
        header,
        ui_components['status_panel'],
        widgets.HBox([
            widgets.VBox([
                widgets.HTML(value="<h3>Ratio Split Dataset</h3>"),
                train_slider,
                val_slider,
                test_slider,
                total_label,
                stratified_checkbox,
                random_seed
            ], layout=widgets.Layout(width='50%')),
            widgets.VBox([
                widgets.HTML(value="<h3>Path dan Backup</h3>"),
                dataset_path,
                preprocessed_path,
                backup_checkbox,
                backup_dir
            ], layout=widgets.Layout(width='50%'))
        ]),
        save_reset_buttons['container']
    ])
    
    # Tambahkan UI ke komponen
    ui_components['ui'] = split_config_box
    
    return ui_components
