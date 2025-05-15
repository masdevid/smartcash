"""
File: smartcash/ui/training_config/hyperparameters/components/hyperparameters_components.py
Deskripsi: Komponen UI untuk konfigurasi hyperparameter
"""

from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.info_boxes.hyperparameters_info import (
    get_hyperparameters_info,
    get_basic_hyperparameters_info,
    get_optimization_hyperparameters_info,
    get_advanced_hyperparameters_info
)

logger = get_logger(__name__)

def create_hyperparameters_info_panel() -> Tuple[widgets.Output, Any]:
    """
    Membuat panel informasi untuk hyperparameter.
    
    Returns:
        Tuple berisi output widget dan fungsi update
    """
    info_panel = widgets.Output(layout=widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    
    def update_hyperparameters_info(ui_components: Optional[Dict[str, Any]] = None):
        """
        Update informasi hyperparameter di panel info.
        
        Args:
            ui_components: Komponen UI
        """
        with info_panel:
            clear_output(wait=True)
            
            if not ui_components:
                display(widgets.HTML(
                    f"<h3>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h3>"
                    f"<p>Tidak ada informasi yang tersedia.</p>"
                ))
                return
            
            # Dapatkan nilai dari komponen UI
            try:
                batch_size = ui_components.get('batch_size_slider', widgets.IntSlider()).value
                image_size = ui_components.get('image_size_slider', widgets.IntSlider()).value
                epochs = ui_components.get('epochs_slider', widgets.IntSlider()).value
                optimizer = ui_components.get('optimizer_dropdown', widgets.Dropdown()).value
                learning_rate = ui_components.get('learning_rate_slider', widgets.FloatLogSlider()).value
                scheduler = ui_components.get('scheduler_dropdown', widgets.Dropdown()).value
                
                # Tampilkan informasi
                html_content = f"""
                <h3>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;">
                    <div>
                        <h4>Parameter Dasar</h4>
                        <ul>
                            <li><b>Batch Size:</b> {batch_size}</li>
                            <li><b>Image Size:</b> {image_size}</li>
                            <li><b>Epochs:</b> {epochs}</li>
                        </ul>
                    </div>
                    <div>
                        <h4>Optimasi</h4>
                        <ul>
                            <li><b>Optimizer:</b> {optimizer}</li>
                            <li><b>Learning Rate:</b> {learning_rate:.6f}</li>
                            <li><b>Scheduler:</b> {scheduler}</li>
                        </ul>
                    </div>
                </div>
                """
                
                # Tampilkan informasi tambahan jika ada
                if 'momentum_slider' in ui_components and not ui_components['momentum_slider'].disabled:
                    momentum = ui_components['momentum_slider'].value
                    html_content += f"""
                    <div>
                        <h4>Parameter Tambahan</h4>
                        <ul>
                            <li><b>Momentum:</b> {momentum:.4f}</li>
                    """
                    
                    if 'weight_decay_slider' in ui_components and not ui_components['weight_decay_slider'].disabled:
                        weight_decay = ui_components['weight_decay_slider'].value
                        html_content += f"<li><b>Weight Decay:</b> {weight_decay:.6f}</li>"
                    
                    html_content += "</ul></div>"
                
                # Tampilkan informasi early stopping jika diaktifkan
                if 'early_stopping_enabled_checkbox' in ui_components and ui_components['early_stopping_enabled_checkbox'].value:
                    patience = ui_components.get('early_stopping_patience_slider', widgets.IntSlider()).value
                    min_delta = ui_components.get('early_stopping_min_delta_slider', widgets.FloatSlider()).value
                    
                    html_content += f"""
                    <div>
                        <h4>Early Stopping</h4>
                        <ul>
                            <li><b>Patience:</b> {patience} epochs</li>
                            <li><b>Min Delta:</b> {min_delta:.6f}</li>
                        </ul>
                    </div>
                    """
                
                # Tampilkan informasi save best jika diaktifkan
                if 'save_best_checkbox' in ui_components and ui_components['save_best_checkbox'].value:
                    metric = ui_components.get('checkpoint_metric_dropdown', widgets.Dropdown()).value
                    
                    html_content += f"""
                    <div>
                        <h4>Save Best Model</h4>
                        <ul>
                            <li><b>Metric:</b> {metric}</li>
                        </ul>
                    </div>
                    """
                
                display(widgets.HTML(html_content))
            except Exception as e:
                display(widgets.HTML(
                    f"<h3>{ICONS.get('error', '❌')} Error</h3>"
                    f"<p>Error saat menampilkan informasi hyperparameter: {str(e)}</p>"
                ))
    
    return info_panel, update_hyperparameters_info

def create_hyperparameters_basic_components() -> Dict[str, Any]:
    """
    Membuat komponen UI dasar untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    # Batch size slider
    batch_size_slider = widgets.IntSlider(
        value=16,
        min=1,
        max=128,
        step=1,
        description='Batch Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='90%')
    )
    
    # Image size slider
    image_size_slider = widgets.IntSlider(
        value=640,
        min=320,
        max=1280,
        step=32,
        description='Image Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='90%')
    )
    
    # Epochs slider
    epochs_slider = widgets.IntSlider(
        value=100,
        min=1,
        max=500,
        step=1,
        description='Epochs:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='90%')
    )
    
    # Augment checkbox
    augment_checkbox = widgets.Checkbox(
        value=True,
        description='Gunakan Augmentasi',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Komponen dasar
    basic_components = {
        'batch_size_slider': batch_size_slider,
        'image_size_slider': image_size_slider,
        'epochs_slider': epochs_slider,
        'augment_checkbox': augment_checkbox
    }
    
    # Basic box
    basic_box = widgets.VBox([
        widgets.HTML('<h3>Parameter Dasar</h3>'),
        batch_size_slider,
        image_size_slider,
        epochs_slider,
        augment_checkbox
    ], layout=widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    
    basic_components['basic_box'] = basic_box
    
    return basic_components

def create_hyperparameters_optimization_components() -> Dict[str, Any]:
    """
    Membuat komponen UI optimasi untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    # Optimizer dropdown
    optimizer_dropdown = widgets.Dropdown(
        options=['SGD', 'Adam', 'AdamW', 'RMSprop'],
        value='SGD',
        description='Optimizer:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Learning rate slider
    learning_rate_slider = widgets.FloatLogSlider(
        value=0.01,
        base=10,
        min=-6,  # 10^-6
        max=-1,  # 10^-1
        step=0.1,
        description='Learning Rate:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.6f',
        layout=widgets.Layout(width='90%')
    )
    
    # Momentum slider
    momentum_slider = widgets.FloatSlider(
        value=0.937,
        min=0.0,
        max=0.999,
        step=0.001,
        description='Momentum:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.4f',
        layout=widgets.Layout(width='90%')
    )
    
    # Weight decay slider
    weight_decay_slider = widgets.FloatLogSlider(
        value=0.0005,
        base=10,
        min=-6,  # 10^-6
        max=-2,  # 10^-2
        step=0.1,
        description='Weight Decay:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.6f',
        layout=widgets.Layout(width='90%')
    )
    
    # Scheduler dropdown
    scheduler_dropdown = widgets.Dropdown(
        options=['none', 'cosine', 'linear', 'step'],
        value='cosine',
        description='Scheduler:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Warmup epochs slider
    warmup_epochs_slider = widgets.IntSlider(
        value=3,
        min=0,
        max=10,
        step=1,
        description='Warmup Epochs:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='90%')
    )
    
    # Warmup momentum slider
    warmup_momentum_slider = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=0.999,
        step=0.001,
        description='Warmup Momentum:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.4f',
        layout=widgets.Layout(width='90%')
    )
    
    # Warmup bias lr slider
    warmup_bias_lr_slider = widgets.FloatLogSlider(
        value=0.1,
        base=10,
        min=-3,  # 10^-3
        max=0,  # 10^0
        step=0.1,
        description='Warmup Bias LR:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.4f',
        layout=widgets.Layout(width='90%')
    )
    
    # Buat info box untuk parameter optimasi
    optimization_info = get_optimization_hyperparameters_info(open_by_default=False)
    
    # Komponen optimasi
    optimization_components = {
        'optimizer_dropdown': optimizer_dropdown,
        'learning_rate_slider': learning_rate_slider,
        'momentum_slider': momentum_slider,
        'weight_decay_slider': weight_decay_slider,
        'scheduler_dropdown': scheduler_dropdown,
        'warmup_epochs_slider': warmup_epochs_slider,
        'warmup_momentum_slider': warmup_momentum_slider,
        'warmup_bias_lr_slider': warmup_bias_lr_slider
    }
    
    # Optimization box
    optimization_box = widgets.VBox([
        widgets.HTML('<h3>Parameter Optimasi</h3>'),
        optimizer_dropdown,
        learning_rate_slider,
        momentum_slider,
        weight_decay_slider,
        widgets.HTML('<h4>Learning Rate Scheduler</h4>'),
        scheduler_dropdown,
        warmup_epochs_slider,
        warmup_momentum_slider,
        warmup_bias_lr_slider
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0',
        width='49%'  # Lebar 49% untuk memberikan sedikit jarak
    ))
    
    optimization_components['optimization_box'] = optimization_box
    
    return optimization_components

def create_hyperparameters_advanced_components() -> Dict[str, Any]:
    """
    Membuat komponen UI lanjutan untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    # Early stopping checkbox
    early_stopping_enabled_checkbox = widgets.Checkbox(
        value=True,
        description='Aktifkan Early Stopping',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Early stopping patience slider
    early_stopping_patience_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Patience:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='90%')
    )
    
    # Early stopping min delta slider
    early_stopping_min_delta_slider = widgets.FloatLogSlider(
        value=0.001,
        base=10,
        min=-6,  # 10^-6
        max=-1,  # 10^-1
        step=0.1,
        description='Min Delta:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.6f',
        layout=widgets.Layout(width='90%')
    )
    
    # Save best checkbox
    save_best_checkbox = widgets.Checkbox(
        value=True,
        description='Simpan Model Terbaik',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Checkpoint metric dropdown
    checkpoint_metric_dropdown = widgets.Dropdown(
        options=['mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall', 'f1', 'loss'],
        value='mAP_0.5',
        description='Metric:',
        disabled=False,
        layout=widgets.Layout(width='90%')
    )
    
    # Komponen lanjutan
    advanced_components = {
        'early_stopping_enabled_checkbox': early_stopping_enabled_checkbox,
        'early_stopping_patience_slider': early_stopping_patience_slider,
        'early_stopping_min_delta_slider': early_stopping_min_delta_slider,
        'save_best_checkbox': save_best_checkbox,
        'checkpoint_metric_dropdown': checkpoint_metric_dropdown
    }
    
    # Early stopping box
    early_stopping_box = widgets.VBox([
        widgets.HTML('<h4>Early Stopping</h4>'),
        early_stopping_enabled_checkbox,
        early_stopping_patience_slider,
        early_stopping_min_delta_slider
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        margin='10px 0'
    ))
    
    # Checkpoint box
    checkpoint_box = widgets.VBox([
        widgets.HTML('<h4>Checkpoint</h4>'),
        save_best_checkbox,
        checkpoint_metric_dropdown
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        margin='10px 0'
    ))
    
    # Advanced box
    advanced_box = widgets.VBox([
        widgets.HTML('<h3>Parameter Lanjutan</h3>'),
        early_stopping_box,
        checkpoint_box
    ], layout=widgets.Layout(
        width='49%',  # Lebar 49% untuk memberikan space between yang lebih baik
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    
    advanced_components['early_stopping_box'] = early_stopping_box
    advanced_components['checkpoint_box'] = checkpoint_box
    advanced_components['advanced_box'] = advanced_box
    
    return advanced_components

def create_hyperparameters_button_components() -> Dict[str, Any]:
    """
    Membuat komponen tombol untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    # Save button
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        disabled=False,
        button_style='primary',
        tooltip='Simpan konfigurasi hyperparameter',
        icon=ICONS.get('save', '💾'),
        layout=widgets.Layout(width='auto')
    )
    
    # Reset button
    reset_button = widgets.Button(
        description='Reset ke Default',
        disabled=False,
        button_style='warning',
        tooltip='Reset konfigurasi hyperparameter ke default',
        icon=ICONS.get('reset', '🔄'),
        layout=widgets.Layout(width='auto')
    )
    
    # Status panel
    status_panel = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='50px',
        max_height='100px',
        overflow='auto',
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    
    # Komponen tombol
    button_components = {
        'save_button': save_button,
        'reset_button': reset_button,
        'status': status_panel
    }
    
    # Button box (tidak digunakan lagi karena tombol ditampilkan di form_container)
    button_box = widgets.HBox([
        save_button,
        reset_button
    ], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        justify_content='space-between',
        margin='10px 0'
    ))
    
    button_components['button_box'] = button_box
    
    return button_components

def create_hyperparameters_ui_components() -> Dict[str, Any]:
    """
    Membuat semua komponen UI untuk hyperparameter.
    
    Returns:
        Dict berisi semua komponen UI
    """
    # Import tab factory
    from smartcash.ui.components.tab_factory import create_tab_widget
    from smartcash.ui.utils.header_utils import create_header
    
    # Buat komponen UI
    basic_components = create_hyperparameters_basic_components()
    optimization_components = create_hyperparameters_optimization_components()
    advanced_components = create_hyperparameters_advanced_components()
    button_components = create_hyperparameters_button_components()
    
    # Buat panel info
    info_panel, update_hyperparameters_info = create_hyperparameters_info_panel()
    
    # Gabungkan semua komponen
    ui_components = {
        **basic_components,
        **optimization_components,
        **advanced_components,
        **button_components,
        'info_panel': info_panel,
        'update_hyperparameters_info': update_hyperparameters_info
    }
    
    # Buat header
    header = create_header(
        title="Konfigurasi Hyperparameter",
        description="Pengaturan parameter untuk proses training model",
        icon=ICONS.get('settings', '⚙️')
    )
    
    # Buat form container untuk tab konfigurasi
    form_container = widgets.VBox([
        widgets.HBox([
            basic_components['basic_box'],
            widgets.HBox([
                optimization_components['optimization_box'],
                advanced_components['advanced_box']
            ], layout=widgets.Layout(
                width='70%',  # Lebar total untuk container kanan
                display='flex',
                flex_flow='row nowrap',  # Memastikan tidak ada wrap
                align_items='flex-start',
                justify_content='space-between'
            ))
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='row wrap',
            align_items='flex-start',
            justify_content='space-between'
        )),
        widgets.HBox([
            button_components['save_button'], 
            button_components['reset_button']
        ], layout=widgets.Layout(
            justify_content='space-between', 
            margin='20px 0px 10px 0px'
        )),
        widgets.HTML(
            value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} "
                  f"Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.</div>"
        )
    ])
    
    # Buat info box umum untuk hyperparameter
    general_info = get_hyperparameters_info(open_by_default=True)
    
    # Buat info container untuk tab informasi
    info_container = widgets.VBox([
        widgets.HTML("<h4>Informasi Hyperparameter</h4>"),
        general_info,
        info_panel
    ])
    
    # Buat tab untuk form dan info
    tab_items = [
        ('Konfigurasi', form_container),
        ('Informasi', info_container)
    ]
    tabs = create_tab_widget(tab_items)
    
    # Set tab yang aktif
    tabs.selected_index = 0
    
    # Buat info boxes untuk footer
    basic_info = get_basic_hyperparameters_info(open_by_default=False)
    optimization_info = get_optimization_hyperparameters_info(open_by_default=False)
    advanced_info = get_advanced_hyperparameters_info(open_by_default=False)
    
    # Buat footer dengan info boxes yang menumpuk (stacked)
    # Set accordion behavior agar hanya satu yang terbuka
    basic_info.selected_index = None
    optimization_info.selected_index = None
    advanced_info.selected_index = None
    
    # Fungsi untuk menutup accordion lain saat satu dibuka
    def on_accordion_select(change, accordion_list):
        if change['new'] is not None:  # Jika ada yang dibuka
            # Tutup semua accordion lain
            for acc in accordion_list:
                if acc != change['owner']:
                    acc.selected_index = None
    
    # Daftar semua accordion
    accordion_list = [basic_info, optimization_info, advanced_info]
    
    # Tambahkan observer ke masing-masing accordion
    for acc in accordion_list:
        acc.observe(lambda change, acc_list=accordion_list: on_accordion_select(change, acc_list), names='selected_index')
    
    # Buat footer dengan info boxes yang menumpuk (stacked)
    footer_info = widgets.VBox([
        widgets.HTML("<h4>Informasi Parameter</h4>"),
        widgets.VBox([
            basic_info,
            optimization_info,
            advanced_info
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        ))
    ], layout=widgets.Layout(
        width='100%',
        margin='20px 0 0 0',
        padding='10px',
        border_top='1px solid #ddd'
    ))
    
    # Buat container utama
    main_container = widgets.VBox([
        header,
        tabs,
        button_components['status'],
        footer_info
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    ui_components['main_container'] = main_container
    ui_components['main_layout'] = main_container  # Untuk kompatibilitas
    ui_components['tabs'] = tabs
    
    return ui_components
