"""
File: smartcash/ui/setup/dependency_installer/components/ui_components.py
Deskripsi: Komponen UI untuk dependency installer
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML

def create_main_container() -> Dict[str, Any]:
    """Membuat container utama untuk dependency installer
    
    Returns:
        Dictionary komponen UI
    """
    # Buat container utama
    main_container = widgets.VBox(
        layout=widgets.Layout(padding='10px', width='100%')
    )
    
    # Buat header
    header = widgets.HTML(value='<h3>⚠️ Deps (Fallback Mode)</h3>')
    
    # Buat output widget untuk log
    log_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            padding='10px',
            width='100%'
        )
    )
    
    # Tambahkan komponen ke container
    main_container.children = [header, log_output]
    
    # Buat dictionary komponen UI
    ui_components = {
        'main_container': main_container,
        'header': header,
        'log_output': log_output,
        'ui': main_container,
        'status': log_output
    }
    
    return ui_components

def create_control_panel() -> Tuple[widgets.HBox, Dict[str, widgets.Button]]:
    """Membuat panel kontrol dengan tombol-tombol
    
    Returns:
        Tuple (panel, dictionary tombol)
    """
    # Buat tombol-tombol
    install_button = widgets.Button(
        description='Install',
        button_style='primary',
        icon='download',
        layout=widgets.Layout(width='auto')
    )
    
    analyze_button = widgets.Button(
        description='Analyze',
        button_style='info',
        icon='search',
        layout=widgets.Layout(width='auto')
    )
    
    reset_button = widgets.Button(
        description='Reset',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(width='auto')
    )
    
    # Buat panel kontrol
    control_panel = widgets.HBox(
        [install_button, analyze_button, reset_button],
        layout=widgets.Layout(
            padding='10px',
            justify_content='flex-start',
            width='100%'
        )
    )
    
    # Buat dictionary tombol
    buttons = {
        'install_button': install_button,
        'analyze_button': analyze_button,
        'reset_button': reset_button
    }
    
    return control_panel, buttons

def create_progress_container() -> Dict[str, Any]:
    """Membuat container progress
    
    Returns:
        Dictionary komponen progress
    """
    # Buat progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='100%')
    )
    
    # Buat label progress
    progress_label = widgets.HTML(
        value='Siap',
        layout=widgets.Layout(padding='5px', width='100%')
    )
    
    # Buat container progress
    progress_container = widgets.VBox(
        [progress_bar, progress_label],
        layout=widgets.Layout(
            padding='10px',
            border='1px solid #ddd',
            margin='10px 0',
            width='100%',
            visibility='hidden'
        )
    )
    
    # Buat dictionary komponen progress
    progress_components = {
        'progress_container': progress_container,
        'progress_bar': progress_bar,
        'progress_label': progress_label
    }
    
    return progress_components

def create_status_panel() -> Dict[str, Any]:
    """Membuat panel status
    
    Returns:
        Dictionary komponen status
    """
    # Buat widget status
    status_widget = widgets.HTML(
        value='Siap',
        layout=widgets.Layout(
            padding='10px',
            width='100%',
            background_color='#17a2b8',
            color='white'
        )
    )
    
    # Buat container status
    status_container = widgets.VBox(
        [status_widget],
        layout=widgets.Layout(
            padding='0',
            border='1px solid #17a2b8',
            margin='10px 0',
            width='100%'
        )
    )
    
    # Buat dictionary komponen status
    status_components = {
        'status_container': status_container,
        'status_widget': status_widget
    }
    
    return status_components

def create_error_widget() -> widgets.Output:
    """Membuat widget untuk menampilkan error
    
    Returns:
        Output widget
    """
    return widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            padding='10px',
            width='100%'
        )
    )

def assemble_ui_components() -> Dict[str, Any]:
    """Membuat dan menggabungkan semua komponen UI
    
    Returns:
        Dictionary komponen UI
    """
    # Buat komponen-komponen UI
    ui_components = create_main_container()
    control_panel, buttons = create_control_panel()
    progress_components = create_progress_container()
    status_components = create_status_panel()
    error_widget = create_error_widget()
    
    # Gabungkan semua komponen
    ui_components.update(buttons)
    ui_components.update(progress_components)
    ui_components.update(status_components)
    ui_components['error_widget'] = error_widget
    
    # Susun komponen dalam container utama
    main_container = ui_components['main_container']
    main_container.children = [
        ui_components['header'],
        control_panel,
        ui_components['status_container'],
        ui_components['progress_container'],
        ui_components['log_output'],
        ui_components['error_widget']
    ]
    
    # Sembunyikan error widget secara default
    ui_components['error_widget'].layout.visibility = 'hidden'
    
    return ui_components
