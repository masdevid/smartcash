"""
File: smartcash/ui/components/config_form.py
Deskripsi: Komponen shared untuk form konfigurasi umum
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple, Callable
from smartcash.ui.utils.constants import ICONS, COLORS

def create_config_form(
    fields: List[Dict[str, Any]] = None,
    title: str = "Konfigurasi",
    description: str = None,
    width: str = "100%",
    icon: str = "settings",
    with_save_button: bool = True,
    with_reset_button: bool = True,
    save_handler: Callable = None,
    reset_handler: Callable = None
) -> Dict[str, Any]:
    """
    Buat form konfigurasi yang dapat digunakan di berbagai modul.
    
    Args:
        fields: List dictionary untuk field form dengan format:
               {
                   'type': 'dropdown|text|slider|checkbox',
                   'name': 'nama_field',
                   'description': 'Deskripsi Field:',
                   'options': [('Label 1', 'value1'), ('Label 2', 'value2')], # untuk dropdown
                   'value': nilai_default,
                   'min': nilai_min, # untuk slider
                   'max': nilai_max, # untuk slider
                   'step': nilai_step # untuk slider
               }
        title: Judul form
        description: Deskripsi tambahan (opsional)
        width: Lebar komponen
        icon: Ikon untuk judul
        with_save_button: Apakah perlu menambahkan tombol save
        with_reset_button: Apakah perlu menambahkan tombol reset
        save_handler: Handler untuk tombol save
        reset_handler: Handler untuk tombol reset
        
    Returns:
        Dictionary berisi komponen form
    """
    # Default fields jika tidak ada yang diberikan
    if fields is None:
        fields = [
            {
                'type': 'dropdown',
                'name': 'option1',
                'description': 'Opsi 1:',
                'options': [('Pilihan 1', 'value1'), ('Pilihan 2', 'value2')],
                'value': 'value1'
            },
            {
                'type': 'text',
                'name': 'text1',
                'description': 'Teks:',
                'value': ''
            }
        ]
    
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk form
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat field untuk setiap item
    form_fields = {}
    field_widgets = []
    
    for field in fields:
        field_type = field.get('type', 'text')
        name = field.get('name', 'field')
        description = field.get('description', 'Field:')
        field_value = field.get('value', '')
        
        if field_type == 'dropdown':
            options = field.get('options', [('Default', 'default')])
            widget = widgets.Dropdown(
                options=options,
                value=field_value,
                description=description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width=width)
            )
        elif field_type == 'text':
            widget = widgets.Text(
                value=field_value,
                description=description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width=width)
            )
        elif field_type == 'slider':
            min_val = field.get('min', 0)
            max_val = field.get('max', 100)
            step = field.get('step', 1)
            widget = widgets.IntSlider(
                value=field_value,
                min=min_val,
                max=max_val,
                step=step,
                description=description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width=width)
            )
        elif field_type == 'checkbox':
            widget = widgets.Checkbox(
                value=field_value,
                description=description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width=width)
            )
        else:
            # Default to text field
            widget = widgets.Text(
                value=str(field_value),
                description=description,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width=width)
            )
        
        field_widgets.append(widget)
        form_fields[name] = widget
    
    # Buat tombol Save dan Reset jika diperlukan
    buttons = {}
    button_widgets = []
    
    if with_save_button:
        save_button = widgets.Button(
            description='Simpan',
            button_style='primary',
            icon=ICONS.get('save', '💾'),
            tooltip='Simpan konfigurasi',
            layout=widgets.Layout(width='100px')
        )
        if save_handler:
            save_button.on_click(save_handler)
        button_widgets.append(save_button)
        buttons['save_button'] = save_button
    
    if with_reset_button:
        reset_button = widgets.Button(
            description='Reset',
            button_style='warning',
            icon=ICONS.get('reset', '🔄'),
            tooltip='Reset konfigurasi ke default',
            layout=widgets.Layout(width='100px')
        )
        if reset_handler:
            reset_button.on_click(reset_handler)
        button_widgets.append(reset_button)
        buttons['reset_button'] = reset_button
    
    # Buat container untuk tombol jika ada
    button_container = None
    if button_widgets:
        button_container = widgets.HBox(
            button_widgets,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row nowrap',
                justify_content='flex-end',
                align_items='center',
                gap='10px',
                width='auto',
                margin='10px 0px'
            )
        )
        buttons['container'] = button_container
    
    # Buat container untuk form
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.extend(field_widgets)
    if button_container:
        widgets_list.append(button_container)
    
    container = widgets.VBox(
        widgets_list,
        layout=widgets.Layout(
            margin='10px 0px',
            padding='10px',
            border='1px solid #eee',
            border_radius='4px'
        )
    )
    
    # Status panel untuk menampilkan pesan
    status_panel = widgets.Output(
        layout=widgets.Layout(width=width, min_height='30px')
    )
    
    return {
        'container': container,
        'fields': form_fields,
        'buttons': buttons,
        'header': header,
        'status_panel': status_panel
    }
