
"""
File: smartcash/ui/components/config_form.py
Deskripsi: Komponen shared untuk form konfigurasi umum dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Callable
from smartcash.ui.utils.constants import ICONS, COLORS

def create_config_form(fields: List[Dict[str, Any]] = None, title: str = "Konfigurasi", description: str = None,
                      width: str = "100%", icon: str = "settings", with_save_button: bool = True,
                      with_reset_button: bool = True, save_handler: Callable = None, reset_handler: Callable = None) -> Dict[str, Any]:
    """Buat form konfigurasi dengan one-liner style."""
    fields = fields or [{'type': 'dropdown', 'name': 'option1', 'description': 'Opsi 1:', 
                        'options': [('Pilihan 1', 'value1'), ('Pilihan 2', 'value2')], 'value': 'value1'},
                       {'type': 'text', 'name': 'text1', 'description': 'Teks:', 'value': ''}]
    
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    
    def _create_field(field):
        field_type, name, desc, field_value = field.get('type', 'text'), field.get('name', 'field'), field.get('description', 'Field:'), field.get('value', '')
        if field_type == 'dropdown':
            return widgets.Dropdown(options=field.get('options', [('Default', 'default')]), value=field_value, description=desc,
                                   style={'description_width': 'initial'}, layout=widgets.Layout(width=width))
        elif field_type == 'slider':
            return widgets.IntSlider(value=field_value, min=field.get('min', 0), max=field.get('max', 100), step=field.get('step', 1),
                                   description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width=width))
        elif field_type == 'checkbox':
            return widgets.Checkbox(value=field_value, description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width=width))
        else:
            return widgets.Text(value=str(field_value), description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width=width))
    
    form_fields = {field.get('name', 'field'): _create_field(field) for field in fields}
    field_widgets = list(form_fields.values())
    
    buttons, button_widgets = {}, []
    if with_save_button:
        save_button = widgets.Button(description='Simpan', button_style='primary', icon=ICONS.get('save', '💾'),
                                   tooltip='Simpan konfigurasi', layout=widgets.Layout(width='100px'))
        save_handler and save_button.on_click(save_handler)
        button_widgets.append(save_button), buttons.update({'save_button': save_button})
    
    if with_reset_button:
        reset_button = widgets.Button(description='Reset', button_style='warning', icon=ICONS.get('reset', '🔄'),
                                    tooltip='Reset konfigurasi ke default', layout=widgets.Layout(width='100px'))
        reset_handler and reset_button.on_click(reset_handler)
        button_widgets.append(reset_button), buttons.update({'reset_button': reset_button})
    
    button_container = (widgets.HBox(button_widgets, layout=widgets.Layout(display='flex', flex_flow='row nowrap',
                                                                          justify_content='flex-end', align_items='center',
                                                                          gap='10px', width='auto', margin='10px 0px')) if button_widgets else None)
    button_container and buttons.update({'container': button_container})
    
    widgets_list = [header] + ([description_widget] if description_widget else []) + field_widgets + ([button_container] if button_container else [])
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px'))
    status_panel = widgets.Output(layout=widgets.Layout(width=width, min_height='30px'))
    
    return {'container': container, 'fields': form_fields, 'buttons': buttons, 'header': header, 'status_panel': status_panel}
