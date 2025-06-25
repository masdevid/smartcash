"""
File: smartcash/ui/dataset/split/components/ui_layout.py
Deskripsi: Layout komponen untuk UI split dataset - refactored dengan reusable components
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.components import create_header, create_responsive_two_column, create_log_accordion
from smartcash.ui.dataset.split.components.ui_form import create_ratio_section, create_path_section
from smartcash.ui.info_boxes.split_info import get_split_info


def create_split_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Buat layout utama untuk UI split dataset dengan responsive design"""
    
    # Header dengan icon
    header = create_header("Konfigurasi Split Dataset", "Pengaturan pembagian dataset untuk training, validation, dan testing", "✂️")
    
    # Create sections
    ratio_section = create_ratio_section(form_components)
    path_section = create_path_section(form_components)
    
    # Info accordion
    info_accordion = get_split_info()
    
    # Create log accordion
    log_accordion = create_log_accordion()
    
    # Ensure all components are widgets
    form_container_children = [
        create_responsive_two_column(ratio_section, path_section),
        form_components.get('save_reset_container', widgets.HTML(''))
    ]
    
    # Create form container
    form_container = widgets.VBox(
        [c for c in form_container_children if c is not None],
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Create main container with all widgets
    main_children = [
        header,
        form_components.get('status_panel', widgets.HTML('')),
        form_container,
        info_accordion if info_accordion is not None else widgets.HTML(''),
        log_accordion if log_accordion is not None else widgets.HTML('')
    ]
    
    main_container = widgets.VBox(
        [c for c in main_children if c is not None],
        layout=widgets.Layout(width='100%', padding='10px')
    )
    
    components = {
        'header': header, 
        'ratio_section': ratio_section, 
        'path_section': path_section,
        'form_container': form_container, 
        'info_accordion': info_accordion, 
        'log_accordion': log_accordion,
        'main_container': main_container
    }
    
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(components)
    return components

