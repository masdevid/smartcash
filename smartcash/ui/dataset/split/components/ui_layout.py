"""
File: smartcash/ui/dataset/split/components/ui_layout.py
Deskripsi: Layout komponen untuk UI split dataset - refactored dengan reusable components
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.dataset.split.components.ui_form import create_ratio_section, create_path_section
from smartcash.ui.components import create_header, create_responsive_two_column
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
    
    # Layout containers - remove header from main_container since it's already in header
    form_container = widgets.VBox([
        create_responsive_two_column(ratio_section, path_section),
        form_components['save_reset_container']
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    main_container = widgets.VBox([
        header, 
        form_components['status_panel'], 
        form_container, 
        info_accordion['container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    return {
        'header': header, 'ratio_section': ratio_section, 'path_section': path_section,
        'form_container': form_container, 'info_accordion': info_accordion, 'main_container': main_container
    }

