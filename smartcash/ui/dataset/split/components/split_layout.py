"""
File: smartcash/ui/dataset/split/components/split_layout.py
Deskripsi: Layout komponen untuk UI split dataset - refactored dengan reusable components
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.dataset.split.components.split_form import create_ratio_section, create_path_section

def create_split_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Buat layout utama untuk UI split dataset dengan responsive design"""
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.layout_utils import create_responsive_two_column
    
    # Header dengan existing utility
    header = create_header("Konfigurasi Split Dataset", "Pengaturan pembagian dataset untuk training, validation, dan testing")
    
    # Create sections
    ratio_section = create_ratio_section(form_components)
    path_section = create_path_section(form_components)
    
    # Info accordion
    info_content = _create_info_content()
    info_accordion = create_info_accordion("Informasi Split Dataset", info_content, "info")
    
    # Layout containers
    form_container = widgets.VBox([
        create_responsive_two_column(ratio_section, path_section),
        form_components['save_reset_container']
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    main_container = widgets.VBox([
        header, form_components['status_panel'], 
        form_container, info_accordion['container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    return {
        'header': header, 'ratio_section': ratio_section, 'path_section': path_section,
        'form_container': form_container, 'info_accordion': info_accordion, 'main_container': main_container
    }


def _create_info_content() -> widgets.HTML:
    """Buat info content dengan consolidated HTML styling"""
    return widgets.HTML(value="""
        <div style='padding: 10px; background-color: #f8f9fa;'>
        <p><strong>Split Dataset</strong> membagi data menjadi tiga bagian:</p>
        <ul>
        <li><strong>Train (70-80%):</strong> Data untuk melatih model</li>
        <li><strong>Validation (10-15%):</strong> Data untuk validasi selama training</li>
        <li><strong>Test (10-15%):</strong> Data untuk evaluasi final</li>
        </ul>
        <p><strong>Stratified Split:</strong> Mempertahankan distribusi kelas yang sama di semua split</p>
        <p><strong>Random Seed:</strong> Untuk reproduksibilitas hasil split</p>
        </div>
        """)