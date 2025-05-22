"""
File: smartcash/ui/dataset/split/components/split_layout.py
Deskripsi: Layout komponen untuk UI split dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.dataset.split.components.split_form import create_ratio_section, create_path_section


def create_split_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat layout utama untuk UI split dataset.
    
    Args:
        form_components: Komponen form yang sudah dibuat
        
    Returns:
        Dict berisi komponen layout
    """
    # Header
    header = create_header(
        title="Konfigurasi Split Dataset",
        description="Pengaturan pembagian dataset untuk training, validation, dan testing"
    )
    
    # Sections
    ratio_section = create_ratio_section(form_components)
    path_section = create_path_section(form_components)
    
    # Info accordion
    info_content = widgets.HTML(
        value="""
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
        """
    )
    
    info_accordion = create_info_accordion(
        title="Informasi Split Dataset",
        content=info_content,
        icon="info",
        open_by_default=False
    )
    
    # Form container dengan 2 kolom
    form_container = widgets.VBox([
        widgets.HBox([
            widgets.Box([ratio_section], layout=widgets.Layout(width='48%')),
            widgets.Box([path_section], layout=widgets.Layout(width='48%'))
        ], layout=widgets.Layout(
            width='100%',
            display='flex',
            justify_content='space-between',
            align_items='flex-start'
        )),
        widgets.VBox([
            form_components['save_reset_container']
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
    ])
    
    # Main container
    main_container = widgets.VBox([
        header,
        form_components['status_panel'],
        form_container,
        info_accordion['container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    return {
        'header': header,
        'ratio_section': ratio_section,
        'path_section': path_section,
        'form_container': form_container,
        'info_accordion': info_accordion,
        'main_container': main_container
    }