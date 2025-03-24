"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies dengan memanfaatkan ui_helpers untuk konsistensi tampilan
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """Buat komponen UI untuk instalasi dependencies dengan ui_helpers."""
    
    # Import komponen dari ui_helpers untuk konsistensi
    from smartcash.ui.utils.alert_utils import create_info_box
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.helpers.ui_helpers import create_spacing
    from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
    from smartcash.ui.setup.package_requirements import get_package_categories
    
    # Header dengan komponen standar
    header = create_header(
        "📦 Instalasi Dependencies", 
        "Setup package yang diperlukan untuk SmartCash"
    )
    
    # Status panel menggunakan komponen alert standar
    status_panel = widgets.HTML(
        create_info_box(
            "Pilih Packages", 
            "Pilih packages yang akan diinstall dan klik \"Install Packages\"",
            style="info"
        ).value
    )
    
    # Menggunakan get_package_categories yang sudah ada
    package_categories = get_package_categories()
    
    # Simpan referensi ke checkbox dan status widgets
    checkboxes = {}
    group_boxes = []
    
    # Buat UI untuk setiap kategori package
    for category in package_categories:
        # Header untuk grup
        group_header = widgets.HTML(f"""
        <div style="padding:5px 0">
            <h3 style="margin:5px 0;color:inherit">{category['icon']} {category['name']}</h3>
            <p style="margin:2px 0;color:{COLORS['muted']}">{category['description']}</p>
        </div>
        """)
        
        # Buat checkboxes dengan layout untuk tampilan vertikal
        package_rows = []
        for package in category['packages']:
            # Widget status di samping checkbox
            status_widget = widgets.HTML(f"<div style='width:100px;color:{COLORS['muted']}'>Memeriksa...</div>")
            
            # Checkbox untuk package
            checkbox = widgets.Checkbox(
                description=package['name'],
                value=package['default'],
                indent=False,
                layout=widgets.Layout(width='auto'),
                tooltip=package['description']
            )
            
            # Row untuk checkbox dan status
            row = widgets.HBox(
                [checkbox, status_widget],
                layout=widgets.Layout(
                    width='100%',
                    align_items='center'
                )
            )
            
            package_rows.append(row)
            
            # Simpan referensi ke checkbox dan status
            checkboxes[package['key']] = checkbox
            checkboxes[f"{package['key']}_status"] = status_widget
        
        # VBox untuk group checkboxes
        group_checkboxes = widgets.VBox(
            package_rows,
            layout=widgets.Layout(
                margin='5px 0 10px 15px',  # Memberikan indentasi
                width='100%'
            )
        )
        
        # Box untuk grup dengan border
        group_box = widgets.VBox(
            [group_header, group_checkboxes],
            layout=widgets.Layout(
                margin='10px 0',
                padding='10px',
                border=f'1px solid {COLORS["border"]}',
                border_radius='5px',
                width='100%'  # Gunakan lebar penuh untuk layout kolom
            )
        )
        
        group_boxes.append(group_box)
    
    # Container untuk groups dengan VBox (column layout)
    packages_container = widgets.VBox(
        group_boxes,
        layout=widgets.Layout(
            width='100%',
            margin='10px 0'
        )
    )
    
    # Custom package input
    custom_header = widgets.HTML(f"""
    <div style="padding:5px 0">
        <h3 style="margin:5px 0;color:{COLORS['dark']}">{ICONS.get('edit', '📝')} Custom Packages</h3>
        <p style="margin:2px 0;color:{COLORS['muted']}">Package tambahan yang dibutuhkan (satu per baris)</p>
    </div>
    """)
    
    custom_packages = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)',
        layout=widgets.Layout(
            width='100%',
            height='80px'
        )
    )
    
    custom_section = widgets.VBox(
        [custom_header, custom_packages],
        layout=widgets.Layout(
            margin='10px 0',
            padding='10px',
            border=f'1px solid {COLORS["border"]}',
            border_radius='5px',
            width='100%'
        )
    )
    
    # Tombol aksi 
    install_button = widgets.Button(
        description='Mulai Instalasi',
        button_style='primary',
        icon='download',
        tooltip="Mulai proses instalasi package",
        layout=widgets.Layout(margin='5px')
    )
    
    # Layout tombol ke tengah
    button_container = widgets.HBox(
        [install_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Progress bar
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            visibility='hidden'  # Hidden by default
        ),
        style={'description_width': 'initial', 'bar_color': COLORS['primary']}
    )
    
    # Progress label
    progress_label = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin='5px 0',
            visibility='hidden'  # Hidden by default
        )
    )
    
    # Status output area
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Info box menggunakan komponen standar
    info_box = get_dependencies_info()
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header,
            status_panel,
            packages_container,
            custom_section,
            button_container,
            progress,
            progress_label,
            status,
            info_box
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Struktur final komponen UI
    ui_components = {
        'ui': main,
        'status': status,
        'status_panel': status_panel,
        'install_button': install_button,
        'install_progress': progress,
        'progress_label': progress_label,
        'custom_packages': custom_packages,
        'module_name': 'dependency_installer',
        **checkboxes  # Sertakan semua checkbox dan status widgets
    }
    
    return ui_components