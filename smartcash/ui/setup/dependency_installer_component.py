"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies dengan alur 3 tahap otomatis
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk instalasi dependencies.
    
    Args:
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components
    """
    # Import komponen UI standar
    from smartcash.ui.utils.alert_utils import create_info_box
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
    from smartcash.ui.setup.package_requirements import get_package_categories
    
    # Header dengan styling konsisten
    header = create_header(
        "📦 Instalasi Dependencies", 
        "Setup package otomatis untuk SmartCash"
    )
    
    # Status panel menggunakan komponen alert standar
    status_panel = widgets.HTML(
        create_info_box(
            "Auto-Deteksi Dependencies", 
            "Sistem akan mendeteksi dependencies yang dibutuhkan dan menginstalnya secara otomatis",
            style="info"
        ).value
    )
    
    # Kategori package dengan metadata
    package_categories = get_package_categories()
    
    # Tempat untuk menyimpan checkbox widgets
    checkboxes = {}
    category_widgets = []
    
    # Buat widget untuk setiap kategori package
    for category in package_categories:
        # Header kategori
        category_header = widgets.HTML(f"""
        <div style="padding:5px 0">
            <h3 style="margin:5px 0;color:{COLORS['dark']}">{category['icon']} {category['name']}</h3>
            <p style="margin:2px 0;color:{COLORS['muted']}">{category['description']}</p>
        </div>
        """)
        
        # Buat checkbox untuk setiap package dalam kategori
        package_widgets = []
        for package in category['packages']:
            # Widget status di samping checkbox
            pkg_status = widgets.HTML(f"<div style='width:100px;color:{COLORS['muted']}'>Memeriksa...</div>")
            
            # Checkbox untuk package
            checkbox = widgets.Checkbox(
                description=package['name'],
                value=package['default'],
                indent=False,
                layout=widgets.Layout(width='auto'),
                tooltip=package['description']
            )
            
            # Simpan referensi ke checkbox
            checkboxes[package['key']] = checkbox
            
            # Package row dengan status
            package_row = widgets.HBox(
                [checkbox, pkg_status],
                layout=widgets.Layout(
                    width='100%',
                    align_items='center'
                )
            )
            
            # Simpan referensi status
            checkboxes[f"{package['key']}_status"] = pkg_status
            
            package_widgets.append(package_row)
        
        # Buat VBox untuk package dalam kategori
        packages_box = widgets.VBox(
            package_widgets,
            layout=widgets.Layout(
                margin='0 0 0 15px',  # Berikan indentasi
                width='100%'
            )
        )
        
        # Container untuk kategori
        category_box = widgets.VBox(
            [category_header, packages_box],
            layout=widgets.Layout(
                margin='10px 0',
                padding='10px',
                border=f'1px solid {COLORS["border"]}',
                border_radius='5px',
                width='100%'
            )
        )
        
        category_widgets.append(category_box)
    
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
            *category_widgets,
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