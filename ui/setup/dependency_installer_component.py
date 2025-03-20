"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies dengan memanfaatkan ui_helpers untuk konsistensi tampilan
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """Buat komponen UI untuk instalasi dependencies dengan ui_helpers."""
    
    # Import komponen dari ui_helpers untuk konsistensi
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.alert_utils import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.helpers.ui_helpers import create_button_group, create_divider, create_spacing
    
    # Header dengan komponen standar
    header = create_header(
        "📦 Instalasi Dependencies", 
        "Setup package yang diperlukan untuk SmartCash"
    )
    
    # Package groups dengan metainfo untuk memudahkan integrasi
    package_groups = {
        'core': [
            ('YOLOv5 requirements', 'yolov5_req', True, "YOLOv5 dependencies (numpy, opencv, torch, etc)"),
            ('SmartCash utils', 'smartcash_req', True, "SmartCash utility packages (pyyaml, termcolor, etc)"),
            ('Notebook tools', 'notebook_req', True, "Jupyter notebook utilities (ipywidgets, tqdm, etc)")
        ],
        'ml': [
            ('PyTorch', 'torch_req', True, "Deep learning framework dan toolkits"),
            ('OpenCV', 'opencv_req', True, "Computer vision library"),
            ('Albumentations', 'albumentations_req', True, "Augmentasi gambar untuk training")
        ],
        'viz': [
            ('Matplotlib', 'matplotlib_req', True, "Visualisasi data dan plot"),
            ('Pandas', 'pandas_req', True, "Manipulasi dan analisis data"),
            ('Seaborn', 'seaborn_req', True, "Visualisasi statistik")
        ]
    }
    
    # Status panel menggunakan komponen alert standar
    status_panel = widgets.HTML(
        create_info_box(
            "Pilih Packages", 
            "Pilih packages yang akan diinstall dan klik \"Install Packages\"",
            style="info"
        ).value
    )
    
    # Buat UI untuk package groups dengan layout yang lebih kompak
    package_section_widgets = []
    for group_name, packages in package_groups.items():
        # Header untuk grup
        group_header = widgets.HTML(f"""
        <div style="padding:5px 0">
            <h3 style="margin:5px 0;color:inherit">{ICONS.get('package', '📌')} {group_name.upper()} Packages</h3>
        </div>
        """)
        
        # Buat checkboxes dengan layout grid untuk tampilan lebih kompak
        checkboxes = []
        for desc, key, default_value, tooltip in packages:
            checkbox = widgets.Checkbox(
                value=default_value, 
                description=desc,
                indent=False,
                layout=widgets.Layout(padding='2px 0', width='100%'),
                tooltip=tooltip
            )
            checkboxes.append(checkbox)
        
        # Gunakan GridBox untuk layout yang lebih kompak
        checkboxes_group = widgets.GridBox(
            checkboxes,
            layout=widgets.Layout(
                grid_template_columns='100%',
                grid_gap='0px',
                padding='0px 10px',
                width='100%'
            )
        )
        
        # Box untuk grup
        group_box = widgets.VBox(
            [group_header, checkboxes_group],
            layout=widgets.Layout(
                margin='3px',
                padding='5px',
                border=f'1px solid {COLORS["border"]}',
                border_radius='5px',
                width='31%',
                min_width='180px'
            )
        )
        package_section_widgets.append(group_box)
    
    # Container untuk groups dengan HBox (row layout)
    packages_container = widgets.HBox(
        package_section_widgets,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='flex-start',
            width='100%'
        )
    )
    
    # Custom package input
    custom_header = widgets.HTML(f"""
    <div style="padding:5px 0">
        <h3 style="margin:5px 0;color:inherit">{ICONS.get('edit', '📝')} Custom Packages</h3>
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
            border_radius='5px'
        )
    )
    
    # Tombol aksi menggunakan komponen button_group dari ui_helpers
    button_layout = widgets.Layout(width='auto', margin='5px', height='auto')
    
    buttons = [
        ("Check All", "info", "check-square", None),
        ("Uncheck All", "warning", "square", None),
        ("Install Packages", "primary", "download", None),
        ("Check Installations", "success", "check", None)
    ]
    
    button_group = create_button_group(buttons, 
        widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='center',
            align_items='center',
            margin='5px 0',
            gap='5px'
        )
    )
    
    # Progress bar dengan styling standar
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            visibility='hidden'
        ),
        style={'description_width': 'initial', 'bar_color': COLORS['primary']}
    )
    
    # Status output dengan styling standar
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
    info_box = create_info_box(
        "Tentang Package Installation",
        f"""
        <p>Package diurutkan instalasi dari kecil ke besar:</p>
        <ol>
            <li>Notebook tools (ipywidgets, tqdm)</li>
            <li>Utility packages (pyyaml, termcolor)</li>
            <li>Data processing (matplotlib, pandas)</li>
            <li>Computer vision (OpenCV, Albumentations)</li>
            <li>Machine learning (PyTorch)</li>
        </ol>
        <p><strong>{ICONS.get('warning', '⚠️')} Catatan:</strong> Instalasi PyTorch mungkin memerlukan waktu lebih lama</p>
        """,
        'info',
        collapsed=True
    )
    
    # Gunakan komponen spacing dari ui_helpers
    spacing = create_spacing('15px')
    
    # Container utama
    main = widgets.VBox(
        [
            header,
            status_panel,
            packages_container,
            custom_section,
            button_group,
            progress,
            status,
            spacing,
            info_box
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Construct checkboxes mapping
    checkboxes = {}
    for i, (group_name, packages) in enumerate(package_groups.items()):
        group_widget = packages_container.children[i]
        for j, (desc, key, _, _) in enumerate(packages):
            checkboxes[key] = group_widget.children[1].children[j]
    
    # Struktur final komponen UI dengan seluruh referensi untuk handler
    ui_components = {
        'ui': main,
        'status': status,
        'status_panel': status_panel,
        'install_progress': progress,
        'check_all_button': button_group.children[0],
        'uncheck_all_button': button_group.children[1],
        'install_button': button_group.children[2],
        'check_button': button_group.children[3],
        'custom_packages': custom_packages,
        **checkboxes
    }
    
    return ui_components