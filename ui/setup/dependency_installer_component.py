"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies dengan styling yang konsisten
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.components.headers import create_header, create_section_title
from smartcash.ui.components.alerts import create_info_box
from smartcash.ui.components.widget_layouts import (
    main_container, card_container, output_area, 
    BUTTON_LAYOUTS, GROUP_LAYOUTS, CONTAINER_LAYOUTS
)
from smartcash.ui.utils.constants import COLORS, ICONS

def create_dependency_installer_ui(env=None, config=None):
    """Buat komponen UI untuk instalasi dependencies."""
    # Header 
    header = create_header(
        "📦 Instalasi Dependencies", 
        "Setup package yang diperlukan untuk SmartCash"
    )
    
    # Package groups
    package_groups = {
        'core': [
            ('YOLOv5 requirements', 'yolov5_req'),
            ('SmartCash utils', 'smartcash_req'),
            ('Notebook tools', 'notebook_req')
        ],
        'ml': [
            ('PyTorch', 'torch_req'),
            ('OpenCV', 'opencv_req'),
            ('Albumentations', 'albumentations_req')
        ],
        'viz': [
            ('Matplotlib', 'matplotlib_req'),
            ('Pandas', 'pandas_req'),
            ('Seaborn', 'seaborn_req')
        ]
    }
    
    # Panel status dan kontrol
    status_panel = widgets.HTML(value=f"""
    <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
               color:{COLORS['alert_info_text']}; 
               border-radius:4px; margin:10px 0;
               border-left:4px solid {COLORS['alert_info_text']};">
        <h3 style="color:inherit; margin:5px 0">{ICONS['config']} Status Dependencies</h3>
        <p style="margin:5px 0">ℹ️ Pilih packages yang akan diinstall dan klik 'Install Packages'</p>
    </div>
    """)
    
    # Konstruksi package group widgets dalam cards
    package_section_widgets = []
    for group_name, packages in package_groups.items():
        # Header untuk setiap grup
        group_header = create_section_title(
            f"{ICONS.get(group_name.lower(), ICONS['config'])} {group_name.upper()} Packages"
        )
        
        # Checkbox untuk package dalam grup
        checkboxes = []
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc,
                indent=False,
                layout=widgets.Layout(margin='5px 0')
            )
            checkboxes.append(checkbox)
        
        # Layout untuk checkboxes
        checkboxes_group = widgets.VBox(
            checkboxes,
            layout=widgets.Layout(
                margin='5px 10px',
                padding='5px'
            )
        )
        
        # Card untuk grup
        group_card = widgets.VBox(
            [group_header, checkboxes_group],
            layout=card_container
        )
        
        package_section_widgets.append(group_card)
    
    # Container untuk grup package
    packages_container = widgets.HBox(
        package_section_widgets,
        layout=GROUP_LAYOUTS['horizontal']
    )
    
    # Custom package input
    custom_packages_header = create_section_title(f"{ICONS['add']} Custom Packages")
    custom_packages = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)',
        layout=widgets.Layout(
            width='100%', 
            height='80px', 
            border=f'1px solid {COLORS["border"]}',
            border_radius='4px',
            padding='10px'
        )
    )
    
    custom_packages_card = widgets.VBox(
        [custom_packages_header, custom_packages],
        layout=card_container
    )
    
    # Tombol aksi
    check_all_button = widgets.Button(
        description='Check All', 
        button_style='info', 
        icon='check-square',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    uncheck_all_button = widgets.Button(
        description='Uncheck All', 
        button_style='warning', 
        icon='square',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    install_button = widgets.Button(
        description='Install Packages', 
        button_style='primary', 
        icon='download',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    check_button = widgets.Button(
        description='Check Installations', 
        button_style='success', 
        icon='check',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    buttons = widgets.HBox(
        [check_all_button, uncheck_all_button, install_button, check_button],
        layout=GROUP_LAYOUTS['horizontal']
    )
    
    # Progress bar
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:', 
        bar_style='info',
        layout=widgets.Layout(
            width='100%', 
            visibility='hidden',
            height='25px',
            margin='10px 0'
        )
    )
    
    # Status output
    status = widgets.Output(layout=output_area)
    
    # Info box
    info_box = create_info_box(
        "Tentang Package Installation", 
        """
        <p>Package diurutkan instalasi dari kecil ke besar:</p>
        <ol>
            <li>Notebook tools (ipywidgets, tqdm)</li>
            <li>Utility packages (pyyaml, termcolor)</li>
            <li>Data processing (matplotlib, pandas)</li>
            <li>Computer vision (OpenCV, Albumentations)</li>
            <li>Machine learning (PyTorch)</li>
        </ol>
        """, 
        style='info',
        collapsed=True
    )
    
    # Container utama
    main = widgets.VBox([
        header,
        status_panel,
        packages_container,
        custom_packages_card,
        buttons,
        progress,
        status,
        info_box
    ], layout=main_container)
    
    # Construct checkboxes mapping
    checkboxes = {}
    for i, (group_name, packages) in enumerate(package_groups.items()):
        group_widget = packages_container.children[i]
        for j, (desc, key) in enumerate(packages):
            checkboxes[key] = group_widget.children[1].children[j]
    
    return {
        'ui': main,
        'status': status,
        'status_panel': status_panel,
        'install_progress': progress,
        'install_button': install_button,
        'check_button': check_button,
        'check_all_button': check_all_button,
        'uncheck_all_button': uncheck_all_button,
        'custom_packages': custom_packages,
        **checkboxes
    }