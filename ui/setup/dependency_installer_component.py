"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.headers import create_header
from smartcash.ui.components.alerts import create_status_indicator, create_info_box
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
    
    # Status panel
    status_panel = create_status_indicator(
        'info',
        'ℹ️ Pilih packages yang akan diinstall dan klik "Install Packages"'
    )
    
    # Buat UI untuk package groups
    package_section_widgets = []
    for group_name, packages in package_groups.items():
        # Header untuk grup
        group_header = widgets.HTML(f"""
        <div style="padding:5px 0">
            <h3 style="margin:5px 0;color:inherit">📌 {group_name.upper()} Packages</h3>
        </div>
        """)
        
        # Checkbox untuk packages
        checkboxes = []
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc,
                layout=widgets.Layout(padding='3px 0')
            )
            checkboxes.append(checkbox)
        
        # VBox untuk checkboxes
        checkboxes_group = widgets.VBox(
            checkboxes,
            layout=widgets.Layout(padding='5px 10px')
        )
        
        # Box untuk grup
        group_box = widgets.VBox(
            [group_header, checkboxes_group],
            layout=widgets.Layout(
                margin='5px',
                padding='10px',
                border='1px solid #dee2e6',
                border_radius='5px',
                width='150px'
            )
        )
        package_section_widgets.append(group_box)
    
    # Container untuk groups
    packages_container = widgets.HBox(
        package_section_widgets,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='space-between',
            width='100%'
        )
    )
    
    # Custom package input
    custom_header = widgets.HTML("""
    <div style="padding:5px 0">
        <h3 style="margin:5px 0;color:inherit">📝 Custom Packages</h3>
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
            border='1px solid #dee2e6',
            border_radius='5px'
        )
    )
    
    # Tombol aksi
    check_all_button = widgets.Button(
        description='Check All', 
        button_style='info', 
        icon='check-square',
        layout=widgets.Layout(margin='5px')
    )
    
    uncheck_all_button = widgets.Button(
        description='Uncheck All', 
        button_style='warning', 
        icon='square',
        layout=widgets.Layout(margin='5px')
    )
    
    install_button = widgets.Button(
        description='Install Packages', 
        button_style='primary', 
        icon='download',
        layout=widgets.Layout(margin='5px')
    )
    
    check_button = widgets.Button(
        description='Check Installations', 
        button_style='success', 
        icon='check',
        layout=widgets.Layout(margin='5px')
    )
    
    buttons = widgets.HBox(
        [check_all_button, uncheck_all_button, install_button, check_button],
        layout=widgets.Layout(
            display='flex',
            justify_content='center',
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
            margin='10px 0'
        )
    )
    progress.layout.visibility = 'hidden'
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #dee2e6',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Info box
    info_box = create_info_box  (
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
        style="info",
        collapsed=True
    )
    
    # Container utama
    main = widgets.VBox(
        [
            header,
            status_panel,
            packages_container,
            custom_section,
            buttons,
            progress,
            status,
            info_box
        ],
        layout=widgets.Layout(
            width='100%',
            padding='15px'
        )
    )
    
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