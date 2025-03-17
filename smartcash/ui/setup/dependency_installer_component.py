"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_dependency_installer_ui(env=None, config=None):
    """Buat komponen UI untuk instalasi dependencies."""
    # Header 
    header = widgets.HTML("""<div style="background:#f8f9fa;padding:15px;border-radius:5px;border-left:5px solid #3498db;margin-bottom:15px"><h1 style="margin:0;color:#2c3e50">📦 Instalasi Dependencies</h1><p style="margin:5px 0;color:#7f8c8d">Setup package yang diperlukan untuk SmartCash</p></div>""")
    
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
    status_panel = widgets.HTML("""<div style="padding:10px;margin:10px 0;background-color:#d1ecf1;color:#0c5460;border-radius:4px">ℹ️ Pilih packages yang akan diinstall dan klik 'Install Packages'</div>""")
    
    # Buat UI untuk package groups
    package_section_widgets = []
    for group_name, packages in package_groups.items():
        # Header untuk grup
        group_header = widgets.HTML(f"<h3 style='margin:10px 0'>📌 {group_name.upper()} Packages</h3>")
        
        # Checkbox untuk packages
        checkboxes = []
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc
            )
            checkboxes.append(checkbox)
        
        # VBox untuk checkboxes
        checkboxes_group = widgets.VBox(checkboxes)
        
        # Box untuk grup
        group_box = widgets.VBox([group_header, checkboxes_group])
        package_section_widgets.append(group_box)
    
    # Container untuk groups
    packages_container = widgets.HBox(package_section_widgets)
    
    # Custom package input
    custom_header = widgets.HTML("<h3 style='margin:10px 0'>📝 Custom Packages</h3>")
    custom_packages = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)'
    )
    
    # Tombol aksi
    check_all_button = widgets.Button(
        description='Check All', 
        button_style='info', 
        icon='check-square'
    )
    
    uncheck_all_button = widgets.Button(
        description='Uncheck All', 
        button_style='warning', 
        icon='square'
    )
    
    install_button = widgets.Button(
        description='Install Packages', 
        button_style='primary', 
        icon='download'
    )
    
    check_button = widgets.Button(
        description='Check Installations', 
        button_style='success', 
        icon='check'
    )
    
    buttons = widgets.HBox([check_all_button, uncheck_all_button, install_button, check_button])
    
    # Progress bar
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:'
    )
    progress.layout.visibility = 'hidden'
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box
    info_box = widgets.HTML("""
    <div style="padding:10px;margin:10px 0;background-color:#d1ecf1;color:#0c5460;border-radius:4px">
        <h4 style="margin-top:0">ℹ️ Tentang Package Installation</h4>
        <p>Package diurutkan instalasi dari kecil ke besar:</p>
        <ol>
            <li>Notebook tools (ipywidgets, tqdm)</li>
            <li>Utility packages (pyyaml, termcolor)</li>
            <li>Data processing (matplotlib, pandas)</li>
            <li>Computer vision (OpenCV, Albumentations)</li>
            <li>Machine learning (PyTorch)</li>
        </ol>
    </div>
    """)
    
    # Container utama
    main = widgets.VBox([
        header,
        status_panel,
        packages_container,
        custom_header,
        custom_packages,
        buttons,
        progress,
        status,
        info_box
    ])
    
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