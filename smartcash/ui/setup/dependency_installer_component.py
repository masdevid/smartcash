"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies dengan layout checkbox yang lebih kompak
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_dependency_installer_ui(env=None, config=None):
    """Buat komponen UI untuk instalasi dependencies."""

    from smartcash.ui.utils.headers import create_header
    from smartcash.ui.utils.alerts import create_info_box
    
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
    
    # Status panel - Gunakan widgets.HTML bukan create_status_indicator yang mengembalikan HTML
    status_panel = widgets.HTML(value="""
        <div style="margin: 5px 0; padding: 8px 12px; 
                    border-radius: 4px; background-color: #f8f9fa;">
            <span style="color: #0c5460; font-weight: bold;"> 
                ℹ️ Pilih packages yang akan diinstall dan klik "Install Packages"
            </span>
        </div>
    """)
    
    # Buat UI untuk package groups dengan layout yang lebih kompak
    package_section_widgets = []
    for group_name, packages in package_groups.items():
        # Header untuk grup
        group_header = widgets.HTML(f"""
        <div style="padding:5px 0">
            <h3 style="margin:5px 0;color:inherit">📌 {group_name.upper()} Packages</h3>
        </div>
        """)
        
        # Buat checkboxes dengan layout grid untuk tampilan lebih kompak
        checkboxes = []
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc,
                indent=False,
                layout=widgets.Layout(padding='2px 0', width='100%')
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
                border='1px solid #dee2e6',
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
    
    # Tombol aksi dalam grid layout untuk tampilan lebih kompak
    button_layout = widgets.Layout(
        width='auto',
        margin='5px',
        height='auto'
    )

    check_all_button = widgets.Button(
        description='Check All', 
        button_style='info', 
        icon='check-square',
        layout=button_layout
    )
    
    uncheck_all_button = widgets.Button(
        description='Uncheck All', 
        button_style='warning', 
        icon='square',
        layout=button_layout
    )
    
    install_button = widgets.Button(
        description='Install Packages', 
        button_style='primary', 
        icon='download',
        layout=button_layout
    )
    
    check_button = widgets.Button(
        description='Check Installations', 
        button_style='success', 
        icon='check',
        layout=button_layout
    )
    
    buttons = widgets.HBox(
        [check_all_button, uncheck_all_button, install_button, check_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='center',
            align_items='center',
            margin='5px 0',
            gap='5px'
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
    
    info_box = create_info_box(
        f"Tentang Package Installation",
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
        'info',
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
            padding='10px'
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