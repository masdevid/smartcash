"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies SmartCash
"""

import ipywidgets as widgets
from smartcash.ui.components.headers import create_header, create_section_title
from smartcash.ui.components.alerts import create_info_box
from smartcash.ui.components.widget_layouts import (
    main_container, button, section_container, output_area, create_divider
)
from smartcash.ui.utils.constants import COLORS, ICONS

def create_dependency_installer_ui(env=None, config=None):
    """
    Buat komponen UI untuk instalasi dependencies.
    
    Args:
        env: Environment manager (opsional)
        config: Konfigurasi (opsional)
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Header dengan styling konsisten
    header = create_header(
        "📦 Instalasi Dependencies", 
        "Setup package yang diperlukan untuk SmartCash"
    )
    
    # Package groups
    package_groups = widgets.GridBox(
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='10px',
            width='100%'
        )
    )
    
    # Definisi kelompok package
    package_groups_config = {
        'Core Packages': [
            ('YOLOv5 requirements', 'yolov5_req'),
            ('SmartCash utils', 'smartcash_req'),
            ('Notebook tools', 'notebook_req')
        ],
        'ML Packages': [
            ('PyTorch', 'torch_req'),
            ('OpenCV', 'opencv_req'),
            ('Albumentations', 'albumentations_req')
        ],
        'Visualization': [
            ('Matplotlib', 'matplotlib_req'),
            ('Pandas', 'pandas_req'),
            ('Seaborn', 'seaborn_req')
        ]
    }
    
    # Buat kelompok package
    package_group_widgets = []
    for group_name, packages in package_groups_config.items():
        group = widgets.VBox([
            widgets.HTML(f"<h4>{ICONS['tools']} {group_name}</h4>")
        ])
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc,
                layout=widgets.Layout(width='auto')
            )
            group.children += (checkbox,)
        package_group_widgets.append(group)
    
    package_groups.children = package_group_widgets
    
    # Custom package area
    custom_area = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)',
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    # Progress bar
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:', 
        bar_style='info',
        layout={'width': '100%', 'visibility': 'hidden'}
    )
    
    # Tombol aksi
    buttons = widgets.HBox([
        widgets.Button(description='Check All', button_style='info', icon='check-square'),
        widgets.Button(description='Uncheck All', button_style='warning', icon='square'),
        widgets.Button(description='Install Packages', button_style='primary', icon='download'),
        widgets.Button(description='Check Installations', button_style='success', icon='check')
    ])
    
    # Status output
    status = widgets.Output(layout=output_area)
    
    # Info box
    info_box = create_info_box(
        "Tentang Package Installation", 
        """
        <p>Package diurutkan instalasi dari kecil ke besar untuk efisiensi:</p>
        <ol>
            <li><strong>Notebook tools</strong>: ipywidgets, tqdm (kecil, diperlukan UI)</li>
            <li><strong>Utility packages</strong>: pyyaml, termcolor (kecil, diperlukan)</li>
            <li><strong>Data processing</strong>: matplotlib, pandas (menengah)</li>
            <li><strong>Computer vision</strong>: OpenCV, Albumentations (besar)</li>
            <li><strong>Machine learning</strong>: PyTorch (paling besar)</li>
        </ol>
        """, 
        style='info',
        collapsed=False
    )
    
    # Buat container utama
    main = widgets.VBox([
        header,
        package_groups,
        create_section_title('📝 Custom Packages'),
        custom_area,
        buttons,
        progress,
        status,
        info_box
    ], layout=main_container)
    
    # Return UI components dengan referensi yang diperlukan
    return {
        'ui': main,
        'status': status,
        'install_progress': progress,
        'install_button': buttons.children[2],
        'check_button': buttons.children[3],
        'check_all_button': buttons.children[0],
        'uncheck_all_button': buttons.children[1],
        'custom_packages': custom_area,
        **{key: group.children[i+1] for group in package_groups.children 
           for i, (_, key) in enumerate(list(package_groups_config.values())[list(package_groups_config.keys()).index(group.children[0].value)])},
    }