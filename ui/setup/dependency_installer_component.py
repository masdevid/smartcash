"""
File: smartcash/ui/setup/dependency_installer_component.py
Deskripsi: Komponen UI untuk instalasi dependencies
"""

import ipywidgets as widgets
from smartcash.ui.components.headers import create_header, create_section_title
from smartcash.ui.components.alerts import create_info_box
from smartcash.ui.components.widget_layouts import (
    main_container, 
    output_area, 
    card_container, 
    create_layout
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
    
    # Buat grid package dengan layout yang lebih modern
    package_grid = widgets.GridBox(
        layout=create_layout(
            display='grid',
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='15px',
            width='100%',
            background_color=COLORS['light'],
            padding='15px',
            border_radius='8px'
        )
    )
    
    # Konstruksi package group widgets
    package_group_widgets = []
    for group_name, packages in package_groups.items():
        group = widgets.VBox(
            [widgets.HTML(f"<h4 style='color:{COLORS['secondary']}'>{ICONS['config']} {group_name.upper()} Packages</h4>")],
            layout=card_container
        )
        
        for desc, key in packages:
            checkbox = widgets.Checkbox(
                value=True, 
                description=desc,
                layout=create_layout(
                    width='auto', 
                    margin='5px 0',
                    color=COLORS['dark']
                )
            )
            group.children += (checkbox,)
        
        package_group_widgets.append(group)
    
    package_grid.children = package_group_widgets
    
    # Custom package input
    custom_packages = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)',
        layout=create_layout(
            width='100%', 
            height='100px', 
            border=f'1px solid {COLORS["border"]}',
            border_radius='4px',
            padding='10px'
        )
    )
    
    # Tombol aksi dengan styling yang konsisten
    buttons = widgets.HBox([
        widgets.Button(
            description='Check All', 
            button_style='info', 
            icon='check-square',
            layout=create_layout(margin='0 5px')
        ),
        widgets.Button(
            description='Uncheck All', 
            button_style='warning', 
            icon='square',
            layout=create_layout(margin='0 5px')
        ),
        widgets.Button(
            description='Install Packages', 
            button_style='primary', 
            icon='download',
            layout=create_layout(margin='0 5px')
        ),
        widgets.Button(
            description='Check Installations', 
            button_style='success', 
            icon='check',
            layout=create_layout(margin='0 5px')
        )
    ])
    
    # Progress bar dengan styling yang lebih menarik
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:', 
        bar_style='info',
        layout=create_layout(
            width='100%', 
            visibility='hidden',
            border_radius='4px',
            height='25px'
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
        style='info'
    )
    
    # Container utama
    main = widgets.VBox([
        header,
        package_grid,
        create_section_title('📝 Custom Packages'),
        custom_packages,
        buttons,
        progress,
        status,
        info_box
    ], layout=main_container)
    
    # Return UI components
    return {
        'ui': main,
        'status': status,
        'install_progress': progress,
        'install_button': buttons.children[2],
        'check_button': buttons.children[3],
        'check_all_button': buttons.children[0],
        'uncheck_all_button': buttons.children[1],
        'custom_packages': custom_packages,
        **{key: group.children[i+1] for group in package_grid.children 
           for i, (_, key) in enumerate(list(package_groups.values())[list(package_groups.keys()).index(group.children[0].value.split()[-1].lower())])}
    }