"""
File: smartcash/ui_components/dependency_installer.py
Author: Refactored
Deskripsi: Komponen UI untuk instalasi dependencies SmartCash dengan layout sederhana dan efisien.
"""

import ipywidgets as widgets
from IPython.display import HTML

def create_dependency_installer_ui():
    """
    Buat komponen UI untuk instalasi dependencies.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
              border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
        <h2 style="color: inherit; margin-top: 0;">📦 Package Installation</h2>
        <p style="color: inherit; margin-bottom: 0;">Instalasi package yang diperlukan untuk SmartCash</p>
    </div>
    """)
    
    # Package groups dengan layout grid
    package_groups = widgets.GridBox(
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='10px',
            width='100%'
        )
    )
    
    # Core packages
    core_group = widgets.VBox([
        widgets.HTML("<h4>🛠️ Core Packages</h4>"),
        widgets.Checkbox(value=True, description='YOLOv5 requirements'),
        widgets.Checkbox(value=True, description='SmartCash utils'),
        widgets.Checkbox(value=True, description='Notebook tools')
    ])
    
    # ML packages
    ml_group = widgets.VBox([
        widgets.HTML("<h4>🧠 ML Packages</h4>"),
        widgets.Checkbox(value=True, description='PyTorch'),
        widgets.Checkbox(value=True, description='OpenCV'),
        widgets.Checkbox(value=True, description='Albumentations')
    ])
    
    # Viz packages
    viz_group = widgets.VBox([
        widgets.HTML("<h4>📊 Visualization</h4>"),
        widgets.Checkbox(value=True, description='Matplotlib'),
        widgets.Checkbox(value=True, description='Pandas'),
        widgets.Checkbox(value=True, description='Seaborn')
    ])
    
    package_groups.children = [core_group, ml_group, viz_group]
    
    # Custom packages
    custom_area = widgets.Textarea(
        placeholder='Tambahkan package tambahan (satu per baris)',
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    # Options
    options = widgets.HBox([
        widgets.Checkbox(value=False, description='Force reinstall')
    ], layout=widgets.Layout(margin='10px 0'))
    
    # Buttons
    buttons = widgets.HBox([
        widgets.Button(
            description='Check All',
            button_style='info',
            icon='check-square',
            layout=widgets.Layout(margin='0 5px 0 0')
        ),
        widgets.Button(
            description='Uncheck All',
            button_style='warning',
            icon='square',
            layout=widgets.Layout(margin='0 5px')
        ),
        widgets.Button(
            description='Install Packages',
            button_style='primary',
            icon='download',
            layout=widgets.Layout(margin='0 5px')
        ),
        widgets.Button(
            description='Check Installations',
            button_style='success',
            icon='check',
            layout=widgets.Layout(margin='0 0 0 5px')
        )
    ], layout=widgets.Layout(margin='10px 0'))
    
    # Progress bar
    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Installing:',
        bar_style='info',
        layout={'width': '100%', 'visibility': 'hidden'}
    )
    
    # Status output
    status = widgets.Output(
        layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '150px', 'margin': '10px 0'}
    )
    
    # Info box
    info_box = widgets.HTML("""
    <div style="padding: 10px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
             color: #0c5460; margin: 10px 0; border-radius: 4px;">
        <h4 style="margin-top: 0; color: inherit;">ℹ️ Package Installation</h4>
        <p>Package diurutkan instalasi dari kecil ke besar untuk efisiensi:</p>
        <ol>
            <li><strong>Notebook tools</strong>: ipywidgets, tqdm (kecil, diperlukan UI)</li>
            <li><strong>Utility packages</strong>: pyyaml, termcolor (kecil, diperlukan)</li>
            <li><strong>Data processing</strong>: matplotlib, pandas (menengah)</li>
            <li><strong>Computer vision</strong>: OpenCV, Albumentations (besar)</li>
            <li><strong>Machine learning</strong>: PyTorch (paling besar)</li>
        </ol>
    </div>
    """)
    
    # Assemble UI
    main.children = [
        header,
        package_groups,
        widgets.HTML("<h4>📝 Custom Packages</h4>"),
        custom_area,
        options,
        buttons,
        progress,
        status,
        info_box
    ]
    
    # Create mapping for checkboxes
    checkboxes = {
        'yolov5_req': core_group.children[1],
        'smartcash_req': core_group.children[2],
        'notebook_req': core_group.children[3],
        'torch_req': ml_group.children[1],
        'opencv_req': ml_group.children[2],
        'albumentations_req': ml_group.children[3],
        'matplotlib_req': viz_group.children[1],
        'pandas_req': viz_group.children[2],
        'seaborn_req': viz_group.children[3]
    }
    
    # Return UI components
    return {
        'ui': main,
        'status': status,
        'install_progress': progress,
        'install_button': buttons.children[2],
        'check_button': buttons.children[3],
        'check_all_button': buttons.children[0],
        'uncheck_all_button': buttons.children[1],
        'custom_packages': custom_area,
        'force_reinstall': options.children[0],
        **checkboxes  # Include all checkboxes
    }