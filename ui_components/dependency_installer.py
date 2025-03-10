"""
File: smartcash/ui_components/dependency_installer.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk instalasi dependencies SmartCash dengan grid layout dan uncheck all button
"""

import ipywidgets as widgets
from IPython.display import HTML
from pathlib import Path

def create_dependency_ui():
    """Buat UI untuk instalasi dependencies dengan grid layout dan uncheck all button"""
    main = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("<h1>📦 Package Installation</h1><p>Instalasi package yang diperlukan untuk SmartCash</p>")
    
    # Package card
    package_card = widgets.VBox(layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='4px',
        padding='15px',
        margin='10px 0',
        width='100%'
    ))
    
    # Package list - checkboxes
    package_list_header = widgets.HTML("<h3>🛠️ Package List</h3>")
    
    # Checkbox group with grid layout
    checkbox_grid = widgets.GridBox(
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='10px',
            width='100%',
            padding='10px 0'
        )
    )
    
    # Create checkboxes for packages
    yolov5_req = widgets.Checkbox(
        value=True,
        description='YOLOv5 requirements',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    torch_req = widgets.Checkbox(
        value=True,
        description='PyTorch',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    albumentations_req = widgets.Checkbox(
        value=True,
        description='Albumentations',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    notebook_req = widgets.Checkbox(
        value=True,
        description='Notebook tools',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    smartcash_req = widgets.Checkbox(
        value=True,
        description='SmartCash requirements',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    opencv_req = widgets.Checkbox(
        value=True,
        description='OpenCV',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    matplotlib_req = widgets.Checkbox(
        value=True,
        description='Matplotlib',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    pandas_req = widgets.Checkbox(
        value=True,
        description='Pandas',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    seaborn_req = widgets.Checkbox(
        value=True,
        description='Seaborn',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Add checkboxes to grid
    checkbox_grid.children = [
        yolov5_req, torch_req, albumentations_req, 
        notebook_req, smartcash_req, opencv_req,
        matplotlib_req, pandas_req, seaborn_req
    ]
    
    # Button to check/uncheck all
    button_container = widgets.HBox(layout=widgets.Layout(
        justify_content='space-between',
        width='100%',
        margin='5px 0 15px 0'
    ))
    
    check_all_button = widgets.Button(
        description='Check All',
        button_style='primary',
        icon='check-square',
        layout=widgets.Layout(width='auto')
    )
    
    uncheck_all_button = widgets.Button(
        description='Uncheck All',
        button_style='warning',
        icon='square',
        layout=widgets.Layout(width='auto')
    )
    
    button_container.children = [check_all_button, uncheck_all_button]
    
    # Custom package list
    custom_packages = widgets.Textarea(
        value='',
        placeholder='Tambahkan package tambahan (satu per baris)',
        description='Custom:',
        disabled=False,
        layout=widgets.Layout(width='60%', height='100px', margin='10px 0')
    )
    
    # Install button and options
    action_button_group = widgets.HBox(layout=widgets.Layout(display='flex', flex_flow='row wrap'))
    
    install_button = widgets.Button(
        description='Install Packages',
        button_style='primary',
        icon='download',
        tooltip='Install all selected packages',
        layout=widgets.Layout(margin='10px 5px 10px 0')
    )
    
    check_button = widgets.Button(
        description='Check Installations',
        button_style='info',
        icon='check',
        tooltip='Check installed packages',
        layout=widgets.Layout(margin='10px 0 10px 5px')
    )
    
    # Add buttons to group
    action_button_group.children = [install_button, check_button]
    
    # Force reinstall option
    force_reinstall = widgets.Checkbox(
        value=False,
        description='Force reinstall',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Progress bar
    install_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Installing:',
        bar_style='info',
        orientation='horizontal',
        layout={'width': '100%', 'margin': '10px 0', 'visibility': 'hidden'}
    )
    
    # Status output
    status = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '150px', 'margin': '10px 0'})
    
    # Assemble package card
    package_card.children = [
        package_list_header,
        button_container,
        checkbox_grid,
        custom_packages,
        force_reinstall,
        action_button_group,
        install_progress
    ]
    
    # Help accordion
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Installation Guide</h4>
            <ol>
                <li><b>Select Packages:</b> Pilih package yang akan diinstall</li>
                <li><b>Custom Packages:</b> Tambahkan package khusus (satu per baris)</li>
                <li><b>Force Reinstall:</b> Paksa reinstall package meskipun sudah terinstall</li>
                <li><b>Check Installations:</b> Periksa versi package yang terinstall</li>
            </ol>
            <p><b>Catatan Package:</b></p>
            <ul>
                <li><b>YOLOv5 requirements:</b> Package dasar untuk YOLOv5 (numpy, opencv, scipy, dll)</li>
                <li><b>SmartCash requirements:</b> Package khusus untuk SmartCash (pyyaml, termcolor, tqdm)</li>
                <li><b>Notebook tools:</b> ipywidgets, tqdm, dan tools lain untuk notebook</li>
            </ul>
        </div>
    """)], selected_index=None)
    
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Assemble UI
    main.children = [
        header,
        package_card,
        status,
        help_info
    ]
    
    # Return components
    return {
        'ui': main,
        'yolov5_req': yolov5_req,
        'torch_req': torch_req,
        'albumentations_req': albumentations_req,
        'notebook_req': notebook_req,
        'smartcash_req': smartcash_req,
        'opencv_req': opencv_req,
        'matplotlib_req': matplotlib_req,
        'pandas_req': pandas_req,
        'seaborn_req': seaborn_req,
        'custom_packages': custom_packages,
        'force_reinstall': force_reinstall,
        'install_button': install_button,
        'check_button': check_button,
        'check_all_button': check_all_button,
        'uncheck_all_button': uncheck_all_button,
        'install_progress': install_progress,
        'status': status,
        'checkbox_grid': checkbox_grid
    }