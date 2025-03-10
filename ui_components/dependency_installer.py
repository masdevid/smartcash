"""
File: smartcash/ui_components/dependency_installer.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk instalasi dependencies SmartCash
"""

import ipywidgets as widgets
from IPython.display import HTML
from pathlib import Path

def create_dependency_ui():
    """Buat UI untuk instalasi dependencies"""
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
    
    yolov5_req = widgets.Checkbox(
        value=True,
        description='YOLOv5 requirements',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    torch_req = widgets.Checkbox(
        value=True,
        description='PyTorch (torch, torchvision)',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    albumentations_req = widgets.Checkbox(
        value=True,
        description='Albumentations (augmentation)',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    notebook_req = widgets.Checkbox(
        value=True,
        description='Notebook (ipywidgets, tqdm)',
        disabled=False,
        indent=False,
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Custom package list
    custom_packages = widgets.Textarea(
        value='',
        placeholder='Tambahkan package tambahan (satu per baris)',
        description='Custom:',
        disabled=False,
        layout=widgets.Layout(width='60%', height='100px', margin='10px 0')
    )
    
    # Install button and options
    button_group = widgets.HBox(layout=widgets.Layout(display='flex', flex_flow='row wrap'))
    
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
    button_group.children = [install_button, check_button]
    
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
        yolov5_req,
        torch_req,
        albumentations_req,
        notebook_req,
        custom_packages,
        force_reinstall,
        button_group,
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
            <p><b>Catatan:</b> YOLOv5 requirements sudah mencakup package seperti numpy, opencv, dan scipy.</p>
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
        'custom_packages': custom_packages,
        'force_reinstall': force_reinstall,
        'install_button': install_button,
        'check_button': check_button,
        'install_progress': install_progress,
        'status': status
    }