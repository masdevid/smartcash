"""
File: smartcash/ui_components/dependency_installer.py
Author: Refactored
Deskripsi: Komponen UI untuk instalasi dependencies SmartCash dengan grid layout yang lebih fleksibel.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import create_component_header, create_info_box

def create_dependency_ui():
    """Buat UI untuk instalasi dependencies dengan grid layout yang fleksibel"""
    # Container utama
    main = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Package Installation",
        "Instalasi package yang diperlukan untuk SmartCash",
        "📦"
    )
    
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
    
    # Checkbox grid dengan layout yang lebih fleksibel
    checkbox_grid = widgets.GridBox(
        layout=widgets.Layout(
            grid_template_columns='repeat(auto-fill, minmax(200px, 1fr))',
            grid_gap='10px',
            width='100%',
            padding='10px 0'
        )
    )
    
    # Buat checkboxes untuk packages
    packages = [
        ('yolov5_req', 'YOLOv5 requirements'),
        ('torch_req', 'PyTorch'),
        ('albumentations_req', 'Albumentations'),
        ('notebook_req', 'Notebook tools'),
        ('smartcash_req', 'SmartCash requirements'),
        ('opencv_req', 'OpenCV'),
        ('matplotlib_req', 'Matplotlib'),
        ('pandas_req', 'Pandas'),
        ('seaborn_req', 'Seaborn')
    ]
    
    checkbox_widgets = {}
    for name, desc in packages:
        checkbox_widgets[name] = widgets.Checkbox(
            value=True,
            description=desc,
            disabled=False,
            indent=False,
            layout=widgets.Layout(margin='5px 0')
        )
    
    # Tambahkan checkboxes ke grid
    checkbox_grid.children = list(checkbox_widgets.values())
    
    # Button untuk check/uncheck all
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
        layout=widgets.Layout(width='100%', height='100px', margin='10px 0')
    )
    
    # Install button and options
    action_container = widgets.HBox(layout=widgets.Layout(
        justify_content='space-between',
        width='100%',
        margin='15px 0'
    ))
    
    action_button_group = widgets.HBox(layout=widgets.Layout(display='flex', flex_flow='row wrap'))
    
    install_button = widgets.Button(
        description='Install Packages',
        button_style='primary',
        icon='download',
        tooltip='Install all selected packages',
        layout=widgets.Layout(margin='0 5px 0 0')
    )
    
    check_button = widgets.Button(
        description='Check Installations',
        button_style='info',
        icon='check',
        tooltip='Check installed packages',
        layout=widgets.Layout(margin='0 0 0 5px')
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
    
    action_container.children = [action_button_group, force_reinstall]
    
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
    
    # Info box
    info_box = create_info_box(
        "Tentang Package Installation",
        """
        <p><b>Package yang tersedia:</b></p>
        <ul>
            <li><b>YOLOv5 requirements:</b> Package dasar untuk YOLOv5 (numpy, opencv, scipy)</li>
            <li><b>SmartCash requirements:</b> Package khusus untuk SmartCash (pyyaml, termcolor, tqdm)</li>
            <li><b>Notebook tools:</b> ipywidgets, tqdm, dan tools lain untuk notebook</li>
        </ul>
        <p><i>Gunakan <b>Force reinstall</b> untuk instalasi ulang package yang sudah ada.</i></p>
        """,
        'info'
    )
    
    # Assemble package card
    package_card.children = [
        package_list_header,
        button_container,
        checkbox_grid,
        widgets.HTML("<h3>📝 Additional Packages</h3>"),
        custom_packages,
        action_container,
        install_progress
    ]
    
    # Assemble UI
    main.children = [
        header,
        info_box,
        package_card,
        status
    ]
    
    # Return dictionary of components
    result = {
        'ui': main,
        'status': status,
        'install_progress': install_progress,
        'install_button': install_button,
        'check_button': check_button,
        'check_all_button': check_all_button,
        'uncheck_all_button': uncheck_all_button,
        'custom_packages': custom_packages,
        'force_reinstall': force_reinstall,
        'checkbox_grid': checkbox_grid
    }
    
    # Add individual checkboxes to result
    for name, widget in checkbox_widgets.items():
        result[name] = widget
    
    return result