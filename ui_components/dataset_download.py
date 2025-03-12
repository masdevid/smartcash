"""
File: smartcash/ui_components/dataset_download.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk download dataset SmartCash dari berbagai sumber.
"""

import ipywidgets as widgets
from IPython.display import HTML

def create_dataset_download_ui():
    """
    Buat komponen UI untuk download dataset.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utamanya
    """
    # Container utama
    main_container = widgets.VBox([], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML(
        "<h2 style='color: #3498db; margin-bottom: 10px;'>📊 Dataset Download</h2>"
        "<p style='color: #555; margin-bottom: 15px;'>Download dataset untuk training model SmartCash</p>"
    )
    
    # Download options
    download_options = widgets.RadioButtons(
        options=['Roboflow (Online)', 'Local Data (Upload)'],
        description='Source:',
        style={'description_width': 'initial'},
    )
    
    # Roboflow settings
    roboflow_settings = widgets.VBox([
        widgets.Password(
            value='',
            description='API Key:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', display='none')  # Hidden by default
        ),
        widgets.Text(
            value='smartcash-wo2us',
            description='Workspace:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value='rupiah-emisi-2022',
            description='Project:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value='3',
            description='Version:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        )
    ])
    
    # Local upload widget
    local_upload = widgets.VBox([
        widgets.FileUpload(
            description='Upload ZIP:',
            accept='.zip',
            multiple=False,
            layout=widgets.Layout(width='300px')
        ),
        widgets.Text(
            value='data/uploaded',
            description='Target dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Download settings container (NEW!)
    download_settings_container = widgets.VBox([roboflow_settings])
    
    # Download button
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Status and progress
    download_status = widgets.Output(
        layout=widgets.Layout(width='100%', border='1px solid #ddd', min_height='100px', margin='10px 0')
    )
    
    download_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(margin='10px 0', width='100%')
    )
    
    # Help accordion
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Cara Download Dataset</h4>
            <ol>
                <li><b>Roboflow:</b> Masukkan API key Roboflow dan konfigurasi workspace/project</li>
                <li><b>Local:</b> Upload file ZIP yang berisi dataset dalam format YOLOv5</li>
            </ol>
        </div>
    """)], selected_index=None)
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Assemble UI
    main_container.children = [
        header,
        download_options,
        download_settings_container,  # Make sure this key exists
        download_button,
        download_progress,
        download_status,
        help_info
    ]
    
    # Return UI components dict
    ui_components = {
        'ui': main_container,
        'download_options': download_options,
        'roboflow_settings': roboflow_settings,
        'local_upload': local_upload,
        'download_button': download_button,
        'download_progress': download_progress,
        'download_status': download_status,
        'download_settings_container': download_settings_container  # Add this key
    }
    
    return ui_components