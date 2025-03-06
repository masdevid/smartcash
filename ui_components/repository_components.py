"""
File: ui_components/repository_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk clone repository dan manajemen repository.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def create_repository_ui():
    """
    Buat UI komponen untuk clone dan manajemen repository.
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Buat header
    header = widgets.HTML("<h2>🔄 Clone Repository</h2>")
    description = widgets.HTML("<p>Clone repository SmartCash dan siapkan environment.</p>")
    
    # Repository URL
    default_repo_url = "https://github.com/smartcash-ai/smartcash.git"
    repo_url_input = widgets.Text(
        value=default_repo_url,
        placeholder='URL Repository',
        description='URL Repository:',
        disabled=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Gunakan custom repo
    custom_repo_checkbox = widgets.Checkbox(
        value=False,
        description='Gunakan repository custom',
        style={'description_width': 'initial'}
    )
    
    # Direktori output
    output_dir_input = widgets.Text(
        value='smartcash',
        placeholder='Direktori Output',
        description='Output Directory:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Branch
    branch_dropdown = widgets.Dropdown(
        options=['main', 'dev', 'stable', 'feature/enhanced-ui'],
        value='main',
        description='Branch:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Clone button
    clone_button = widgets.Button(
        description='Clone Repository',
        button_style='primary',
        icon='download',
        tooltip='Clone repository SmartCash'
    )
    
    # Install dependencies checkbox
    install_deps_checkbox = widgets.Checkbox(
        value=True,
        description='Install dependensi setelah clone',
        style={'description_width': 'initial'}
    )
    
    # Setup empty output
    output_area = widgets.Output()
    
    # Status indicator
    status_indicator = widgets.HTML(
        value="<p>Status: <span style='color: gray'>Siap</span></p>"
    )
    
    # Create layout
    repository_options = widgets.VBox([
        widgets.HBox([repo_url_input, custom_repo_checkbox]),
        widgets.HBox([output_dir_input, branch_dropdown]),
        widgets.HBox([clone_button, install_deps_checkbox]),
        status_indicator
    ])
    
    # Main UI
    main_ui = widgets.VBox([
        header,
        description,
        repository_options,
        output_area
    ])
    
    # Return komponen
    return {
        'ui': main_ui,
        'repo_url_input': repo_url_input,
        'custom_repo_checkbox': custom_repo_checkbox,
        'output_dir_input': output_dir_input,
        'branch_dropdown': branch_dropdown,
        'clone_button': clone_button,
        'install_deps_checkbox': install_deps_checkbox,
        'status_indicator': status_indicator,
        'output_area': output_area
    }