"""
File: smartcash/ui_components/fallback_repository_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI fallback untuk klon repository jika modul utama tidak tersedia
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def create_fallback_repository_ui():
    """
    Buat UI fallback sederhana untuk klon repository.
    
    Returns:
        Dictionary berisi komponen UI fallback
    """
    # Buat UI components
    repo_url = widgets.Text(
        value='https://github.com/smartcash-project/smartcash.git',
        description='Repository URL:',
        style={'description_width': 'initial'}
    )
    
    output_dir = widgets.Text(
        value='smartcash',
        description='Output Directory:',
        style={'description_width': 'initial'}
    )
    
    branch = widgets.Dropdown(
        options=['main', 'master', 'develop'],
        value='main',
        description='Branch:',
        style={'description_width': 'initial'}
    )
    
    install_deps = widgets.Checkbox(
        value=True,
        description='Install Dependencies'
    )
    
    clone_button = widgets.Button(
        description='Clone Repository',
        button_style='primary',
        icon='git-branch'
    )
    
    status_indicator = widgets.HTML(
        value="<p>Status: <span style='color: gray'>Siap</span></p>"
    )
    
    output_area = widgets.Output()
    
    # Buat layout
    fallback_ui = widgets.VBox([
        widgets.HTML("<h3>🚀 Clone Repository (Fallback UI)</h3>"),
        repo_url,
        output_dir,
        branch,
        install_deps,
        clone_button,
        status_indicator,
        output_area
    ])
    
    # Return komponen dalam dictionary untuk kemudahan penggunaan
    return {
        'ui': fallback_ui,
        'repo_url': repo_url,
        'output_dir': output_dir,
        'branch': branch,
        'install_deps': install_deps,
        'clone_button': clone_button,
        'status_indicator': status_indicator,
        'output_area': output_area
    }