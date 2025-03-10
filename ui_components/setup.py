"""
File: smartcash/ui_components/setup.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk setup project SmartCash
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import create_header, create_section_title

def create_project_setup_ui():
    """Buat UI setup project SmartCash"""
    main = widgets.VBox()
    
    # Header
    header = create_header("🚀 Project Setup", "Konfigurasi SmartCash")
    
    # Repo section
    repo_section = create_section_title("1.1 - Repository", "📦")
    repo_url = widgets.Text(value='https://github.com/masdevid/smartcash.git', description='URL:')
    clone_btn = widgets.Button(description='Clone', button_style='primary', icon='download')
    repo_status = widgets.Output()
    
    # Env section
    env_section = create_section_title("1.2 - Environment", "⚙️")
    env_type = widgets.RadioButtons(options=['Local', 'Google Colab'], description='Env:')
    colab_btn = widgets.Button(description='Connect Drive', button_style='info', icon='link', layout={'display': 'none'})
    env_status = widgets.Output()
    
    # Deps section
    deps_section = create_section_title("1.3 - Dependencies", "📚")
    packages = widgets.Textarea(value='torch>=1.7.0\ntorchvision>=0.8.1\nopencv-python>=4.5.1', description='Packages:')
    install_btn = widgets.Button(description='Install', button_style='primary', icon='cog')
    deps_status = widgets.Output()
    
    # Help
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <h4>Petunjuk:</h4>
        <ul>
            <li>Clone: Input URL & klik</li>
            <li>Env: Pilih tipe</li>
            <li>Deps: Set & install</li>
        </ul>
    """)], selected_index=None)
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Assemble
    main.children = [
        header,
        repo_section, widgets.HBox([repo_url, clone_btn]), repo_status,
        widgets.HTML("<hr>"),
        env_section, widgets.VBox([env_type, colab_btn]), env_status,
        widgets.HTML("<hr>"),
        deps_section, widgets.VBox([packages, install_btn]), deps_status,
        widgets.HTML("<hr>"),
        help_info,
        widgets.Output()
    ]
    
    return {
        'ui': main,
        'repo_url': repo_url, 'clone_button': clone_btn, 'repo_status': repo_status,
        'env_type': env_type, 'colab_connect_button': colab_btn, 'env_status': env_status,
        'required_packages': packages, 'install_button': install_btn, 'deps_status': deps_status,
        'overall_status': main.children[-1]
    }