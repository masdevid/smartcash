"""
File: smartcash/ui_components/project_setup.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk setup project SmartCash
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from pathlib import Path
from smartcash.utils.ui_utils import create_header, create_section_title, create_info_alert

def create_project_setup_ui():
    """Buat UI setup project SmartCash"""
    main = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_header("🚀 Project Setup", "Konfigurasi awal SmartCash untuk deteksi mata uang")
    
    # Repo section
    repo_section = create_section_title("1.1 - Repository", "📦")
    repo_url = widgets.Text(
        value='https://github.com/ultralytics/yolov5.git',
        description='YOLOv5 URL:',
        style={'description_width': 'initial'},
        layout={'width': '60%'}
    )
    
    smartcash_url = widgets.Text(
        value='https://github.com/yourusername/smartcash.git',
        description='SmartCash URL:',
        style={'description_width': 'initial'},
        layout={'width': '60%'}
    )
    
    clone_btn = widgets.Button(
        description='Clone Repos',
        button_style='primary',
        icon='download',
        tooltip='Clone both repositories'
    )
    
    repo_status = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '50px'})
    
    # Env section
    env_section = create_section_title("1.2 - Environment", "⚙️")
    env_type = widgets.RadioButtons(
        options=['Local', 'Google Colab'],
        description='Environment:',
        style={'description_width': 'initial'}
    )
    
    colab_btn = widgets.Button(
        description='Connect Drive',
        button_style='info',
        icon='link',
        tooltip='Connect to Google Drive',
        layout={'display': 'none'}
    )
    
    env_status = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '50px'})
    
    # Deps section
    deps_section = create_section_title("1.3 - Dependencies", "📚")
    packages = widgets.Textarea(
        value='torch>=1.7.0\ntorchvision>=0.8.1\nopencv-python>=4.5.1\npycocotools>=2.0.2\ntqdm>=4.64.0\nseaborn>=0.11.2\nalbumentations>=1.0.3\nPillow>=9.0.0',
        description='Required Packages:',
        style={'description_width': 'initial'},
        layout={'width': '60%', 'height': '150px'}
    )
    
    install_btn = widgets.Button(
        description='Install Dependencies',
        button_style='primary',
        icon='cog',
        tooltip='Install required packages'
    )
    
    deps_status = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '50px'})
    
    # Help accordion
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Petunjuk Setup:</h4>
            <ol>
                <li><b>Repository:</b> Klik tombol "Clone Repos" untuk mengunduh YOLOv5 dan SmartCash</li>
                <li><b>Environment:</b> Pilih environment yang sesuai (Local/Colab)</li>
                <li><b>Dependencies:</b> Klik "Install Dependencies" untuk menginstall package yang diperlukan</li>
            </ol>
            <p><i>Note: Jika menggunakan Google Colab, hubungkan ke Google Drive terlebih dahulu</i></p>
        </div>
    """)], selected_index=None)
    
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Overall status output
    overall_status = widgets.Output(layout={'width': '100%', 'padding': '10px'})
    
    # Assemble UI
    main.children = [
        header,
        repo_section, 
        widgets.VBox([repo_url, smartcash_url, clone_btn]), 
        repo_status,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        env_section, 
        widgets.VBox([env_type, colab_btn]), 
        env_status,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        deps_section, 
        widgets.VBox([packages, install_btn]), 
        deps_status,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        help_info,
        overall_status
    ]
    
    # Return UI components dictionary
    return {
        'ui': main,
        'repo_url': repo_url, 
        'smartcash_url': smartcash_url,
        'clone_button': clone_btn, 
        'repo_status': repo_status,
        'env_type': env_type, 
        'colab_connect_button': colab_btn, 
        'env_status': env_status,
        'required_packages': packages, 
        'install_button': install_btn, 
        'deps_status': deps_status,
        'overall_status': overall_status
    }