"""
File: smartcash/ui_components/dataset.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk persiapan dataset SmartCash termasuk
           download, preprocessing, konfigurasi split, dan augmentasi data.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os
from pathlib import Path

from smartcash.utils.ui_utils import (
    create_header, 
    create_section_title, 
    create_info_alert,
    create_status_indicator,
    create_tab_view,
    create_info_box,
    create_loading_indicator,
    create_metric_display,
    styled_html
)

def create_dataset_preparation_ui():
    """
    Buat komponen UI untuk persiapan dataset SmartCash.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utamanya
    """
    # Container utama
    main_container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    
    # Header
    header = create_header(
        "📊 Dataset Preparation",
        "Persiapan dataset untuk training model SmartCash"
    )
    
    # Dataset download section
    download_section = create_section_title("2.1 - Dataset Download", "🔽")
    
    download_options = widgets.RadioButtons(
        options=['Roboflow (Online)', 'Local Data (Upload)', 'Sample Data'],
        description='Source:',
        style={'description_width': 'initial'},
    )
    
    # Roboflow settings
    roboflow_settings = widgets.VBox([
        widgets.Text(
            value='',
            description='API Key:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
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
    
    # Local upload widget (placeholder - in Jupyter actual upload would be used)
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
    
    # Sample data settings
    sample_data = widgets.VBox([
        widgets.Dropdown(
            options=['Default Sample (200 images)', 'Mini Sample (50 images)', 'Full Sample (500 images)'],
            description='Sample set:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Conditionally show settings based on selection
    download_settings_container = widgets.VBox([roboflow_settings])
    
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download'
    )
    
    download_status = widgets.Output()
    download_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal'
    )
    
    # Preprocessing section
    preprocess_section = create_section_title("2.2 - Preprocessing", "🔧")
    
    preprocess_options = widgets.VBox([
        widgets.IntRangeSlider(
            value=[640, 640],
            min=320,
            max=1280,
            step=32,
            description='Image size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=True,
            description='Enable normalization',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable caching',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            description='Workers:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    preprocess_button = widgets.Button(
        description='Run Preprocessing',
        button_style='primary',
        icon='cog'
    )
    
    preprocess_status = widgets.Output()
    
    # Split configuration section
    split_section = create_section_title("2.3 - Split Configuration", "✂️")
    
    split_options = widgets.VBox([
        widgets.BoundedFloatText(
            value=70.0,
            min=50.0,
            max=90.0,
            description='Train %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.BoundedFloatText(
            value=15.0,
            min=5.0,
            max=30.0,
            description='Validation %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.BoundedFloatText(
            value=15.0,
            min=5.0,
            max=30.0, 
            description='Test %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Stratified split (preserve class distribution)',
            style={'description_width': 'initial'}
        )
    ])
    
    split_button = widgets.Button(
        description='Apply Split',
        button_style='primary',
        icon='scissors'
    )
    
    split_status = widgets.Output()
    
    # Data augmentation section
    augmentation_section = create_section_title("2.4 - Data Augmentation", "🔄")
    
    augmentation_options = widgets.VBox([
        widgets.SelectMultiple(
            options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
            value=['Combined (Recommended)'],
            description='Augmentations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='100px')
        ),
        widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            description='Variations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Text(
            value='aug',
            description='Prefix:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Validate results',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Resume if interrupted',
            style={'description_width': 'initial'}
        )
    ])
    
    augmentation_button = widgets.Button(
        description='Run Augmentation',
        button_style='primary',
        icon='random'
    )
    
    augmentation_status = widgets.Output()
    augmentation_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal'
    )
    
    # Dataset stats section (appears after processes)
    stats_section = create_section_title("Dataset Statistics", "📈")
    stats_container = widgets.Output()
    
    # Setup tabs for viewing dataset details
    dataset_tabs = create_tab_view({
        '📊 Overview': widgets.Output(),
        '🏷️ Classes': widgets.Output(),
        '📏 Dimensions': widgets.Output(),
        '🔍 Samples': widgets.Output()
    })
    
    # Hide stats initially
    stats_section.layout.display = 'none'
    stats_container.layout.display = 'none'
    dataset_tabs.layout.display = 'none'
    
    # Observer and event logic for section dependencies
    def update_download_options(change):
        if change['new'] == 'Roboflow (Online)':
            download_settings_container.children = [roboflow_settings]
        elif change['new'] == 'Local Data (Upload)':
            download_settings_container.children = [local_upload]
        else:  # Sample Data
            download_settings_container.children = [sample_data]
    
    download_options.observe(update_download_options, names='value')
    
    # Pasang semua komponen
    main_container.children = [
        header,
        download_section,
        download_options,
        download_settings_container,
        widgets.HBox([download_button]),
        download_progress,
        download_status,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        preprocess_section,
        preprocess_options,
        widgets.HBox([preprocess_button]),
        preprocess_status,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        split_section,
        split_options,
        widgets.HBox([split_button]),
        split_status, 
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        augmentation_section,
        augmentation_options,
        widgets.HBox([augmentation_button]),
        augmentation_progress,
        augmentation_status,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        stats_section,
        stats_container,
        dataset_tabs
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        # Download components
        'download_options': download_options, 
        'roboflow_settings': roboflow_settings,
        'local_upload': local_upload,
        'sample_data': sample_data,
        'download_button': download_button,
        'download_progress': download_progress,
        'download_status': download_status,
        
        # Preprocessing components
        'preprocess_options': preprocess_options,
        'preprocess_button': preprocess_button,
        'preprocess_status': preprocess_status,
        
        # Split components
        'split_options': split_options,
        'split_button': split_button,
        'split_status': split_status,
        
        # Augmentation components
        'augmentation_options': augmentation_options,
        'augmentation_button': augmentation_button,
        'augmentation_progress': augmentation_progress,
        'augmentation_status': augmentation_status,
        
        # Stats components
        'stats_section': stats_section,
        'stats_container': stats_container,
        'dataset_tabs': dataset_tabs
    }
    
    return ui_components