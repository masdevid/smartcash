"""
File: smartcash/ui/dataset/split/components/ui_components.py
Deskripsi: Komponen UI untuk dataset split dengan layout lengkap menggunakan shared components
"""

import ipywidgets as widgets
from typing import Dict, Any

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components import create_log_accordion

# Import split specific components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout

def create_split_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    🎨 Buat komponen UI untuk dataset split configuration
    
    Args:
        config: Configuration dictionary untuk split settings
        
    Returns:
        Dictionary berisi semua komponen UI
    """
    # Initialize components dictionary
    ui_components = {}
    
    # Use default config if none provided
    if config is None:
        config = {
            'data': {
                'split_ratios': {
                    'train': 0.7,
                    'valid': 0.15,
                    'test': 0.15
                },
                'stratified_split': True,
                'random_seed': 42
            },
            'split_settings': {
                'dataset_path': 'data',
                'preprocessed_path': 'data/preprocessed',
                'backup_dir': 'data/splits_backup',
                'backup_before_split': True
            }
        }
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="📊 Dataset Split Configuration",
        subtitle="Konfigurasi pembagian dataset untuk training, validation, dan testing"
    )
    ui_components['header_container'] = header_container
    
    # 2. Create Form Components
    form_components = create_split_form(config)
    
    # 3. Create Form Container
    form_container = create_form_container()
    
    # Create Form Layout and place it in the form container
    split_layout = create_split_layout(form_components)
    form_container['form_container'].children = (split_layout['container'],)
    
    # Store form components in ui_components
    ui_components.update(form_components)
    ui_components['form_container'] = form_container['container']
    
    # No summary container needed for split UI
    
    # 4. Create Log Accordion
    log_accordion = create_log_accordion()
    ui_components['log_accordion'] = log_accordion
    ui_components['log_output'] = log_accordion  # Alias for compatibility
    
    # 5. Create Footer Container with ONLY log_output and info_box
    footer_container = create_footer_container(
        show_buttons=False,  # No buttons in footer
        log_accordion=log_accordion,
        info_box=widgets.HTML(
            value="""
            <div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8; margin: 10px 0;">
                <h5>ℹ️ Tips</h5>
                <ul>
                    <li>Pastikan total split ratio = 1.0</li>
                    <li>Gunakan stratified split untuk dataset yang tidak seimbang</li>
                    <li>Aktifkan backup jika ingin menyimpan dataset asli</li>
                </ul>
            </div>
            """
        )
    )
    ui_components['footer_container'] = footer_container
    
    # 6. Assemble Main Container
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=ui_components['form_container'],
        footer_container=footer_container.container
    )
    
    # Store the main UI container
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
        
    return ui_components
