"""
File: smartcash/ui/dataset/downloader/components/ui_layout.py
Deskripsi: Main UI layout untuk downloader dengan flex/grid layout
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.common.environment import get_environment_manager

from .ui_form import create_form_fields

def create_downloader_ui(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Membuat UI downloader dengan flex/grid layout."""
    config = config or {}
    
    # Environment info
    env_manager = env or get_environment_manager()
    drive_status = "🔗 Drive terhubung" if env_manager.is_drive_mounted else "⚠️ Drive tidak terhubung"
    storage_info = f" | Storage: {'Drive' if env_manager.is_drive_mounted else 'Local'}"
    
    # Components
    header = create_header(
        f"{ICONS.get('download', '📥')} Dataset Downloader", 
        f"Download dataset untuk SmartCash{storage_info}"
    )
    
    initial_status = f"{drive_status} - Siap untuk download dataset"
    status_panel = create_status_panel(initial_status, "info")
    
    # Form fields
    form_fields = create_form_fields(config)
    
    # Action buttons
    action_buttons = create_action_buttons()
    
    # Log section
    log_section = create_log_section()
    
    # Save/Reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi download saat ini",
        reset_tooltip="Reset konfigurasi ke pengaturan default",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan saat disimpan atau direset.",
        button_width="120px",
        container_width="100%"
    )
    
    # Progress tracking
    progress_components = create_progress_tracking_container()
    
    # Main layout dengan flex/grid
    main_container = widgets.VBox([
        header,
        status_panel,
        _create_storage_info_widget(env_manager),
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        _create_options_panel(form_fields),
        save_reset_buttons['container'],
        create_divider(),
        action_buttons['container'],
        log_section['confirmation_area'], 
        progress_components['container'],
        log_section['log_accordion'],
        log_section['summary_container']
    ], layout=widgets.Layout(
        width='100%', display='flex', flex_flow='column',
        align_items='stretch', padding='10px', border='1px solid #ddd',
        border_radius='5px', background_color='#fff'
    ))
    
    # Combine all UI components
    ui_components = {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'drive_info': main_container.children[2],
        'module_name': 'downloader',
        'env_manager': env_manager,
    }
    
    # Add progress components
    ui_components.update(progress_components)
    
    # Add form fields
    ui_components.update(form_fields)
    
    # Add action buttons
    ui_components.update({
        'download_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
    })
    
    # Add log section
    ui_components.update(log_section)
    
    # Verify progress integration
    _verify_progress_integration(ui_components)
    
    return ui_components

def _create_storage_info_widget(env_manager):
    """Membuat widget untuk info storage."""
    if env_manager.is_drive_mounted:
        info_html = f"""
        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive: {env_manager.drive_path}</span>
        </div>
        """
    else:
        info_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal (hilang saat restart)</span>
        </div>
        """
    return widgets.HTML(info_html)

def _create_options_panel(form_fields: Dict[str, Any]) -> widgets.Widget:
    """Membuat panel opsi dengan flex/grid layout."""
    
    # 📊 Left Column: Dataset Information
    dataset_info_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 8px;">📊 Dataset Information</h4>'),
        form_fields['workspace_field'], 
        form_fields['project_field'], 
        form_fields['version_field'], 
        form_fields['api_key_field']
    ], layout=widgets.Layout(
        width='calc(50% - 8px)',
        margin='0 4px 0 0',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # ⚙️ Right Column: Process Options
    process_options_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #28a745; padding-bottom: 8px;">⚙️ Process Options</h4>'),
        form_fields['organize_dataset'],
        form_fields['backup_checkbox'],
        widgets.HTML('<div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px; border-left: 4px solid #2196f3;"><small style="color: #0d47a1;"><strong>🔍 Verifikasi:</strong> Setelah download selesai, gunakan tombol <strong>"Check Dataset"</strong> untuk memverifikasi hasil.</small></div>'),
        widgets.HTML('<div style="margin-top: 8px; padding: 10px; background: #e8f5e8; border-radius: 4px; border-left: 4px solid #28a745;"><small style="color: #155724;"><strong>💡 Organisasi:</strong> Dataset akan otomatis dipindah ke struktur final /data/train, /data/valid, /data/test</small></div>')
    ], layout=widgets.Layout(
        width='calc(50% - 8px)',
        margin='0 0 0 4px',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # 📁 Row 1: Two Columns Layout
    row1_container = widgets.HBox([
        dataset_info_section,
        process_options_section
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='stretch',
        margin='0 0 15px 0',
        overflow='hidden'
    ))
    
    # 📁 Row 2: Directory Settings (Full Width)
    directory_settings_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #ffc107; padding-bottom: 8px;">📁 Directory Settings</h4>'),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML('<label style="font-weight: bold; color: #495057; margin-bottom: 5px; display: block;">📥 Download Directory</label>'),
                form_fields['output_dir_field'],
                widgets.HTML('<small style="color: #6c757d; margin-top: 5px; display: block;">Lokasi sementara untuk download dataset sebelum diorganisir</small>')
            ], layout=widgets.Layout(width='calc(50% - 8px)', margin='0 4px 0 0', overflow='hidden')),
            widgets.VBox([
                widgets.HTML('<label style="font-weight: bold; color: #495057; margin-bottom: 5px; display: block;">💾 Backup Directory</label>'),
                form_fields['backup_dir_field'],
                widgets.HTML('<small style="color: #6c757d; margin-top: 5px; display: block;">Lokasi penyimpanan backup dataset lama (jika diperlukan)</small>')
            ], layout=widgets.Layout(width='calc(50% - 8px)', margin='0 0 0 4px', overflow='hidden'))
        ], layout=widgets.Layout(width='100%', overflow='hidden', flex_flow='row', justify_content='space-between', gap='8px')),
        form_fields['structure_info']
    ], layout=widgets.Layout(
        width='100%',
        margin='0',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # 📦 Main Panel Container
    panel = widgets.VBox([
        row1_container,
        directory_settings_section
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        overflow='hidden'
    ))
    
    return panel

def create_action_buttons():
    """Membuat action buttons dengan styling konsisten."""
    
    # Primary button
    download_button = widgets.Button(
        description='Download Dataset',
        icon='download',
        button_style='primary',
        tooltip='Download dataset dari Roboflow',
        layout=widgets.Layout(width='auto')
    )
    
    # Secondary buttons
    check_button = widgets.Button(
        description='Check Dataset',
        icon='search',
        button_style='info',
        tooltip='Periksa dataset yang sudah didownload',
        layout=widgets.Layout(width='auto')
    )
    
    cleanup_button = widgets.Button(
        description='Hapus Hasil',
        icon='trash',
        button_style='danger',
        tooltip='Hapus file hasil download tanpa backup',
        layout=widgets.Layout(width='auto')
    )
    
    # Button container
    button_container = widgets.HBox([
        download_button,
        check_button,
        cleanup_button
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        margin='10px 0',
        gap='10px'
    ))
    
    return {
        'container': button_container,
        'download_button': download_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button
    }

def create_log_section():
    """Membuat log section dengan styling konsisten."""
    
    # Confirmation area
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            min_height='50px',
            max_height='200px',
            margin='10px 0',
            padding='10px',
            border='1px solid #ddd',
            border_radius='5px',
            overflow='auto',
            display='none'
        )
    )
    
    # Log accordion
    log_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            max_height='300px',
            overflow='auto',
            border='1px solid #ddd',
            padding='10px'
        )
    )
    
    log_accordion = widgets.Accordion(
        children=[log_output],
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    log_accordion.set_title(0, '📋 Log')
    log_accordion.selected_index = None  # Collapsed by default
    
    # Summary container
    summary_container = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            min_height='50px',
            margin='10px 0',
            padding='10px',
            border='1px solid #ddd',
            border_radius='5px',
            overflow='auto',
            display='none'
        )
    )
    
    return {
        'confirmation_area': confirmation_area,
        'log_output': log_output,
        'log_accordion': log_accordion,
        'summary_container': summary_container
    }

def _verify_progress_integration(ui_components: Dict[str, Any]) -> None:
    """Verifikasi latest progress tracking methods tersedia."""
    # Expected methods dari latest ProgressTracker
    expected_methods = [
        'show_for_operation', 'update_progress', 'complete_operation', 
        'error_operation', 'reset_all', 'tracker'
    ]
    
    missing_methods = []
    for method in expected_methods:
        if method not in ui_components:
            missing_methods.append(method)
    
    logger = ui_components.get('logger')
    if missing_methods and logger:
        logger.warning(f"⚠️ Missing progress methods: {', '.join(missing_methods)}")
        logger.info("🔧 Fallback progress handling akan digunakan")
    elif logger:
        logger.debug("✅ Latest progress tracking methods tersedia")
    
    # Add fallback methods jika diperlukan
    _add_progress_fallbacks(ui_components, missing_methods)

def _add_progress_fallbacks(ui_components: Dict[str, Any], missing_methods: list) -> None:
    """Menambahkan fallback methods untuk missing progress tracking."""
    
    # Simple fallback implementations
    fallback_implementations = {
        'show_for_operation': lambda operation: None,
        'update_progress': lambda progress_type, value, message, color=None: None,
        'complete_operation': lambda message: None,
        'error_operation': lambda message: None,
        'reset_all': lambda: None
    }
    
    for method in missing_methods:
        if method in fallback_implementations:
            ui_components[method] = fallback_implementations[method]
    
    # Log fallback usage
    if missing_methods:
        logger = ui_components.get('logger')
        logger and logger.info(f"🆘 Added fallback methods: {', '.join(missing_methods)}")
