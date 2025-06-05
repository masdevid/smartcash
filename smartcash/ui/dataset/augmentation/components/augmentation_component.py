"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: UI component dengan service integration dan reuse existing components
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    UI assembly dengan service integration dan component reuse.
    
    Returns:
        Dictionary berisi widget UI dengan service support
    """
    # Import reused components
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.augmentation_info import get_augmentation_info
    from smartcash.ui.utils.layout_utils import create_divider
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.dataset.augmentation.components.basic_options_widget import create_basic_options_widget
    from smartcash.ui.dataset.augmentation.components.advanced_options_widget import create_advanced_options_widget
    from smartcash.ui.dataset.augmentation.components.augmentation_types_widget import create_augmentation_types_widget
    
    from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES

    MODULE_LOGGER_NAME = KNOWN_NAMESPACES[AUGMENTATION_LOGGER_NAMESPACE]
    
    # Header dengan service context
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Augmentasi dataset dengan service integration untuk performa optimal")

    # Status panel dengan service info
    status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")

    # Widget groups dengan service support
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()

    # Control buttons dengan service integration
    config_buttons = create_save_reset_buttons(
        save_label="Simpan Config", reset_label="Reset Config",
        save_tooltip="Simpan konfigurasi augmentation",
        reset_tooltip="Reset ke konfigurasi default",
        with_sync_info=True, button_width="120px"
    )

    action_buttons = create_action_buttons(
        primary_label="🎯 Run Augmentation Pipeline", primary_icon="action",
        secondary_buttons=[
            ("🔍 Check Dataset Status", "search", "info"),
            ("🧹 Cleanup Augmentation Data", "cleanup", "warning")
        ],
        cleanup_enabled=True, button_width="200px"
    )

    # Output areas dengan service support
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', margin='8px 0', overflow='hidden'
    ))
    
    # Progress tracking dengan dual level untuk overall dan step progress
    progress_components = create_dual_progress_tracker()
    log_components = create_log_accordion('augmentation', '200px')
    help_panel = get_augmentation_info()

    # Layout assembly dengan service optimization
    settings_row1 = widgets.HBox([
        _create_service_section("Opsi Dasar", basic_options['container']),
        _create_service_section("Opsi Lanjutan", advanced_options['container'])
    ], layout=widgets.Layout(width='100%', justify_content='flex-start'))

    settings_row2 = _create_service_section(None, augmentation_types['container'], full_width=True)
    
    config_container = widgets.Box([config_buttons['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='8px 0'))

    settings_container = widgets.VBox([
        settings_row1, settings_row2, config_container
    ], layout=widgets.Layout(width='100%', overflow='hidden'))

    # Main UI assembly dengan service integration
    ui = widgets.VBox([
        header, status_panel, settings_container, create_divider(),
        action_buttons['container'], confirmation_area,
        progress_components['container'], log_components['log_accordion'], help_panel
    ], layout=widgets.Layout(width='100%', overflow='hidden'))

    # Component mapping dengan service integration
    return {
        'ui': ui, 'header': header, 'status_panel': status_panel,
        'settings_container': settings_container, 'confirmation_area': confirmation_area,
        
        # Widget mappings dengan service support
        **basic_options['widgets'], **advanced_options['widgets'], **augmentation_types['widgets'],
        
        # Button mappings dengan service context
        'augment_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        'save_button': config_buttons['save_button'],
        'reset_button': config_buttons['reset_button'],
        
        # Progress dan log dengan service integration
        **progress_components, 'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'status': log_components['log_output'],  # Compatibility
        
        # Service metadata
        'module_name': MODULE_LOGGER_NAME,
        'logger_namespace': AUGMENTATION_LOGGER_NAMESPACE,
        'service_integrated': True,
        'env': env, 'config': config or {}
    }

def _create_service_section(title: str, content_widget, full_width: bool = False):
    """Create settings section dengan service styling"""
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    children = []
    if title:
        children.append(widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0; font-size: 14px;'>{ICONS['settings']} {title}</h5>"))
    children.append(content_widget)
    
    width = '100%' if full_width else '48%'
    margin = '8px 0' if full_width else '0 1% 0 0'
    
    return widgets.VBox(children, layout=widgets.Layout(
        width=width, padding='6px', margin=margin,
        border='1px solid #e0e0e0', border_radius='4px',
        background_color='rgba(248, 249, 250, 0.8)'  # Service background
    ))