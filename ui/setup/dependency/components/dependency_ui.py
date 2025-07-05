# file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/dependency/components/dependency_ui.py
# Deskripsi: Berisi fungsi untuk membuat komponen UI Dependency Manager.

from typing import Optional, Dict, Any

from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

from .dependency_tabs import create_dependency_tabs

def create_dependency_ui_components(config: Optional[Dict[str, Any]] = None, logger=None, **kwargs) -> Dict[str, Any]:
    """Create dependency UI components dengan standard layout

    Args:
        config (Optional[Dict[str, Any]], optional): Konfigurasi untuk UI components. Defaults to None.
        logger (Any, optional): Logger instance untuk mencatat aktivitas UI. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary berisi semua komponen UI yang telah dibuat.
    """
    
    current_config = config or {}
    
    # Header container
    header_container = create_header_container(
        title="📦 Dependency Manager",
        subtitle="Kelola packages untuk SmartCash dengan interface yang mudah",
        status_message="Siap mengelola dependencies",
        status_type="info"
    )
    
    # Create form container with tabs
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="0",
        gap="0"
    )
    
    # Add tabs to form container
    dependency_tabs = create_dependency_tabs(current_config, logger)
    form_container['add_item'](dependency_tabs, height="auto")
    
    # Action container
    action_container = create_action_container(
        buttons=[
            {'button_id': 'install_button', 'text': '📥 Install Selected', 'style': 'primary', 'order': 1},
            {'button_id': 'check_updates_button', 'text': '🔄 Check Updates', 'style': 'info', 'order': 2},
            {'button_id': 'uninstall_button', 'text': '🗑️ Uninstall Selected', 'style': 'danger', 'order': 3}
        ],
        title="🚀 Package Operations",
        alignment="left"
    )
    
    # Footer container with InfoAccordion for tips
    from smartcash.ui.components.footer_container import PanelConfig, PanelType
    
    tips_content = """
    <div style="padding: 10px;">
        <h5>💡 Tips Penggunaan</h5>
        <ul>
            <li>Gunakan tab pertama untuk packages berdasarkan kategori</li>
            <li>Tab kedua untuk packages custom yang tidak tersedia di kategori</li>
            <li>Default packages ditandai dengan ⭐ dan direkomendasikan untuk diinstall</li>
            <li>Status package akan terupdate secara real-time</li>
        </ul>
    </div>
    """
    
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="💡 Tips & Info",
                content=tips_content,
                style="info",
                flex="1",
                min_width="300px",
                open_by_default=True
            )
        ],
        style={"border_top": "1px solid #e0e0e0", "padding": "10px 0"},
        flex_flow="row wrap",
        justify_content="space-between",
        align_items="flex-start"
    )
    
    # Create operation container for progress and dialogs
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="Dependency"
    )
    
    # Main container
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=form_container['container'],
        action_container=action_container['container'],
        operation_container=operation_container['container'],
        footer_container=footer_container.container
    )
    
    return {
        'main_container': main_container.container,
        'ui': main_container.container,  # Alias for compatibility
        'header_container': header_container,
        'form_container': form_container,
        'dependency_tabs': dependency_tabs,
        'action_container': action_container,
        'footer_container': footer_container,
        'operation_container': operation_container,
        'status_panel': header_container.status_panel,
        'install_button': action_container['buttons'].get('install_button'),
        'check_updates_button': action_container['buttons'].get('check_updates_button'),
        'uninstall_button': action_container['buttons'].get('uninstall_button'),
        'progress_tracker': operation_container['progress_tracker'],
        'confirmation_dialog': operation_container['dialog'],
        'log_accordion': operation_container['log_accordion'],
    }
