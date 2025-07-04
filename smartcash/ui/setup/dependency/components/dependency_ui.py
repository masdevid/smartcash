# file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/dependency/components/dependency_ui.py
# Deskripsi: Berisi fungsi untuk membuat komponen UI Dependency Manager.

from typing import Optional, Dict, Any

from smartcash.ui.core.shared.logger import get_enhanced_logger

from .header_container import create_header_container
from .dependency_tabs import create_dependency_tabs
from .action_container import create_action_container
from .footer_container import create_footer_container
from .main_container import create_main_container

def create_dependency_ui_components(config: Optional[Dict[str, Any]] = None, logger=None, **kwargs) -> Dict[str, Any]:
    """Create dependency UI components dengan standard layout

    Args:
        config (Optional[Dict[str, Any]], optional): Konfigurasi untuk UI components. Defaults to None.
        logger (Any, optional): Logger instance untuk mencatat aktivitas UI. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary berisi semua komponen UI yang telah dibuat.
    """
    
    current_config = config or {}
    logger = logger or get_enhanced_logger(__name__)
    
    # Header container
    header_container = create_header_container(
        title="📦 Dependency Manager",
        subtitle="Kelola packages untuk SmartCash dengan interface yang mudah",
        status_message="Siap mengelola dependencies",
        status_type="info"
    )
    
    # Form container (tabs)
    dependency_tabs = create_dependency_tabs(current_config, logger)
    
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
    
    # Footer container
    footer_container = create_footer_container(
        show_logs=True,
        show_info=True,
        log_module_name="Dependency",
        info_title="💡 Tips & Info",
        info_content="""
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
    )
    
    # Main container
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=dependency_tabs,
        action_container=action_container['container'],
        footer_container=footer_container.container
    )
    
    return {
        'main_container': main_container.container,
        'ui': main_container.container,  # Alias for compatibility
        'header_container': header_container,
        'dependency_tabs': dependency_tabs,
        'action_container': action_container,
        'footer_container': footer_container,
        'status_panel': header_container.status_panel,
        'install_button': action_container['buttons'].get('install_button'),
        'check_updates_button': action_container['buttons'].get('check_updates_button'),
        'uninstall_button': action_container['buttons'].get('uninstall_button'),
        'progress_tracker': None,  # Will be added by action container if needed
        'confirmation_dialog': action_container.get('dialog_area'),
        'log_accordion': footer_container.log_accordion,
        'logger': logger
    }
