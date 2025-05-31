"""
File: smartcash/ui/setup/dependency_installer/handlers/setup_handlers.py
Deskripsi: Enhanced setup handlers dengan integrasi notification observer dan progress tracking yang konsisten
"""

from typing import Dict, Any
from smartcash.components.observer.manager_observer import get_observer_manager
from smartcash.components.notification.notification_observer import create_notification_observer

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handlers untuk dependency installer dengan integrasi notification observer"""
    # Import existing handlers
    from smartcash.ui.setup.dependency_installer.handlers.install_handler import setup_install_handler
    from smartcash.ui.setup.dependency_installer.handlers.analyzer_handler import setup_analyzer_handler
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    from smartcash.common.logger import get_logger
    
    # Setup progress tracking menggunakan existing component
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    
    # Create logger
    logger = get_logger('smartcash.setup.dependency_installer')
    ui_components['logger'] = logger
    
    # Create progress tracker dengan enhanced settings
    progress_container = create_progress_tracking_container()
    ui_components.update({
        'progress_tracker': progress_container['tracker'],
        'progress_container': progress_container['container'],
        'update_progress': progress_container['update_progress'],
        'complete_operation': progress_container['complete_operation'],
        'error_operation': progress_container['error_operation'],
        'show_for_operation': progress_container['show_for_operation']
    })
    
    # Setup observer manager untuk notifikasi
    observer_manager = get_observer_manager()
    ui_components['observer_manager'] = observer_manager
    
    # Register observer untuk event dependency installer
    _setup_dependency_installer_observers(ui_components, observer_manager)
    
    # Setup handlers
    setup_install_handler(ui_components)
    setup_analyzer_handler(ui_components)
    
    # Deteksi packages yang sudah terinstall dengan progress tracking
    try:
        # Tampilkan progress tracker untuk analisis
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('analyze')
        
        # Update progress
        if 'update_progress' in ui_components:
            ui_components['update_progress']('step', 0, "Menganalisis packages terinstall...", "#007bff")
        
        # Jalankan analisis
        analyze_installed_packages(ui_components)
        
        # Update progress selesai
        if 'update_progress' in ui_components:
            ui_components['update_progress']('step', 100, "Analisis packages selesai", "#28a745")
            
        log_message(ui_components, f"✅ Berhasil mendeteksi packages terinstall", "success")
    except Exception as e:
        log_message(ui_components, f"⚠️ Gagal mendeteksi packages: {str(e)}", "warning")
        
        # Update progress error
        if 'error_operation' in ui_components:
            ui_components['error_operation'](f"Gagal mendeteksi packages: {str(e)}")
    
    # Tandai sebagai diinisialisasi
    ui_components['dependency_installer_initialized'] = True
    
    return ui_components

def _setup_dependency_installer_observers(ui_components: Dict[str, Any], observer_manager):
    """Setup observer untuk event dependency installer"""
    # Import logger helper
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Event types untuk dependency installer
    event_types = [
        'DEPENDENCY_INSTALL_START',
        'DEPENDENCY_INSTALL_PROGRESS',
        'DEPENDENCY_INSTALL_COMPLETE',
        'DEPENDENCY_INSTALL_ERROR',
        'DEPENDENCY_ANALYZE_START',
        'DEPENDENCY_ANALYZE_COMPLETE',
        'DEPENDENCY_ANALYZE_ERROR'
    ]
    
    # Callback untuk event dependency installer
    def dependency_event_callback(event_type, sender, **kwargs):
        # Log event ke UI
        message = kwargs.get('message', f"Event {event_type}")
        level = kwargs.get('level', 'info')
        
        # Map event type ke level
        level_map = {
            'DEPENDENCY_INSTALL_START': 'info',
            'DEPENDENCY_INSTALL_PROGRESS': 'info',
            'DEPENDENCY_INSTALL_COMPLETE': 'success',
            'DEPENDENCY_INSTALL_ERROR': 'error',
            'DEPENDENCY_ANALYZE_START': 'info',
            'DEPENDENCY_ANALYZE_COMPLETE': 'success',
            'DEPENDENCY_ANALYZE_ERROR': 'error'
        }
        
        # Gunakan level dari map jika tersedia
        level = level_map.get(event_type, level)
        
        # Log ke UI
        log_message(ui_components, message, level)
        
        # Update progress jika tersedia
        if 'progress' in kwargs and 'update_progress' in ui_components:
            progress = kwargs.get('progress', 0)
            progress_message = kwargs.get('progress_message', message)
            ui_components['update_progress']('overall', progress, progress_message)
    
    # Buat dan register observer untuk setiap event type
    for event_type in event_types:
        observer = create_notification_observer(
            event_type=event_type,
            callback=dependency_event_callback,
            name=f"DependencyInstaller_{event_type}",
            priority=10
        )
        
        try:
            observer_manager.register(observer, event_type)
        except Exception as e:
            # Log error tapi jangan gagalkan setup
            logger = ui_components.get('logger')
            if logger and hasattr(logger, 'debug'):
                logger.debug(f"⚠️ Gagal register observer untuk {event_type}: {str(e)}")