"""
File: smartcash/ui/setup/dependency_installer/handlers/setup_handlers.py
Deskripsi: Enhanced setup handlers dengan integrasi notification observer dan progress tracking yang konsisten
"""

from typing import Dict, Any
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.notification.notification_observer import create_notification_observer
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message, reset_progress_bar, update_status_panel, get_module_logger

def get_observer_manager():
    """Get observer manager untuk dependency installer dengan singleton pattern"""
    # Gunakan singleton pattern untuk menghindari duplikasi observer
    if not hasattr(get_observer_manager, '_instance') or get_observer_manager._instance is None:
        get_observer_manager._instance = ObserverManager()
    return get_observer_manager._instance

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handlers untuk dependency installer dengan integrasi notification observer dan pendekatan one-liner"""
    # Import existing handlers dan utils
    from smartcash.ui.setup.dependency_installer.handlers.install_handler import setup_install_handler
    from smartcash.ui.setup.dependency_installer.handlers.analyzer_handler import setup_analyzer_handler
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.common.logger import get_logger
    
    # Setup progress tracking menggunakan existing component
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    
    # Create logger
    logger = get_module_logger()
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
    
    # Tambahkan fungsi utilitas sebagai lambda untuk pendekatan DRY
    ui_components['log_message'] = lambda message, level="info", icon=None: log_message(ui_components, message, level, icon)
    ui_components['reset_progress_bar'] = lambda value=0, message="", show_progress=True: reset_progress_bar(ui_components, value, message, show_progress)
    ui_components['update_status_panel'] = lambda level="info", message="": update_status_panel(ui_components, level, message)
    
    # Set default suppress_logs jika tidak ada
    if 'suppress_logs' not in ui_components:
        ui_components['suppress_logs'] = config.get('suppress_logs', False) if config else False
    
    try:
        # Setup observer manager untuk notifikasi
        observer_manager = get_observer_manager()
        ui_components['observer_manager'] = observer_manager
        
        # Register observer untuk event dependency installer
        _setup_dependency_installer_observers(ui_components, observer_manager)
    except Exception as e:
        # Tangkap error tapi jangan gagalkan setup (silent fail)
        ui_components['log_message'](f"⚠️ Observer setup error: {str(e)}", "warning")
    
    try:
        # Setup handlers
        setup_install_handler(ui_components)
        setup_analyzer_handler(ui_components)
    except Exception as e:
        # Tangkap error tapi jangan gagalkan setup (silent fail)
        ui_components['log_message'](f"⚠️ Handler setup error: {str(e)}", "warning")
    
    # Deteksi packages yang sudah terinstall dengan progress tracking
    try:
        # Tampilkan progress tracker untuk analisis dengan pendekatan one-liner
        ui_components['show_for_operation']('analyze')
        ui_components['reset_progress_bar'](0, "Menganalisis packages terinstall...", True)
        
        # Jalankan analisis
        analyze_installed_packages(ui_components)
        
        # Update progress selesai dengan pendekatan one-liner
        ui_components['update_progress']('step', 100, "Analisis packages selesai", "#28a745")
        ui_components['log_message'](f"✅ Berhasil mendeteksi packages terinstall", "success")
        ui_components['update_status_panel']("success", "Analisis packages selesai")
    except Exception as e:
        # Handle error dengan pendekatan one-liner
        error_message = f"Gagal mendeteksi packages: {str(e)}"
        ui_components['log_message'](f"⚠️ {error_message}", "warning")
        ui_components['error_operation'](error_message)
        ui_components['update_status_panel']("warning", error_message)
    
    # Tandai sebagai diinisialisasi
    ui_components['dependency_installer_initialized'] = True
    
    return ui_components

def _setup_dependency_installer_observers(ui_components: Dict[str, Any], observer_manager):
    """Setup observer untuk event dependency installer dengan pendekatan one-liner dan penanganan error yang robust"""
    # Import event dispatcher
    from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
    
    # Buat event dispatcher jika belum ada
    if 'event_dispatcher' not in ui_components:
        ui_components['event_dispatcher'] = EventDispatcher()
    
    # Bersihkan observer lama jika ada
    if 'registered_observers' in ui_components and isinstance(ui_components['registered_observers'], list):
        for observer in ui_components['registered_observers']:
            try:
                ui_components['observer_manager'].unregister_observer(observer)
            except Exception as e:
                # Silent fail
                pass
    
    # Definisikan callback untuk event dependency installer
    def dependency_event_callback(event_type, sender, **kwargs):
        """Callback untuk event dependency installer"""
        try:
            # Extract message, level, dan icon dari kwargs
            message = kwargs.get('message', f"Event {event_type}")
            level = kwargs.get('level', "info")
            icon = kwargs.get('icon', "ℹ️")
            
            # Log message jika tersedia
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](message, level, icon)
            
            # Update progress jika tersedia
            if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                # Extract progress dari kwargs
                progress = kwargs.get('progress', None)
                if progress is not None:
                    ui_components['update_progress'](progress, message)
            
            # Update status panel jika tersedia
            if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
                ui_components['update_status_panel'](level, message)
        except Exception as e:
            # Tangkap error tapi jangan tampilkan di console
            logger = ui_components.get('logger')
            if logger and hasattr(logger, 'error') and 'log_message' not in ui_components:
                logger.error(f"🔥 Error pada dependency_event_callback: {str(e)}")
            elif 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"🔥 Error pada callback: {str(e)}", "error")
    
    # Simpan callback di ui_components untuk referensi
    ui_components['dependency_event_callback'] = dependency_event_callback
    registered_observers = []
    try:
        # Import observer components
        from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
        from smartcash.components.notification.notification_observer import NotificationObserver
        from smartcash.components.observer.base_observer import BaseObserver
        
        # Definisikan event_types secara eksplisit sebagai string, bukan dict
        # Gunakan nama event yang sama dengan yang digunakan di observer_helper.py
        event_types = [
            'DEPENDENCY_INSTALL_START',
            'DEPENDENCY_INSTALL_PROGRESS',
            'DEPENDENCY_INSTALL_ERROR',
            'DEPENDENCY_INSTALL_COMPLETE',
            'DEPENDENCY_ANALYZE_START',
            'DEPENDENCY_ANALYZE_PROGRESS',
            'DEPENDENCY_ANALYZE_ERROR',
            'DEPENDENCY_ANALYZE_COMPLETE'
        ]
        
        # Buat callback untuk log message dengan lambda yang menerima 3 parameter
        log_callback = lambda event_type, sender, **kwargs: log_message(
            ui_components, 
            kwargs.get('message', f"Event {event_type}"), 
            kwargs.get('level', "info"), 
            kwargs.get('icon', "ℹ️")
        )
        
        # Buat observers untuk setiap event type
        registered_observers = []
        for event_type in event_types:
            try:
                # Buat observer dengan event_type dan callback
                observer = NotificationObserver(
                    event_type=event_type,
                    callback=log_callback,
                    name=f"DependencyInstaller_{event_type}"
                )
                
                # Verifikasi observer adalah instance dari BaseObserver
                if isinstance(observer, BaseObserver):
                    # Register observer langsung ke EventDispatcher
                    EventDispatcher.register(event_type, observer)
                    registered_observers.append(observer)
                    
                    # Log success
                    log_message(ui_components, f"Observer untuk {event_type} berhasil didaftarkan", "debug", "✅")
                else:
                    log_message(ui_components, f"Observer bukan instance dari BaseObserver: {type(observer)}", "error", "❌")
            except Exception as e:
                log_message(ui_components, f"Gagal setup observer untuk {event_type}: {str(e)}", "error", "❌")
        
        # Log success
        log_message(ui_components, "Observer setup berhasil", "success", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Observer setup gagal: {str(e)}", "error", "❌")
        
    # Simpan daftar observer yang berhasil diregistrasi
    ui_components['registered_observers'] = registered_observers