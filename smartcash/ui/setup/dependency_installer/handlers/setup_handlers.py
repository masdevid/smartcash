"""
File: smartcash/ui/setup/dependency_installer/handlers/setup_handlers.py
Deskripsi: Enhanced setup handlers dengan integrasi notification observer dan progress tracking yang konsisten
"""

from typing import Dict, Any
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.notification.notification_observer import create_notification_observer
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message, reset_progress_bar, update_status_panel, get_module_logger
from smartcash.ui.setup.dependency_installer.utils.status_utils import setup_status_utils
from smartcash.ui.setup.dependency_installer.handlers.progress_handlers import setup_progress_tracking
from smartcash.ui.setup.dependency_installer.handlers.button_handlers import setup_install_button_handler, setup_reset_button_handler
from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages
from smartcash.ui.setup.dependency_installer.utils.package_installer import install_required_packages

def get_observer_manager():
    """Get observer manager untuk dependency installer dengan singleton pattern"""
    # Gunakan singleton pattern untuk menghindari duplikasi observer
    if not hasattr(get_observer_manager, '_instance') or get_observer_manager._instance is None:
        get_observer_manager._instance = ObserverManager()
    return get_observer_manager._instance

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handlers untuk dependency installer dengan integrasi notification observer dan pendekatan one-liner"""
    # Import logger untuk keperluan logging
    from smartcash.common.logger import get_logger
    
    # Setup progress tracking menggunakan existing component
    ui_components['log_message'] = lambda message, level="info", icon=None: log_message(ui_components, message, level, icon)
    
    # Tambahkan flag suppress_logs ke ui_components jika belum ada
    if 'suppress_logs' not in ui_components:
        ui_components['suppress_logs'] = config.get('suppress_logs', False)
    
    # Setup status utils (update_status_panel, highlight_numeric_params)
    setup_status_utils(ui_components)
    
    # Setup progress tracking (update_progress, reset_progress_bar, show_for_operation)
    setup_progress_tracking(ui_components)
    
    # Sembunyikan progress container saat inisialisasi jika diminta
    if config.get('hide_progress', False) and 'progress_container' in ui_components:
        if hasattr(ui_components['progress_container'], 'layout'):
            ui_components['progress_container'].layout.visibility = 'hidden'
    
    # Setup observer manager dengan silent fail
    try:
        observer_manager = get_observer_manager()
        
        # Tambahkan observer untuk log_message
        if 'log_message' in ui_components:
            observer_manager.add_observer('dependency_installer_log', ui_components['log_message'])
        
        # Tambahkan observer untuk reset_progress_bar
        if 'reset_progress_bar' in ui_components:
            observer_manager.add_observer('dependency_installer_reset_progress', ui_components['reset_progress_bar'])
        
        # Tambahkan observer untuk update_status_panel
        if 'update_status_panel' in ui_components:
            observer_manager.add_observer('dependency_installer_update_status', ui_components['update_status_panel'])
    except Exception as e:
        # Silent fail untuk setup observer
        pass
    
    # Buat fungsi analisis yang dapat digunakan baik langsung maupun sebagai delayed function
    def run_package_analysis():
        try:
            # Tampilkan progress tracker untuk analisis dengan pendekatan DRY
            ui_components['show_for_operation']('analyze')
            ui_components['reset_progress_bar'](0, "Menganalisis packages terinstall...", True)
            
            # Jalankan analisis
            analyze_installed_packages(ui_components)
            
            # Update progress selesai dengan pendekatan DRY
            ui_components['update_progress']('step', 100, "Analisis packages selesai", "#28a745")
            ui_components['log_message'](f"✅ Berhasil mendeteksi packages terinstall", "success")
            ui_components['update_status_panel']("success", "Analisis packages selesai")
        except Exception as e:
            # Handle error dengan pendekatan DRY
            error_message = f"Gagal mendeteksi packages: {str(e)}"
            ui_components['log_message'](f"⚠️ {error_message}", "warning")
            ui_components['error_operation'](error_message)
            ui_components['update_status_panel']("warning", error_message)
    
    # Setup handlers dengan silent fail
    try:
        # Setup install button handler
        setup_install_button_handler(ui_components)
        
        # Setup analyze button handler
        setup_analyze_button_handler(ui_components)
        
        # Setup reset button handler
        setup_reset_button_handler(ui_components)
        
        # Simpan fungsi analisis untuk dijalankan langsung atau nanti
        ui_components['run_delayed_analysis'] = lambda: run_package_analysis()
        
        # Jalankan analisis package jika delay_analysis tidak diset atau False
        if not config.get('delay_analysis', False):
            run_package_analysis()
    except Exception as e:
        # Silent fail untuk setup handlers
        if 'log_message' in ui_components and not ui_components.get('suppress_logs', False):
            ui_components['log_message'](f"❌ Error setup handlers: {str(e)}", "error")
    
    # Tandai sebagai diinisialisasi
    ui_components['dependency_installer_initialized'] = True
    
    return ui_components

def setup_analyze_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol analyze
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'analyze_button' not in ui_components or not hasattr(ui_components['analyze_button'], 'on_click'):
        return
    
    # Definisikan handler untuk tombol analyze
    def analyze_click_handler(b):
        # Reset log output
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output()
        
        # Tampilkan progress container
        if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
            ui_components['progress_container'].layout.visibility = 'visible'
        
        # Jalankan analisis menggunakan fungsi run_package_analysis yang sudah didefinisikan
        if 'run_delayed_analysis' in ui_components and callable(ui_components['run_delayed_analysis']):
            ui_components['run_delayed_analysis']()
    
    ui_components['analyze_button'].on_click(analyze_click_handler)

# Fungsi analyze_installed_packages telah dipindahkan ke analyzer_utils.py
# Fungsi install_required_packages telah dipindahkan ke package_installer.py

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