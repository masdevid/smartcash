"""
File: smartcash/ui/dataset/preprocessing/utils/ui_observers.py
Deskripsi: Utilitas untuk mengelola observer UI pada proses preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import ObserverManager, EventTopics
from smartcash.ui.dataset.preprocessing.utils.notification_manager import PreprocessingUIEvents, PREPROCESSING_LOGGER_NAMESPACE

def register_ui_observers(ui_components: Dict[str, Any]) -> ObserverManager:
    """
    Daftarkan observer untuk UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ObserverManager: Manager untuk observer yang terdaftar
    """
    # Get observer manager dari UI components atau buat baru
    observer_manager = ui_components.get('observer_manager', ObserverManager())
    
    # Observer untuk log output
    def log_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan log hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != PREPROCESSING_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.preprocessing'):
            return  # Skip log dari namespace lain
            
        if event_type in [PreprocessingUIEvents.LOG_INFO, PreprocessingUIEvents.LOG_WARNING, 
                         PreprocessingUIEvents.LOG_ERROR, PreprocessingUIEvents.LOG_SUCCESS]:
            if isinstance(ui_components, dict) and 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_stdout'):
                message = kwargs.get('message', '')
                level = kwargs.get('level', 'info')
                
                # Format pesan dengan emoji
                emoji_map = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "success": "✅",
                    "debug": "🔍"
                }
                emoji = kwargs.get('icon', emoji_map.get(level.lower(), "ℹ️"))
                formatted_message = f"{emoji} {message}"
                
                # Tampilkan pesan di log output
                if level.lower() == 'error':
                    ui_components['log_output'].append_stderr(formatted_message)
                else:
                    ui_components['log_output'].append_stdout(formatted_message)
                
                # Pastikan log accordion terbuka
                if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
                    ui_components['log_accordion'].selected_index = 0  # Buka accordion pertama
    
    # Observer untuk progress bar
    def progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan progress hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != PREPROCESSING_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.preprocessing'):
            return  # Skip progress dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [PreprocessingUIEvents.PROGRESS_START, PreprocessingUIEvents.PROGRESS_UPDATE,
                         PreprocessingUIEvents.PROGRESS_COMPLETE, PreprocessingUIEvents.PROGRESS_ERROR]:
            # Pastikan progress container terlihat
            if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.display = 'block'
                ui_components['progress_container'].layout.visibility = 'visible'
                
            # Update progress bar
            if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
                # Update progress bar
                progress = kwargs.get('progress', 0)
                total = kwargs.get('total', 100)
                message = kwargs.get('message', '')
                
                # Pastikan progress adalah integer
                try:
                    progress = int(float(progress))
                except (ValueError, TypeError):
                    progress = 0
                
                # Update nilai progress bar
                ui_components['progress_bar'].value = progress
                ui_components['progress_bar'].description = f"Progress: {progress}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
                
                # Update pesan progress jika ada
                if message and 'overall_label' in ui_components:
                    ui_components['overall_label'].value = message
                    ui_components['overall_label'].layout.visibility = 'visible'
                
                # Jika complete, set progress ke 100%
                if event_type == PreprocessingUIEvents.PROGRESS_COMPLETE:
                    ui_components['progress_bar'].value = 100
                    ui_components['progress_bar'].description = "Progress: 100%"
                    
                # Jika error, tampilkan pesan error
                if event_type == PreprocessingUIEvents.PROGRESS_ERROR:
                    if 'log_output' in ui_components:
                        ui_components['log_output'].append_stderr(f"❌ {message}")
    
    # Observer untuk step progress
    def step_progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan step progress hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != PREPROCESSING_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.preprocessing'):
            return  # Skip step progress dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [PreprocessingUIEvents.STEP_PROGRESS_START, PreprocessingUIEvents.STEP_PROGRESS_UPDATE,
                         PreprocessingUIEvents.STEP_PROGRESS_COMPLETE, PreprocessingUIEvents.STEP_PROGRESS_ERROR]:
            # Update step progress bar
            if 'current_progress' in ui_components and hasattr(ui_components['current_progress'], 'value'):
                step_progress = kwargs.get('step_progress', 0)
                step_total = kwargs.get('step_total', 100)
                step_message = kwargs.get('step_message', '')
                current_step = kwargs.get('current_step', 1)
                total_steps = kwargs.get('total_steps', 5)
                
                # Pastikan progress adalah integer
                try:
                    step_progress = int(float(step_progress))
                except (ValueError, TypeError):
                    step_progress = 0
                
                # Update nilai step progress bar
                ui_components['current_progress'].value = step_progress
                ui_components['current_progress'].description = f"Step {current_step}/{total_steps}"
                ui_components['current_progress'].layout.visibility = 'visible'
                
                # Update pesan step progress jika ada
                if step_message and 'step_label' in ui_components:
                    ui_components['step_label'].value = step_message
                    ui_components['step_label'].layout.visibility = 'visible'
                
                # Jika complete, set progress ke 100%
                if event_type == PreprocessingUIEvents.STEP_PROGRESS_COMPLETE:
                    ui_components['current_progress'].value = 100
                    ui_components['current_progress'].description = f"Step {total_steps}/{total_steps}"
    
    # Observer untuk status panel
    def status_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan status hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != PREPROCESSING_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.preprocessing'):
            return  # Skip status dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [PreprocessingUIEvents.STATUS_STARTED, PreprocessingUIEvents.STATUS_COMPLETED,
                         PreprocessingUIEvents.STATUS_FAILED, PreprocessingUIEvents.STATUS_PAUSED,
                         PreprocessingUIEvents.STATUS_RESUMED]:
            # Update status panel jika ada
            if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
                status = kwargs.get('status', '')
                message = kwargs.get('message', '')
                
                # Map status ke emoji dan warna
                status_map = {
                    "started": ("🔄", "blue", "Running"),
                    "completed": ("✅", "green", "Completed"),
                    "failed": ("❌", "red", "Failed"),
                    "paused": ("⏸️", "orange", "Paused"),
                    "resumed": ("▶️", "blue", "Resumed")
                }
                
                # Get emoji dan warna berdasarkan status
                emoji, color, default_message = status_map.get(status.lower(), ("ℹ️", "gray", "Unknown"))
                
                # Update status panel
                ui_components['status_panel'].value = f"<span style='color: {color};'>{emoji} {message or default_message}</span>"
    
    # Observer untuk config
    def config_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan config hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != PREPROCESSING_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.preprocessing'):
            return  # Skip config dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [PreprocessingUIEvents.CONFIG_UPDATED, PreprocessingUIEvents.CONFIG_SAVED,
                         PreprocessingUIEvents.CONFIG_LOADED, PreprocessingUIEvents.CONFIG_RESET]:
            # Handle config event
            action = kwargs.get('action', '')
            config = kwargs.get('config', {})
            
            # Tampilkan notifikasi berdasarkan action
            if action == 'saved':
                if 'log_output' in ui_components:
                    ui_components['log_output'].append_stdout("💾 Konfigurasi berhasil disimpan")
            elif action == 'loaded':
                if 'log_output' in ui_components:
                    ui_components['log_output'].append_stdout("📂 Konfigurasi berhasil dimuat")
            elif action == 'reset':
                if 'log_output' in ui_components:
                    ui_components['log_output'].append_stdout("🔄 Konfigurasi berhasil direset")
    
    # Daftarkan observer untuk semua event
    log_events = [
        PreprocessingUIEvents.LOG_INFO, 
        PreprocessingUIEvents.LOG_WARNING, 
        PreprocessingUIEvents.LOG_ERROR, 
        PreprocessingUIEvents.LOG_SUCCESS,
        PreprocessingUIEvents.LOG_DEBUG
    ]
    
    progress_events = [
        PreprocessingUIEvents.PROGRESS_START, 
        PreprocessingUIEvents.PROGRESS_UPDATE, 
        PreprocessingUIEvents.PROGRESS_COMPLETE, 
        PreprocessingUIEvents.PROGRESS_ERROR
    ]
    
    step_progress_events = [
        PreprocessingUIEvents.STEP_PROGRESS_START, 
        PreprocessingUIEvents.STEP_PROGRESS_UPDATE, 
        PreprocessingUIEvents.STEP_PROGRESS_COMPLETE, 
        PreprocessingUIEvents.STEP_PROGRESS_ERROR
    ]
    
    status_events = [
        PreprocessingUIEvents.STATUS_STARTED,
        PreprocessingUIEvents.STATUS_COMPLETED,
        PreprocessingUIEvents.STATUS_FAILED,
        PreprocessingUIEvents.STATUS_PAUSED,
        PreprocessingUIEvents.STATUS_RESUMED
    ]
    
    config_events = [
        PreprocessingUIEvents.CONFIG_UPDATED,
        PreprocessingUIEvents.CONFIG_SAVED,
        PreprocessingUIEvents.CONFIG_LOADED,
        PreprocessingUIEvents.CONFIG_RESET
    ]
    
    # Buat dan daftarkan observer
    observer_manager.create_simple_observer(log_events, log_observer, name="preprocessing_log_observer")
    observer_manager.create_simple_observer(progress_events, progress_observer, name="preprocessing_progress_observer")
    observer_manager.create_simple_observer(step_progress_events, step_progress_observer, name="preprocessing_step_progress_observer")
    observer_manager.create_simple_observer(status_events, status_observer, name="preprocessing_status_observer")
    observer_manager.create_simple_observer(config_events, config_observer, name="preprocessing_config_observer")
    
    # Simpan observer_manager ke ui_components
    ui_components['observer_manager'] = observer_manager
    
    return observer_manager

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("preprocessing", {
            'split': split,
            'display_info': display_info
        })

def notify_process_complete(ui_components: Dict[str, Any], result: Dict[str, Any], display_info: str) -> None:
    """
    Notifikasi observer bahwa proses telah selesai dengan sukses.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil dari proses
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['success']} Preprocessing {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("preprocessing", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi observer bahwa proses mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.error(f"{ICONS['error']} Error pada preprocessing: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("preprocessing", error_message)

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """
    Notifikasi observer bahwa proses telah dihentikan oleh pengguna.
    
    Args:
        ui_components: Dictionary komponen UI
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.warning(f"{ICONS['stop']} Proses preprocessing dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("preprocessing", {
            'display_info': display_info
        })

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        disable: True untuk menonaktifkan, False untuk mengaktifkan
    """
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', 'preprocess_button', 'save_button'
    ]
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components:
            widget = ui_components[component]
            if hasattr(widget, 'disabled'):
                widget.disabled = disable
            elif hasattr(widget, 'layout'):
                # For widgets without disabled attribute, use opacity
                if not hasattr(widget.layout, 'opacity'):
                    # Create new layout with opacity
                    new_layout = type(widget.layout)()
                    for key, value in widget.layout.trait_values().items():
                        setattr(new_layout, key, value)
                    widget.layout = new_layout
                widget.layout.opacity = '0.5' if disable else '1'
