"""
File: smartcash/ui/setup/dependency_installer/utils/logger_helper.py
Deskripsi: Utilitas untuk logging yang konsisten di modul dependency installer dengan pendekatan DRY
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import (
    DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, 
    KNOWN_NAMESPACES,
    create_namespace_badge,
    format_log_message
)

# Import fungsi-fungsi dari modul lain untuk menghindari redundansi
from smartcash.ui.setup.dependency_installer.utils.status_utils import highlight_numeric_params
from smartcash.ui.setup.dependency_installer.utils.ui_utils import log_to_ui
from smartcash.ui.setup.dependency_installer.utils.progress_helper import update_progress_step, update_overall_progress

# Konstanta untuk namespace logger
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]
logger = get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)

def get_module_logger():
    """
    Mendapatkan logger yang sudah dikonfigurasi dengan namespace yang benar
    """
    return logger

# Flag global untuk mencegah rekursi tak terbatas
_is_logging = False

# Fungsi format_log_message sudah diimpor dari ui_logger_namespace

def format_log_message_html(message: str, level: str = "info") -> str:
    """
    Format pesan log dengan styling HTML yang konsisten
    
    Args:
        message: Pesan yang akan diformat
        level: Level log (info, success, warning, error)
        
    Returns:
        HTML string yang sudah diformat
    """
    namespace_badge = create_namespace_badge("DEPINST")
    
    level_icon = {
        "info": ICONS.get('info', 'ℹ️'),
        "success": ICONS.get('success', '✅'),
        "warning": ICONS.get('warning', '⚠️'),
        "error": ICONS.get('error', '❌'),
        "debug": ICONS.get('debug', '🔍'),
        "critical": ICONS.get('critical', '🔥')
    }.get(level, 'ℹ️')
    
    # Gunakan highlight_numeric_params dari status_utils.py
    highlighted_message = highlight_numeric_params(message)
    
    # Warna sesuai level
    color_map = {
        "info": COLORS.get('info', '#007bff'), 
        "success": COLORS.get('success', '#28a745'),
        "warning": COLORS.get('warning', '#ffc107'), 
        "error": COLORS.get('error', '#dc3545'), 
        "critical": COLORS.get('critical', '#dc3545'), 
        "debug": COLORS.get('debug', '#6c757d')
    }
    
    return f"""
    <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
               background-color:rgba(248,249,250,0.8);
               border-left:3px solid {color_map.get(level, '#D7BDE2')};">
        <span style="margin-right:5px;">{namespace_badge}</span>
        <span>{level_icon} {highlighted_message}</span>
    </div>
    """

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke komponen UI dengan styling yang konsisten
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan dilog
        level: Level log (info, success, warning, error)
        icon: Icon kustom (opsional)
    """
    global _is_logging
    
    # Cegah rekursi tak terbatas atau log yang ditekan
    if _is_logging or ui_components.get('suppress_logs', False): 
        return
    
    try:
        # Set flag untuk mencegah rekursi
        _is_logging = True
        
        # Check initialization
        if not is_initialized(ui_components):
            # Jika tidak diinisialisasi, log ke logger Python saja
            import logging
            logger.log(
                logging.INFO if level == "info" else
                logging.DEBUG if level == "debug" else
                logging.WARNING if level == "warning" else
                logging.ERROR if level == "error" else
                logging.CRITICAL if level == "critical" else
                logging.INFO,
                message
            )
            return
        
        # Gunakan log_to_ui dari ui_utils.py
        log_to_ui(ui_components, message, level, icon)
        
        # Log ke backend logger
        if level == "info":
            logger.info(message)
        elif level == "success":
            logger.info(f"✅ {message}")
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "critical":
            logger.critical(message)
        else:
            logger.info(message)
        
        # Update status panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            try: 
                ui_components['update_status_panel'](level, message)
            except Exception as e: 
                logger.error(f"Failed to update status panel: {str(e)}")
        
        # Update progress jika level error dan ada update_progress
        if level == "error" and 'update_progress' in ui_components and callable(ui_components['update_progress']):
            try: 
                ui_components['update_progress']('overall', 100, message, "#dc3545")
            except Exception as e: 
                logger.error(f"Failed to update progress: {str(e)}")
    except Exception as e:
        # Fallback ke logger Python jika terjadi error
        logger.error(f"Error saat logging: {str(e)}")
    finally:
        # Reset flag untuk mencegah rekursi
        _is_logging = False

def clear_log_output(log_output) -> None:
    """
    Membersihkan output log dari komponen UI
    
    Args:
        log_output: Komponen output log yang akan dibersihkan
    """
    if hasattr(log_output, 'clear_output'):
        log_output.clear_output()
    
    # Log ke logger bahwa log telah dibersihkan
    logger.info(f"{ICONS.get('cleanup', '🧹')} Log output telah dibersihkan")

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Check apakah dependency installer sudah diinisialisasi
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean yang menunjukkan apakah dependency installer sudah diinisialisasi
    """
    return ui_components.get('dependency_installer_initialized', False) or ui_components.get('module_name') == 'dependency_installer'

def reset_progress_bar(ui_components: Dict[str, Any], value: int = 0, message: str = "", show_progress: bool = True) -> None:
    """
    Reset progress bar dan label dengan nilai awal dan kontrol visibilitas
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai awal progress bar
        message: Pesan yang akan ditampilkan
        show_progress: True untuk menampilkan progress, False untuk menyembunyikan
    """
    # Log operasi reset
    logger.debug(f"Resetting progress bar: value={value}, message='{message}', show={show_progress}")
    
    try:
        # Reset progress bar jika tersedia
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = value
            ui_components['progress_bar'].max = 100
            if hasattr(ui_components['progress_bar'], 'layout'): 
                ui_components['progress_bar'].layout.visibility = 'visible' if show_progress else 'hidden'
            if hasattr(ui_components['progress_bar'], 'reset') and value == 0: 
                ui_components['progress_bar'].reset()
        
        # Reset semua jika value 0 dan reset_all tersedia
        if 'reset_all' in ui_components and callable(ui_components['reset_all']) and value == 0:
            ui_components['reset_all']()
            if 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show') and hasattr(ui_components['tracker'], 'hide'):
                if show_progress:
                    ui_components['tracker'].show()
                else:
                    ui_components['tracker'].hide()
        
        # Update progress label jika tersedia
        if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
            ui_components['progress_label'].value = message or "Siap"
            if hasattr(ui_components['progress_label'], 'layout'): 
                ui_components['progress_label'].layout.visibility = 'visible'
        
        # Update status widget jika tersedia
        if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
            ui_components['status_widget'].value = message or "Siap"
            if hasattr(ui_components['status_widget'], 'layout'): 
                ui_components['status_widget'].layout.visibility = 'visible'
        
        # Update progress jika value bukan 0 dan update_progress tersedia
        if value != 0 and 'update_progress' in ui_components and callable(ui_components['update_progress']): 
            # Gunakan fungsi dari progress_helper
            update_overall_progress(ui_components, value, message)
    except Exception as e:
        # Fallback jika terjadi error
        logger.error(f"Error saat reset progress bar: {str(e)}")

def update_status_panel(ui_components: Dict[str, Any], level: str = "info", message: str = "") -> None:
    """
    Update status panel dengan pesan dan level yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        level: Level status (info, success, warning, error)
        message: Pesan yang akan ditampilkan
    """
    # Log operasi update status
    logger.debug(f"Updating status panel: level={level}, message='{message}'")
    
    try:
        # Import dan gunakan fungsi dari status_utils.py
        from smartcash.ui.setup.dependency_installer.utils.status_utils import update_status_panel as update_status_utils
        
        # Panggil fungsi dari status_utils.py
        update_status_utils(ui_components, level, message)
    except Exception as e:
        # Fallback jika terjadi error
        logger.error(f"Error saat update status panel: {str(e)}")
        
        # Fallback langsung ke status panel
        if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
            ui_components['status_panel'].value = f"<div>{ICONS.get(level, '❓')} {message}</div>"
            if hasattr(ui_components['status_panel'], 'layout'): 
                ui_components['status_panel'].layout.visibility = 'visible'
        
        # Update status widget jika tersedia
        if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
            ui_components['status_widget'].value = message
            if hasattr(ui_components['status_widget'], 'layout'): 
                ui_components['status_widget'].layout.visibility = 'visible'