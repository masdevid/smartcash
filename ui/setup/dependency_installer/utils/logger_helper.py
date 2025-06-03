"""
File: smartcash/ui/setup/dependency_installer/utils/logger_helper.py
Deskripsi: Enhanced logger helper dengan emoji konsisten, warna untuk parameter numerik, dan fungsi-fungsi utilitas untuk progress bar dan status panel
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger
import re
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.alert_utils import update_status_panel as update_status

from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]
logger = get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)

def get_module_logger():
    """Mendapatkan logger yang sudah dikonfigurasi dengan namespace yang benar"""
    return logger

# Flag global untuk mencegah rekursi tak terbatas
_is_logging = False

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """Log pesan ke UI dan logger dengan namespace dependency installer dan highlight parameter numerik"""
    global _is_logging
    
    # Cegah rekursi tak terbatas atau log yang ditekan
    if _is_logging or ui_components.get('suppress_logs', False): return
    
    try:
        # Set flag untuk mencegah rekursi
        _is_logging = True
        
        # Check initialization
        if not is_initialized(ui_components):
            # Jika tidak diinisialisasi, log ke logger Python saja
            logger = get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
            logger.debug(f"[UI tidak diinisialisasi] {message}")
            _is_logging = False
            return
        
        # Cegah duplikasi pesan error yang sama
        if level == "error" and 'last_error_message' in ui_components and ui_components.get('last_error_message') == message:
            _is_logging = False
            return
        
        # Simpan pesan error terakhir jika level error
        if level == "error": ui_components['last_error_message'] = message
        
        # Highlight parameter numerik dengan warna sesuai level
        highlighted_message = highlight_numeric_params(message, level)
        
        # Get logger
        logger = ui_components.get('logger') or get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
        
        # Emoji konsisten dengan modul lain
        emoji_map = {"debug": ICONS.get('debug', '🔍'), 
                    "info": ICONS.get('info', 'ℹ️'), 
                    "success": ICONS.get('success', '✅'), 
                    "warning": ICONS.get('warning', '⚠️'), 
                    "error": ICONS.get('error', '❌'), 
                    "critical": ICONS.get('critical', '🔥')}
        icon = icon or emoji_map.get(level, "ℹ️")
        
        # Flag untuk menandai apakah pesan sudah di-log ke UI
        logged_to_ui = False
        
        # Gunakan ui_log langsung untuk menghindari rekursi
        if ('log_output' in ui_components or 'output' in ui_components or 'status' in ui_components):
            try:
                ui_log(ui_components, message, level, icon)
                logged_to_ui = True
            except Exception as e: logger.error(f"Failed to log to UI: {str(e)}")
        
        # Log ke logger Python dengan level yang sesuai
        if hasattr(logger, level.lower()):
            log_func = getattr(logger, level.lower())
            log_func(f"{icon} {message}")
        else: logger.info(f"{icon} {message}")  # Fallback ke info jika level tidak valid
        
        # Update status panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            try: ui_components['update_status_panel'](level, message)
            except Exception as e: logger.error(f"Failed to update status panel: {str(e)}")
        # Update progress jika level error dan ada update_progress
        if level == "error" and 'update_progress' in ui_components and callable(ui_components['update_progress']):
            try: ui_components['update_progress']('overall', 0, message, "#dc3545")  # Gunakan warna merah untuk error
            except Exception as e: logger.error(f"Error saat update progress: {str(e)}")
        
        # Log ke Python logger dengan prefix hanya jika UI log tidak tersedia atau level tertentu
        if not logged_to_ui or level in ['debug', 'error', 'critical']:
            # Hanya log ke file logger untuk level debug, error, dan critical
            prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
            log_methods = {
                "debug": logger.debug,
                "info": logger.info,
                "success": logger.info,
                "warning": logger.warning,
                "error": logger.error,
                "critical": logger.critical
            }
            log_method = log_methods.get(level, logger.info)
            log_method(prefixed_message)
    except Exception as e:
        # Fallback ke logger Python jika semua gagal
        try:
            logger = get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
            logger.error(f"Error saat logging: {str(e)}. Pesan asli: {message}")
        except Exception:
            # Jika semua gagal, gunakan print sebagai fallback terakhir
            print(f"CRITICAL ERROR: Gagal logging pesan: {message}. Error: {str(e)}")
    finally:
        # Pastikan flag selalu direset
        _is_logging = False

def highlight_numeric_params(message: str, level: str = "info") -> str:
    """Highlight parameter numerik dengan warna sesuai level"""
    # Color map berdasarkan level dengan one-liner
    color_map = {
        "success": COLORS.get('success', '#28a745'), "info": COLORS.get('info', '#007bff'), 
        "warning": COLORS.get('warning', '#ffc107'), "error": COLORS.get('error', '#dc3545'), 
        "critical": COLORS.get('critical', '#dc3545'), "debug": COLORS.get('debug', '#6c757d')
    }
    # Dapatkan warna dan gunakan regex untuk highlight parameter numerik
    return re.sub(r'(\d+(?:\.\d+)?(?:%|s|ms)?)', f'<span style="color:{color_map.get(level, "#007bff")};font-weight:bold">\\1</span>', message)

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """Check apakah dependency installer sudah diinisialisasi dengan pendekatan one-liner"""
    return ui_components.get('dependency_installer_initialized', False) or ui_components.get('module_name') == 'dependency_installer'

def reset_progress_bar(ui_components: Dict[str, Any], value: int = 0, message: str = "", show_progress: bool = True) -> None:
    """Reset progress bar dan label dengan nilai awal dan kontrol visibilitas"""
    # Reset progress bar jika tersedia
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].max = 100
        if hasattr(ui_components['progress_bar'], 'layout'): ui_components['progress_bar'].layout.visibility = 'visible' if show_progress else 'hidden'
        if hasattr(ui_components['progress_bar'], 'reset') and value == 0: ui_components['progress_bar'].reset()
    
    # Reset semua jika value 0 dan reset_all tersedia
    if 'reset_all' in ui_components and callable(ui_components['reset_all']) and value == 0:
        ui_components['reset_all']()
        if 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show') and hasattr(ui_components['tracker'], 'hide'):
            ui_components['tracker'].show() if show_progress else ui_components['tracker'].hide()
    
    # Update progress label jika tersedia
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
        ui_components['progress_label'].value = message or "Siap"
        if hasattr(ui_components['progress_label'], 'layout'): ui_components['progress_label'].layout.visibility = 'visible'
    
    # Update status widget jika tersedia
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
        ui_components['status_widget'].value = message or "Siap"
        if hasattr(ui_components['status_widget'], 'layout'): ui_components['status_widget'].layout.visibility = 'visible'
    
    # Update progress jika value bukan 0 dan update_progress tersedia
    if value != 0 and 'update_progress' in ui_components and callable(ui_components['update_progress']): ui_components['update_progress']('overall', value, message)

def update_status_panel(ui_components: Dict[str, Any], level: str = "info", message: str = "") -> None:
    """Update status panel dengan pesan dan level yang konsisten"""
    # Update status panel jika tersedia
    if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
        ui_components['status_panel'].value = message
        if hasattr(ui_components['status_panel'], 'layout'): ui_components['status_panel'].layout.visibility = 'visible'
    
    # Update status widget jika tersedia
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
        ui_components['status_widget'].value = message
        if hasattr(ui_components['status_widget'], 'layout'): ui_components['status_widget'].layout.visibility = 'visible'
    
    # Gunakan alert_utils jika tersedia
    if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
        try: update_status(ui_components['status_panel'], message, level)
        except Exception as e:
            # Fallback jika terjadi error
            logger = get_module_logger()
            logger.error(f"Error saat update status: {str(e)}")
            # Set nilai langsung sebagai fallback
            ui_components['status_panel'].value = f"<div>{ICONS.get(level, '❓')} {message}</div>"