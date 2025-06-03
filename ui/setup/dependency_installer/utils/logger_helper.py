"""
File: smartcash/ui/setup/dependency_installer/utils/logger_helper.py
Deskripsi: Utilitas untuk logging yang konsisten di modul dependency installer dengan pendekatan DRY
"""

import sys
import logging
from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML
import ipywidgets as widgets
from datetime import datetime
import re

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
from smartcash.ui.setup.dependency_installer.utils.ui_utils import update_package_status

# Setup logger untuk modul ini
logger = get_logger(__name__)

# Namespace untuk UI logger
ui_logger_namespace = "dependency_installer"

# Flag untuk mencegah recursive logging
_is_logging = False

# Konstanta untuk namespace logger
MODULE_LOGGER_NAME = "dependency_installer"
logger = get_logger(MODULE_LOGGER_NAME)

def get_module_logger():
    """
    Mendapatkan logger yang sudah dikonfigurasi dengan namespace yang benar
    """
    return logger

# Flag global untuk mencegah rekursi tak terbatas
_is_logging = False

def format_log_message_html(message: str, level: str = "info", icon: Optional[str] = None) -> str:
    """
    Format pesan log dengan HTML dan styling yang konsisten
    
    Args:
        message: Pesan yang akan diformat
        level: Level log (info, success, warning, error)
        icon: Icon kustom (opsional)
    
    Returns:
        String HTML yang sudah diformat
    """
    # Dapatkan icon berdasarkan level jika tidak disediakan
    if icon is None:
        icon = ICONS.get(level, "ℹ️")
    
    # Gunakan konfigurasi dari constants.py
    from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
    
    # Dapatkan konfigurasi status berdasarkan level
    config = get_status_config(level)
    
    # Gunakan warna dari konfigurasi
    color = config['border']
    
    # Highlight parameter numerik jika ada
    message = highlight_numeric_params(message)
    
    # Buat namespace badge jika namespace tersedia
    namespace_badge = create_namespace_badge(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    # Format pesan dengan HTML
    return f"""
    <div style="margin:2px 0; padding:3px 0;">
        <span style="color:{color}; font-weight:bold;">{icon}</span>
        {namespace_badge}
        <span style="margin-left:5px;">{message}</span>
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
        
        # Dapatkan icon berdasarkan level jika tidak disediakan
        if icon is None:
            icon = ICONS.get(level, "ℹ️")
        
        # Format pesan dengan HTML
        formatted_message = format_log_message_html(message, level, icon)
        
        # Log ke UI output jika tersedia
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            with ui_components['log_output']:
                display(HTML(formatted_message))
        
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
                # Dapatkan konfigurasi untuk level error
                from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
                error_config = get_status_config('error')
                ui_components['update_progress']('overall', 100, message, error_config['border'])
            except Exception as e: 
                logger.error(f"Failed to update progress: {str(e)}")
    except Exception as e:
        # Fallback ke logger Python jika terjadi error
        logger.error(f"Error saat logging: {str(e)}")
    finally:
        # Reset flag
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
    return ui_components is not None and len(ui_components) > 0 and ('log_output' in ui_components or 'log' in ui_components)

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
        
        # Reset package status jika value 0
        if value == 0 and 'categories' in ui_components:
            # Reset status semua package menjadi "Checking..."
            for category in ui_components.get('categories', []):
                for pkg in category.get('packages', []):
                    # Gunakan update_package_status dari ui_utils.py
                    update_package_status(ui_components, pkg['key'], "info", "Checking...")
        
        # Update progress jika value bukan 0 dan update_progress tersedia
        if value != 0 and 'update_progress' in ui_components and callable(ui_components['update_progress']): 
            # Update progress langsung
            ui_components['update_progress']('overall', value, message)
    except Exception as e:
        # Fallback jika terjadi error
        logger.error(f"Error saat reset progress bar: {str(e)}")

def update_status_panel(ui_components: Dict[str, Any], level: str = "info", message: str = "") -> None:
    """Update status panel dengan pesan dan level yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        level: Level status (info, success, warning, error, danger, debug, critical)
        message: Pesan yang akan ditampilkan
    """
    logger.debug(f"Updating status panel: level={level}, message='{message}'")
    
    try:
        if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
            # Gunakan konfigurasi dari constants.py
            config = get_status_config(level)
            
            # Pastikan pesan sudah memiliki emoji, jika belum tambahkan
            if not any(emoji in message for emoji in ["✅", "❌", "⚠️", "ℹ️", "🔍", "📦"]):
                message = f"{config['emoji']} {message}"
            
            # Update status panel HTML dengan styling yang konsisten
            ui_components['status_panel'].value = f"""
            <div style="padding:8px 12px; background-color:{config['bg']}; 
                       color:{config['color']}; border-radius:5px; margin:10px 0;
                       border-left:4px solid {config['border']};">
                <p style="margin:3px 0">{message}</p>
            </div>
            """
            
            # Pastikan status panel terlihat
            if hasattr(ui_components['status_panel'], 'layout'): 
                ui_components['status_panel'].layout.visibility = 'visible'
            
            # Update status widget jika tersedia
            if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
                ui_components['status_widget'].value = message
                if hasattr(ui_components['status_widget'], 'layout'):
                    ui_components['status_widget'].layout.visibility = 'visible'
    except Exception as e:
        logger.error(f"Error saat update status panel: {str(e)}")