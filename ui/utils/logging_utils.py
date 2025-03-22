"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas untuk logging di notebook dengan tampilan berbasis UI alerts dan penangkapan semua output logging
"""

import logging
import sys
import io
from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ALERT_STYLES, ICONS

class UILogHandler(logging.Handler):
    """Custom logging handler for displaying logs in a UI output widget."""
    
    # Class-level constants for better performance and maintainability
    _LEVEL_TO_STYLE: Dict[int, str] = {
        logging.DEBUG: 'info',
        logging.INFO: 'info',
        logging.WARNING: 'warning',
        logging.ERROR: 'error',
        logging.CRITICAL: 'error'
    }
    
    _LEVEL_TO_ICON: Dict[int, str] = {
        logging.DEBUG: ICONS.get('info', 'ℹ️'),
        logging.INFO: ICONS.get('info', 'ℹ️'),
        logging.WARNING: ICONS.get('warning', '⚠️'),
        logging.ERROR: ICONS.get('error', '❌'),
        logging.CRITICAL: ICONS.get('error', '❌')
    }
    
    _IGNORED_MODULES = frozenset(['matplotlib', 'PIL', 'IPython', 'ipykernel', 'traitlets', 'ColabKernelApp'])

    def __init__(self, output_widget: widgets.Output):
        """Initialize UILogHandler with an output widget.

        Args:
            output_widget: Output widget for displaying logs
        """
        super().__init__(level=logging.INFO)  # Changed from DEBUG to INFO
        self.output_widget = output_widget
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        """Display log in output widget with styled alert.

        Args:
            record: Log record to process
        """
        try:
            # Early return for ignored modules
            if record.name in self._IGNORED_MODULES:
                return
                
            # Ignore DEBUG messages completely
            if record.levelno < logging.INFO:
                return

            style_key = self._LEVEL_TO_STYLE.get(record.levelno, 'info')
            style = ALERT_STYLES.get(style_key, ALERT_STYLES['info'])
            icon = self._LEVEL_TO_ICON.get(record.levelno)

            # Single f-string for HTML generation
            log_html = (
                f'<div style="padding: 5px; margin: 2px 0; '
                f'background-color: {style["bg_color"]}; '
                f'color: {style["text_color"]}; '
                f'border-left: 4px solid {style["border_color"]}; '
                f'border-radius: 4px;">'
                f'<span style="margin-right: 8px;">{icon}</span>'
                f'<span>{self.format(record)}</span>'
                f'</div>'
            )

            with self.output_widget:
                display(HTML(log_html))

        except Exception as e:
            print(f"Error in UILogHandler: {str(e)}")
            super().emit(record)

class UILogger(logging.Logger):
    """Logger khusus dengan metode yang disesuaikan untuk alerts styling."""
    
    def success(self, msg, *args, **kwargs):
        """
        Log dengan level INFO tapi dengan style success.
        
        Args:
            msg: Pesan log
            *args, **kwargs: Argumen tambahan
        """
        if self.isEnabledFor(logging.INFO):
            # Tambahkan emoji success jika belum ada
            if not any(emoji in msg for emoji in ['✅', '✓', '🎉']):
                msg = f"✅ {msg}"
            self._log(logging.INFO, msg, args, **kwargs)

def redirect_all_logging(output_widget: widgets.Output) -> List[logging.Handler]:
    """
    Redirect semua output logging Python ke widget output.
    
    Args:
        output_widget: Widget output untuk menampilkan log
        
    Returns:
        List handler yang telah ditambahkan
    """
    # Simpan referensi ke root logger dan handler yang telah ditambahkan
    root_logger = logging.getLogger()
    added_handlers = []
    
    # Hapus semua handler yang sudah ada
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Buat dan tambahkan UI handler ke root logger
    ui_handler = UILogHandler(output_widget)
    root_logger.addHandler(ui_handler)
    added_handlers.append(ui_handler)
    
    # Set level root logger ke INFO, bukan DEBUG
    root_logger.setLevel(logging.INFO)
    
    # Mute logging dari beberapa modul tertentu
    for name in ['ColabKernelApp', 'ipykernel', 'matplotlib']:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # Hentikan propagasi untuk semua logger yang sudah ada
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.propagate = False
    
    return added_handlers

def setup_ipython_logging(
    ui_components: Dict[str, Any],
    logger_name: str = 'ui_logger',
    log_level: int = logging.INFO,  # Changed from logging.DEBUG to logging.INFO
    clear_existing_handlers: bool = True,
    redirect_root: bool = True
) -> Optional[logging.Logger]:
    """
    Setup logger untuk notebook dengan output ke widget.
    
    Args:
        ui_components: Dictionary berisi widget UI
        logger_name: Nama logger
        log_level: Level logging (default INFO, bukan DEBUG)
        clear_existing_handlers: Hapus handler yang sudah ada
        redirect_root: Redirect juga root logger ke widget output
        
    Returns:
        Logger instance atau None jika gagal
    """
    # Register custom logger class
    logging.setLoggerClass(UILogger)
    
    # Dapatkan output widget
    output_widget = None
    for key in ['status', 'log_output', 'output']:
        if key in ui_components and isinstance(ui_components[key], widgets.Output):
            output_widget = ui_components[key]
            break
    
    if not output_widget:
        return None
    
    # Setup logger module-level
    logger = logging.getLogger(logger_name)
    
    # Pastikan level minimal INFO, bukan DEBUG
    log_level = max(log_level, logging.INFO)
    logger.setLevel(log_level)
    
    # Hapus handler existing jika perlu
    if clear_existing_handlers:
        logger.handlers = []
    
    # Tambahkan UI handler
    ui_handler = UILogHandler(output_widget)
    ui_handler.setLevel(log_level)
    logger.addHandler(ui_handler)
    
    # Nonaktifkan propagasi ke root logger agar tidak muncul di console
    logger.propagate = False
    
    # Hapus semua handler console dari logging module untuk memastikan tidak ada log yang muncul di console
    if redirect_root:
        # Hapus handler lama dari root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                root_logger.removeHandler(handler)
        
        # Redirect root logger ke UI widget dengan level INFO
        root_handlers = redirect_all_logging(output_widget)
        ui_components['root_log_handlers'] = root_handlers
    
    return logger

def reset_logging():
    """
    Reset konfigurasi logging ke default (untuk testing atau reset).
    """
    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Tambahkan kembali handler ke console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    console_handler.setLevel(logging.INFO)  # Gunakan level INFO, bukan WARNING
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)  # Default level INFO
    
    # Reset propagation untuk semua logger
    for name in logging.root.manager.loggerDict:
        # Untuk logger yang tidak perlu debug, set ke level WARNING
        if name in ['ColabKernelApp', 'ipykernel', 'matplotlib', 'torch']:
            logging.getLogger(name).setLevel(logging.WARNING)
        else:
            logging.getLogger(name).setLevel(logging.INFO)
            logging.getLogger(name).propagate = True

def create_dummy_logger() -> logging.Logger:
    """
    Buat dummy logger yang tidak melakukan output.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('dummy_logger')
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.INFO)  # Changed from default to INFO
    
    return logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """
    Log messages directly to UI without a logger.

    Args:
        ui_components: Dictionary containing UI widgets
        message: Message to display
        level: Log level ('info', 'success', 'warning', 'error')
    """
    # Ignore if level is DEBUG
    if level == 'debug':
        return
        
    # Find output widget efficiently
    output_widget = next(
        (ui_components[key] for key in ('status', 'log_output', 'output')
         if key in ui_components and isinstance(ui_components[key], widgets.Output)),
        None
    )
    
    if not output_widget:
        return

    # Get style and icon with fallback to 'info'
    style = ALERT_STYLES.get(level, ALERT_STYLES['info'])
    icon = style.get('icon', ICONS.get(level, ICONS['info']))

    # Display using context manager
    with output_widget:
        from smartcash.ui.utils.alert_utils import create_info_log
        display(create_info_log(message, level, icon))

def alert_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """
    Alert messages directly to UI without a logger.

    Args:
        ui_components: Dictionary containing UI widgets
        message: Message to display
        level: Log level ('info', 'success', 'warning', 'error')
    """
    # Ignore if level is DEBUG
    if level == 'debug':
        return
        
    # Find output widget efficiently
    output_widget = next(
        (ui_components[key] for key in ('status', 'log_output', 'output')
         if key in ui_components and isinstance(ui_components[key], widgets.Output)),
        None
    )
    
    if not output_widget:
        return

    # Get style and icon with fallback to 'info'
    style = ALERT_STYLES.get(level, ALERT_STYLES['info'])
    icon = style.get('icon', ICONS.get(level, ICONS['info']))

    # Display using context manager
    with output_widget:
        from smartcash.ui.utils.alert_utils import create_info_alert
        display(create_info_alert(message, level, icon))

# Decorator untuk try-except pada handler
def try_except_decorator(handler_func):
    def wrapper(b):
        try:
            return handler_func(b)
        except Exception as e:
            from smartcash.ui.utils.alert_utils import create_status_indicator
            with ui_components['status']:
                display(create_status_indicator('error', f"❌ Error: {str(e)}"))
            if 'logger' in ui_components and ui_components['logger']:
                ui_components['logger'].error(f"❌ Error dalam handler: {str(e)}")
    return wrapper