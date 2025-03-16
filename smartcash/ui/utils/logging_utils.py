"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging untuk integrasi output SmartCashLogger dengan widgets IPython
"""

from IPython.display import display, HTML
import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable

class IPythonLogCallback:
    """Callback untuk SmartCashLogger yang menampilkan output di IPython widget."""
    
    def __init__(self, output_widget: widgets.Output):
        self.output_widget = output_widget
    
    def __call__(self, level: str, message: str, use_emoji: bool = True, **kwargs):
        """
        Callback untuk SmartCashLogger yang menampilkan log di widget IPython.
        
        Args:
            level: Level log (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
            message: Pesan log
            use_emoji: Tampilkan emoji atau tidak
            **kwargs: Argumen tambahan dari logger
        """
        # Default styles untuk berbagai level
        level_styles = {
            'DEBUG': {'color': '#6c757d', 'emoji': '🔍'},
            'INFO': {'color': '#0c5460', 'emoji': 'ℹ️'},
            'SUCCESS': {'color': '#155724', 'emoji': '✅'},
            'WARNING': {'color': '#856404', 'emoji': '⚠️'},
            'ERROR': {'color': '#721c24', 'emoji': '❌'},
            'CRITICAL': {'color': '#721c24', 'emoji': '🚨'}
        }
        
        # Get emoji dari kwargs atau dari mapping default
        style = level_styles.get(level, {'color': 'black', 'emoji': ''})
        emoji = kwargs.get('emoji', style['emoji']) if use_emoji else ''
        color = kwargs.get('color', style['color'])
        
        # Format HTML
        formatted = f'<span style="color:{color}">{emoji} <b>{level}</b>:</span> {message}'
        
        # Tampilkan output di widget
        with self.output_widget:
            display(HTML(formatted))


def setup_ipython_logging(
    ui_components: Dict[str, Any], 
    output_key: str = 'status', 
    logger_name: Optional[str] = None
) -> Any:
    """
    Setup logging SmartCash ke widget IPython.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        output_key: Key untuk output widget dalam ui_components
        logger_name: Nama logger yang akan digunakan
        
    Returns:
        Logger yang telah dikonfigurasi, atau None jika gagal
    """
    try:
        # Coba import SmartCashLogger dari lokasi yang benar berdasarkan dokumentasi
        from smartcash.common.logger import get_logger
        
        # Get output widget
        output_widget = ui_components.get(output_key)
        if not output_widget:
            # Jika tidak ada output widget, kembalikan logger tanpa callback
            return get_logger(logger_name or 'ui_component')
        
        # Buat logger dengan nama yang ditentukan
        logger = get_logger(logger_name or 'ui_component')
        
        # Buat callback untuk output ke widget
        ipython_callback = IPythonLogCallback(output_widget)
        
        # Tambahkan callback ke logger
        logger.add_callback(ipython_callback)
        
        return logger
    
    except ImportError as e:
        # Fallback ke logging biasa jika SmartCashLogger tidak ditemukan
        import logging
        
        class SimpleLogger:
            """Logger sederhana sebagai fallback."""
            
            def __init__(self, name, output_widget=None):
                self.name = name
                self.output_widget = output_widget
                self.python_logger = logging.getLogger(name)
            
            def _log(self, level, msg, emoji=''):
                """Log pesan dengan level tertentu."""
                self.python_logger.log(getattr(logging, level), msg)
                
                # Jika ada output widget, tampilkan pesan dengan format
                if self.output_widget:
                    level_colors = {
                        'DEBUG': '#6c757d',
                        'INFO': '#0c5460', 
                        'SUCCESS': '#155724',
                        'WARNING': '#856404',
                        'ERROR': '#721c24',
                        'CRITICAL': '#721c24'
                    }
                    color = level_colors.get(level, 'black')
                    with self.output_widget:
                        display(HTML(f'<span style="color:{color}">{emoji} <b>{level}</b>:</span> {msg}'))
            
            def debug(self, msg): self._log('DEBUG', msg, '🔍')
            def info(self, msg): self._log('INFO', msg, 'ℹ️')
            def success(self, msg): self._log('INFO', msg, '✅')  # SUCCESS tidak ada di logging standard
            def warning(self, msg): self._log('WARNING', msg, '⚠️')
            def error(self, msg): self._log('ERROR', msg, '❌')
            def critical(self, msg): self._log('CRITICAL', msg, '🚨')
            
            def add_callback(self, callback): pass  # Dummy method
            def remove_callback(self, callback): pass  # Dummy method
        
        # Get output widget
        output_widget = ui_components.get(output_key)
        
        # Return simple logger
        return SimpleLogger(logger_name or 'ui_component', output_widget)


def register_logging_observer(
    ui_components: Dict[str, Any],
    observer_manager: Any,
    event_types: list,
    output_key: str = 'status',
    observer_group: Optional[str] = None
) -> bool:
    """
    Register observer untuk log events yang menampilkan di IPython widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        observer_manager: Instance ObserverManager
        event_types: List tipe event untuk di-observe
        output_key: Key untuk output widget
        observer_group: Nama group untuk observer
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Get output widget
    output_widget = ui_components.get(output_key)
    if not output_widget:
        return False
    
    # Get observer group
    group = observer_group or ui_components.get('observer_group', 'logging_observers')
    
    # Buat callback untuk update UI
    def log_to_widget_callback(event_type, sender, message=None, level=None, **kwargs):
        if not message:
            return
            
        # Default styles untuk berbagai level
        level_styles = {
            'debug': {'color': '#6c757d', 'emoji': '🔍'},
            'info': {'color': '#0c5460', 'emoji': 'ℹ️'},
            'success': {'color': '#155724', 'emoji': '✅'},
            'warning': {'color': '#856404', 'emoji': '⚠️'},
            'error': {'color': '#721c24', 'emoji': '❌'},
            'critical': {'color': '#721c24', 'emoji': '🚨'}
        }
        
        # Get style berdasarkan level atau default ke info
        log_level = (level or 'info').lower()
        style = level_styles.get(log_level, level_styles['info'])
        emoji = kwargs.get('emoji', style['emoji'])
        color = kwargs.get('color', style['color'])
        
        # Format HTML
        formatted = f'<span style="color:{color}">{emoji} <b>{log_level.upper()}</b>:</span> {message}'
        
        # Tampilkan output di widget
        with output_widget:
            display(HTML(formatted))
    
    # Register observer untuk tiap event type
    for event_type in event_types:
        try:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=log_to_widget_callback,
                name=f"LogUI_{event_type}_Observer",
                group=group
            )
        except Exception:
            return False
    
    return True