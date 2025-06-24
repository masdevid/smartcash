# File: smartcash/ui/setup/env_config/handlers/ui_logger_handler.py
# Deskripsi: Handler untuk UI logging dengan accordion auto-open dan color coding

import time
from typing import Dict, Any

class UILoggerHandler:
    """🔧 Handler untuk UI logging system"""
    
    def create_ui_logger(self, ui_components: Dict[str, Any]) -> 'UILogger':
        """Create UI logger instance dengan auto-open accordion"""
        return UILogger(ui_components)

class UILogger:
    """🔧 Logger yang mengarahkan output ke UI log accordion"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self._ensure_accordion_open()
        
    def _ensure_accordion_open(self):
        """Pastikan accordion log terbuka otomatis"""
        if 'log_accordion' in self.ui_components:
            accordion = self.ui_components['log_accordion']
            if hasattr(accordion, 'selected_index'):
                accordion.selected_index = 0  # Buka tab pertama
        
    def info(self, message: str) -> None:
        """Log info message ke UI"""
        self._append_to_log(message, 'info', '💡')
        
    def warning(self, message: str) -> None:
        """Log warning message ke UI"""
        self._append_to_log(message, 'warning', '⚠️')
        
    def error(self, message: str, exc_info=None) -> None:
        """Log error message ke UI"""
        self._append_to_log(message, 'error', '❌')
        
    def success(self, message: str) -> None:
        """Log success message ke UI"""
        self._append_to_log(message, 'success', '✅')
    
    def _append_to_log(self, message: str, level: str, emoji: str):
        """Append message ke log output dengan color coding"""
        if 'log_output' not in self.ui_components:
            return
            
        log_output = self.ui_components['log_output']
        timestamp = time.strftime('%H:%M:%S')
        
        # Color mapping untuk berbagai level
        colors = {
            'info': '#2196F3',
            'success': '#4CAF50', 
            'warning': '#FF9800',
            'error': '#F44336'
        }
        
        color = colors.get(level, '#333333')
        formatted_msg = f'<span style="color: {color};">[{timestamp}] {emoji} {message}</span><br>'
        
        # Append ke existing content
        current_content = getattr(log_output, 'value', '')
        log_output.value = current_content + formatted_msg