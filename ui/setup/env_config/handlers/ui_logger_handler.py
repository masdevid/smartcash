# File: smartcash/ui/setup/env_config/handlers/ui_logger_handler.py
# Deskripsi: Handler untuk UI logging dengan integrasi log_accordion yang tepat

import time
from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color, get_namespace_id

class UILoggerHandler:
    """🔧 Handler untuk UI logging system dengan integrasi log_accordion"""
    
    def create_ui_logger(self, ui_components: Dict[str, Any]) -> 'UILogger':
        """Create UI logger instance dengan log_accordion integration"""
        return UILogger(ui_components)

class UILogger:
    """🔧 Logger yang menggunakan log_accordion untuk output UI"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.namespace = get_namespace_id(ui_components) or "ENV"
        self._ensure_accordion_available()
        
    def _ensure_accordion_available(self):
        """Pastikan log_accordion tersedia dan terbuka"""
        if 'log_accordion' in self.ui_components:
            accordion = self.ui_components['log_accordion']
            if hasattr(accordion, 'selected_index'):
                accordion.selected_index = 0  # Buka accordion
        
    def info(self, message: str) -> None:
        """Log info message menggunakan log_accordion"""
        self._log_to_accordion(message, 'info', '💡')
        
    def warning(self, message: str) -> None:
        """Log warning message menggunakan log_accordion"""
        self._log_to_accordion(message, 'warning', '⚠️')
        
    def error(self, message: str, exc_info=None) -> None:
        """Log error message menggunakan log_accordion"""
        self._log_to_accordion(message, 'error', '❌')
        
    def success(self, message: str) -> None:
        """Log success message menggunakan log_accordion"""
        self._log_to_accordion(message, 'success', '✅')
    
    def _log_to_accordion(self, message: str, level: str, emoji: str):
        """Log message menggunakan log_accordion.append_log method"""
        # Gunakan log_output dari log_accordion jika tersedia
        log_output = self.ui_components.get('log_output')
        
        if log_output and hasattr(log_output, 'append_log'):
            # Gunakan method append_log dari log_accordion
            log_output.append_log(
                message=f"{emoji} {message}",
                level=level,
                namespace=self.namespace,
                module='env_config'
            )
        else:
            # Fallback ke method lama jika log_accordion tidak tersedia
            self._fallback_log(message, level, emoji)
    
    def _fallback_log(self, message: str, level: str, emoji: str):
        """Fallback logging jika log_accordion tidak tersedia"""
        log_output = self.ui_components.get('log_output')
        if not log_output:
            return
            
        timestamp = time.strftime('%H:%M:%S')
        color = get_namespace_color(self.namespace)
        
        formatted_msg = f'''
        <div style="margin: 2px 0; padding: 4px; word-wrap: break-word;">
            <span style="color: #666; font-size: 12px;">{timestamp}</span> 
            <span style="color: {color};">[{self.namespace}]</span>
            <span style="color: {color};">{level.upper()}</span>
            {emoji} {message}
        </div>
        '''
        
        # Append ke log_output jika ada
        if hasattr(log_output, 'value'):
            current_content = getattr(log_output, 'value', '')
            log_output.value = current_content + formatted_msg