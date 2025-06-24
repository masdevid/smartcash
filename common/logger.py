"""
File: smartcash/common/logger.py
Deskripsi: Enhanced logger dengan safe level handling dan UI integration
"""

import logging
import sys
from typing import Dict, Any, Optional, Union
from enum import Enum

class LogLevel(Enum):
    """Enum untuk log levels dengan backward compatibility"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # Aliases
    WARN = logging.WARNING
    FATAL = logging.CRITICAL

# Global mapping untuk level normalization
LEVEL_MAPPING = {
    'DEBUG': logging.DEBUG, 'debug': logging.DEBUG,
    'INFO': logging.INFO, 'info': logging.INFO,
    'WARNING': logging.WARNING, 'warning': logging.WARNING,
    'ERROR': logging.ERROR, 'error': logging.ERROR,
    'CRITICAL': logging.CRITICAL, 'critical': logging.CRITICAL,
    'WARN': logging.WARNING, 'warn': logging.WARNING,
    'FATAL': logging.CRITICAL, 'fatal': logging.CRITICAL
}

def normalize_log_level(level: Union[str, int, LogLevel]) -> int:
    """Normalize logging level ke integer yang valid"""
    if isinstance(level, int):
        return level if 0 <= level <= 50 else logging.INFO
    if isinstance(level, LogLevel):
        return level.value
    if isinstance(level, str):
        return LEVEL_MAPPING.get(level, logging.INFO)
    return logging.INFO

class SmartCashLogger:
    """Enhanced logger dengan UI integration dan safe level handling"""
    
    def __init__(self, name: str, level: Union[str, int, LogLevel] = LogLevel.INFO, ui_components: Optional[Dict[str, Any]] = None):
        self.name = name
        self.ui_components = ui_components
        self._setup_logger(level)
    
    def _setup_logger(self, level: Union[str, int, LogLevel]):
        """Setup standard Python logger dengan safe level"""
        self.logger = logging.getLogger(self.name)
        safe_level = normalize_log_level(level)
        self.logger.setLevel(safe_level)
        
        # Add console handler jika belum ada
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_to_ui(self, message: str, level: str):
        """Safe logging ke UI components dengan emoji"""
        if not self.ui_components:
            return
            
        try:
            log_output = self.ui_components.get('log_output')
            if not log_output:
                return
                
            # Emoji mapping untuk visual feedback
            emoji_map = {
                'info': 'ℹ️', 'warning': '⚠️', 'warn': '⚠️',
                'error': '❌', 'critical': '🚨', 'fatal': '🚨',
                'success': '✅', 'debug': '🔍'
            }
            
            emoji = emoji_map.get(level.lower(), 'ℹ️')
            formatted_message = f"{emoji} {message}\n"
            
            # Append ke log output widget
            if hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(formatted_message)
            elif hasattr(log_output, 'value'):
                log_output.value += formatted_message
                
        except Exception:
            pass  # Silent fail untuk UI logging
    
    def _log_both(self, level_name: str, message: str):
        """Log ke both standard logger dan UI"""
        # Normalize level untuk standard logger
        level_int = normalize_log_level(level_name)
        
        # Log ke standard logger
        self.logger.log(level_int, message)
        
        # Log ke UI
        self._log_to_ui(message, level_name)
    
    # Public logging methods
    def debug(self, message: str):
        self._log_both('debug', message)
        
    def info(self, message: str):
        self._log_both('info', message)
        
    def warning(self, message: str):
        self._log_both('warning', message)
        
    def warn(self, message: str):  # Alias
        self.warning(message)
        
    def error(self, message: str):
        self._log_both('error', message)
        
    def critical(self, message: str):
        self._log_both('critical', message)
        
    def fatal(self, message: str):  # Alias
        self.critical(message)
    
    # Utility methods
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk logging integration"""
        self.ui_components = ui_components
    
    def set_level(self, level: Union[str, int, LogLevel]):
        """Set logging level dengan safe normalization"""
        safe_level = normalize_log_level(level)
        self.logger.setLevel(safe_level)

# Factory function untuk backward compatibility
def get_logger(name: str, level: Union[str, int, LogLevel] = LogLevel.INFO, ui_components: Optional[Dict[str, Any]] = None) -> SmartCashLogger:
    """Factory function untuk membuat SmartCashLogger instance"""
    return SmartCashLogger(name, level, ui_components)

# Safe logging utility functions
def safe_log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> bool:
    """Utility function untuk safe UI logging"""
    try:
        log_output = ui_components.get('log_output')
        if not log_output:
            return False
            
        emoji_map = {
            'info': 'ℹ️', 'warning': '⚠️', 'error': '❌',
            'critical': '🚨', 'success': '✅', 'debug': '🔍'
        }
        
        emoji = emoji_map.get(level.lower(), 'ℹ️')
        formatted_message = f"{emoji} {message}\n"
        
        if hasattr(log_output, 'append_stdout'):
            log_output.append_stdout(formatted_message)
            return True
        elif hasattr(log_output, 'value'):
            log_output.value += formatted_message
            return True
            
        return False
        
    except Exception:
        return False

def create_ui_logger(name: str, ui_components: Dict[str, Any], level: Union[str, int, LogLevel] = LogLevel.INFO) -> SmartCashLogger:
    """Create logger dengan UI components integration"""
    logger = SmartCashLogger(name, level)
    logger.set_ui_components(ui_components)
    return logger