"""
File: smartcash/utils/simple_logger.py
Author: Alfrida Sabar
Deskripsi: Logger sederhana yang digunakan sebagai fallback jika SmartCashLogger tidak tersedia.
"""

import logging
import sys
from datetime import datetime

class SimpleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def info(self, msg):
        """Log informasi."""
        self.logger.info(f"ℹ️ {msg}")
        
    def warning(self, msg):
        """Log peringatan."""
        self.logger.warning(f"⚠️ {msg}")
        
    def error(self, msg):
        """Log error."""
        self.logger.error(f"❌ {msg}")
        
    def success(self, msg):
        """Log keberhasilan."""
        self.logger.info(f"✅ {msg}")
        
    def start(self, msg):
        """Log mulai proses."""
        self.logger.info(f"🔄 {msg}")