# File: smartcash/utils/simple_logger.py
# Author: Alfrida Sabar
# Deskripsi: Logger sederhana
class SimpleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def info(self, msg):
        self.logger.info(msg)
        print(f"ℹ️ {msg}")
    
    def error(self, msg):
        self.logger.error(msg)
        print(f"❌ {msg}")
    
    def success(self, msg):
        self.logger.info(f"SUCCESS: {msg}")
        print(f"✅ {msg}")
    
    def warning(self, msg):
        self.logger.warning(msg)
        print(f"⚠️ {msg}")
        
    def metric(self, msg):
        self.logger.info(f"METRIC: {msg}")
        print(f"📈 {msg}")