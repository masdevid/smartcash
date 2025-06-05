"""
File: smartcash/ui/pretrained_model/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk modul pretrained model
"""

from typing import Dict, Any, Optional, Callable
import os
import yaml
from pathlib import Path

from smartcash.ui.handlers.config_handlers import BaseConfigHandler

class PretrainedModelConfigHandler(BaseConfigHandler):
    """Config handler untuk modul pretrained model dengan base_config.yaml"""
    
    def __init__(self, module_name: str, parent_module: Optional[str] = None):
        """Inisialisasi dengan base_config.yaml"""
        super().__init__(module_name, None, None, None)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi default dari base_config.yaml"""
        try:
            # Menggunakan base_config.yaml sebagai fallback
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                      'configs', 'base_config.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    base_config = yaml.safe_load(file)
                    # Ekstrak bagian yang relevan untuk pretrained model
                    return {
                        'models_dir': base_config.get('pretrained_models_path', '/content/models'),
                        'drive_models_dir': base_config.get('drive_models_path', '/content/drive/MyDrive/SmartCash/models')
                    }
            
            # Fallback jika file tidak ditemukan
            self.logger.warning(f"⚠️ File konfigurasi tidak ditemukan: {config_path}")
            return {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading base config: {str(e)}")
            return {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models'
            }
    
    def load_config(self) -> Dict[str, Any]:
        """Load config dari file atau gunakan default"""
        return self.get_default_config()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Simpan konfigurasi (tidak diimplementasikan untuk pretrained model)"""
        # Pretrained model tidak menyimpan konfigurasi
        return True
