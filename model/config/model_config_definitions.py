"""
File: smartcash/model/config/model_config_definitions.py
Deskripsi: Definisi konfigurasi model SmartCash
"""

from typing import Dict, Any
from smartcash.model.config.model_constants import DEFAULT_MODEL_CONFIG_FULL, OPTIMIZED_MODELS, get_model_config

def get_default_config() -> Dict[str, Any]:
    """Dapatkan default konfigurasi model sebagai dictionary baru."""
    return DEFAULT_MODEL_CONFIG_FULL.copy()

# Catatan: fungsi get_model_config diimpor dari model_constants.py