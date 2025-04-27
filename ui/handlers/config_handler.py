"""
File: smartcash/ui/handlers/config_handler.py
Deskripsi: Proxy untuk fungsi-fungsi config handler dari training_config
"""

# Re-export fungsi dari training_config.config_handler untuk kompatibilitas
from smartcash.ui.training_config.config_handler import save_config, reset_config

# Tambahkan docstring yang jelas untuk membantu pemahaman
__doc__ = """
Module proxy untuk menjaga kompatibilitas dengan kode yang mengharapkan config_handler di ui/handlers.
Semua fungsi di sini di-forward ke implementasi di smartcash.ui.training_config.config_handler.
"""
