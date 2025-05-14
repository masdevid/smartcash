"""
File: smartcash/common/constants/environment.py
Deskripsi: Konstanta terkait environment dan konfigurasi lingkungan
"""

import os

# Environment variables
ENV_CONFIG_PATH = os.environ.get("SMARTCASH_CONFIG_PATH", "")
ENV_MODEL_PATH = os.environ.get("SMARTCASH_MODEL_PATH", "")
ENV_DATA_PATH = os.environ.get("SMARTCASH_DATA_PATH", "")