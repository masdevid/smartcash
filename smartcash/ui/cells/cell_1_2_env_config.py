"""
File: smartcash/ui/cells/cell_1_2_env_config.py
Deskripsi: Entry point untuk environment configuration cell
"""

def setup_env_config():
    """Setup dan tampilkan UI untuk konfigurasi environment."""
    from smartcash.ui.setup.env_config import initialize_env_config_ui
    return initialize_env_config_ui()

# Eksekusi saat modul diimpor
ui_components = setup_env_config()