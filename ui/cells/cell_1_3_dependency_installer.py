"""
File: smartcash/ui/cells/cell_1_3_dependency_installer.py
Deskripsi: Entry point untuk cell instalasi dependencies dengan sistem deteksi otomatis
"""

def setup_dependency_installer():
    """Setup dan tampilkan UI untuk instalasi dependencies."""
    from smartcash.ui.setup.dependency_installer import initialize_dependency_installer
    return initialize_dependency_installer()

# Eksekusi saat modul diimpor
ui_components = setup_dependency_installer()