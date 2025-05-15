"""
File: smartcash/ui/cells/cell_3_1_backbone_selection.py
Deskripsi: Entry point untuk pemilihan model, backbone dan konfigurasi layer SmartCash
"""

def setup_backbone_selection():
    """Setup dan tampilkan UI untuk pemilihan backbone model."""
    from smartcash.ui.training_config.backbone.backbone_initializer import initialize_backbone_ui
    return initialize_backbone_ui()

# Eksekusi saat modul diimpor
ui_components = setup_backbone_selection()