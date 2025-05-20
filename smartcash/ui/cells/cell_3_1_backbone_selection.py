"""
File: smartcash/ui/cells/cell_3_1_backbone_selection.py
Deskripsi: Entry point untuk pemilihan model, backbone dan konfigurasi layer SmartCash
"""

from smartcash.ui.training_config.backbone.backbone_initializer import initialize_backbone_ui

# Inisialisasi dan tampilkan UI backbone selection
ui_components = initialize_backbone_ui()