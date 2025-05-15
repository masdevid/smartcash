"""
File: smartcash/ui/cells/cell_2_6_pretrained_model.py
Deskripsi: Download model pre-trained YOLOv5 dan EfficientNet-B4 untuk SmartCash dengan sinkronisasi Google Drive
"""

def setup_pretrained_model():
    """Setup dan tampilkan UI untuk pretrained model."""
    # Import modul pretrained model initializer
    from smartcash.ui.model.pretrained_initializer import initialize_pretrained_model_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_pretrained_model_ui()

# Eksekusi saat modul diimpor
ui_components = setup_pretrained_model()
