"""
File: smartcash/ui/cells/cell_2_5_pretrained_model.py
Deskripsi: Download model pre-trained YOLOv5 dan EfficientNet-B4 untuk SmartCash
"""

from smartcash.model.services.pretrained_setup import setup_pretrained_models

# Eksekusi saat modul diimpor
model_info = setup_pretrained_models(models_dir='/content/models')
