"""
/Users/masdevid/Projects/smartcash/smartcash/model/utils/pretrained_model_utils.py

Utilitas untuk pengelolaan dan loading pretrained model dari Google Drive.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
from smartcash.common.logger import SmartCashLogger

logger = SmartCashLogger(__name__)

# Konstanta untuk path model di drive
DRIVE_MODELS_DIR = Path('/drive/MyDrive/SmartCash/models')

# Mapping backbone ke nama file pretrained model
PRETRAINED_MODEL_FILENAMES = {
    'efficientnet_b4': 'efficientnet_b4_huggingface.bin',
    'cspdarknet_s': 'yolov5s.pt'
}

def check_pretrained_model_in_drive(backbone: str) -> Optional[str]:
    """
    Memeriksa ketersediaan pretrained model di Google Drive berdasarkan backbone.
    
    Args:
        backbone: Tipe backbone yang digunakan ('efficientnet_b4' atau 'cspdarknet_s')
        
    Returns:
        Path model jika ditemukan, None jika tidak ditemukan
    """
    if backbone not in PRETRAINED_MODEL_FILENAMES:
        logger.warning(f"❌ Backbone {backbone} tidak didukung untuk pretrained model dari drive")
        return None
        
    # Periksa apakah direktori model di drive ada
    if not DRIVE_MODELS_DIR.exists():
        logger.info(f"📂 Direktori model di drive tidak ditemukan: {DRIVE_MODELS_DIR}")
        return None
        
    # Dapatkan nama file model berdasarkan backbone
    model_filename = PRETRAINED_MODEL_FILENAMES[backbone]
    model_path = DRIVE_MODELS_DIR / model_filename
    
    # Periksa apakah file model ada
    if not model_path.exists():
        logger.info(f"📄 File model {model_filename} tidak ditemukan di drive")
        return None
        
    # Validasi sederhana: periksa ukuran file
    try:
        file_size = model_path.stat().st_size
        if file_size > 0:
            logger.success(f"✅ Pretrained model untuk {backbone} ditemukan di drive (ukuran: {file_size/1024/1024:.2f} MB)")
            return str(model_path)
        else:
            logger.warning(f"⚠️ Pretrained model untuk {backbone} ditemukan di drive tetapi ukurannya 0 bytes")
            return None
    except Exception as e:
        logger.warning(f"⚠️ Gagal memvalidasi pretrained model untuk {backbone}: {str(e)}")
        # Tetap gunakan model jika file ada meskipun validasi gagal
        return str(model_path)

def load_pretrained_model(model, path: str, device: str) -> torch.nn.Module:
    """
    Load pretrained model dari path yang diberikan.
    
    Args:
        model: Model PyTorch yang akan diload dengan pretrained weights
        path: Path ke file pretrained model
        device: Device untuk loading model ('cpu' atau 'cuda')
        
    Returns:
        Model yang telah diload dengan pretrained weights
    """
    try:
        logger.info(f"🔄 Loading pretrained model dari drive: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        logger.success(f"✅ Berhasil load pretrained model dari drive")
        return model
    except Exception as e:
        logger.error(f"❌ Gagal load pretrained model dari drive: {str(e)}")
        raise e
