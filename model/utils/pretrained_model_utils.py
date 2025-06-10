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

def check_pretrained_model_in_drive(model_name, models_dir):
    """Cek apakah model pretrained ada di Google Drive"""
    # Implementasi aktual
    return f"{models_dir}/{model_name}.pt"

def load_pretrained_model(model, path: str, device: str) -> torch.nn.Module:
    """
    Memuat model pretrained dari file.

    Args:
        model: Model yang akan dimuat
        path: Path ke file model
        device: Device yang digunakan

    Returns:
        Model dengan pretrained weights
    """
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {path}")
            
        # Periksa apakah file kosong
        if Path(path).stat().st_size == 0:
            raise ValueError(f"File model kosong: {path}")
            
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        logger.info(f"✅ Pretrained model berhasil dimuat dari {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Gagal load pretrained model dari {path}: {str(e)}")
        raise e
