"""
File: smartcash/ui/utils/color_utils.py
Deskripsi: Utilitas untuk manajemen warna pada komponen UI
"""

from typing import Tuple


def get_color_for_status(status: str) -> Tuple[str, str]:
    """
    Mendapatkan warna latar belakang dan teks berdasarkan status.
    
    Args:
        status: Status komponen ('default', 'preprocessing', 'augmentation')
        
    Returns:
        Tuple berisi (warna_latar_belakang, warna_teks)
    """
    if status == 'preprocessing':
        # Warna biru untuk preprocessing
        return '#e3f2fd', '#0d47a1'
    elif status == 'augmentation':
        # Warna hijau untuk augmentasi
        return '#e8f5e9', '#1b5e20'
    else:
        # Warna abu-abu untuk default
        return '#f5f5f5', '#424242'
