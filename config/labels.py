# File: src/config/labels.py
# Author: Alfrida Sabar
# Deskripsi: Konfigurasi label untuk deteksi nominal uang Rupiah

from typing import List, Dict

class LabelConfig:
    """Konfigurasi label untuk SmartCash Detector"""
    
    # Daftar label dalam format yang diinginkan
    LABELS = ['100k', '10k', '1k', '20k', '2k', '50k', '5k']
    
    def get_num_classes(cls) -> int:
        """Dapatkan jumlah kelas"""
        return len(cls.LABELS)
    
    @classmethod
    def convert_label(cls, old_label: str) -> str:
        """Konversi format label lama ke format baru"""
        return cls.LABEL_MAP.get(old_label, old_label)
    
    @classmethod
    def get_label_idx(cls, label: str) -> int:
        """Dapatkan indeks label"""
        return cls.LABELS.index(label)