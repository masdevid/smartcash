"""
File: smartcash/ui/dataset/preprocessing/tests/test_utils.py
Deskripsi: Utilitas untuk pengujian preprocessing dataset
"""

import logging
import gc
import sys
import os
import warnings
import io

# Variabel global untuk menyimpan stderr asli
_original_stderr = None
_dummy_stderr = None

def setup_test_environment():
    """
    Menyiapkan lingkungan pengujian dengan mengabaikan ResourceWarning
    dan mengganti stderr dengan dummy untuk menghindari ValueError.
    
    Fungsi ini harus dipanggil di setUp dari test case atau di awal test suite.
    """
    global _original_stderr, _dummy_stderr
    
    # Filter ResourceWarning
    warnings.filterwarnings('ignore', category=ResourceWarning)
    
    # Simpan stderr asli jika belum disimpan
    if _original_stderr is None:
        _original_stderr = sys.stderr
    
    # Buat dummy stderr untuk menghindari ValueError: I/O operation on closed file
    _dummy_stderr = io.StringIO()
    sys.stderr = _dummy_stderr

def restore_environment():
    """
    Mengembalikan lingkungan pengujian ke keadaan semula.
    
    Fungsi ini harus dipanggil di tearDown dari test case atau di akhir test suite.
    """
    global _original_stderr
    
    # Kembalikan stderr asli jika ada
    if _original_stderr is not None:
        sys.stderr = _original_stderr

def close_all_loggers():
    """
    Menutup semua file handler pada logger untuk menghindari ResourceWarning.
    Fungsi ini harus dipanggil di tearDown dari test case atau di akhir test suite.
    """
    # Simpan stderr asli untuk menghindari masalah dengan ValueError: I/O operation on closed file
    setup_test_environment()
    
    try:
        # Dapatkan semua logger yang terdaftar
        root_logger = logging.getLogger()
        
        # Tutup dan hapus semua handler dari root logger
        if hasattr(root_logger, 'handlers'):
            for handler in list(root_logger.handlers):
                try:
                    # Tutup handler jika memiliki metode close
                    if hasattr(handler, 'close'):
                        handler.close()
                    root_logger.removeHandler(handler)
                except Exception:
                    pass
        
        # Dapatkan semua logger lainnya
        for name in list(logging.root.manager.loggerDict.keys()):
            logger = logging.getLogger(name)
            if hasattr(logger, 'handlers'):
                for handler in list(logger.handlers):
                    try:
                        # Tutup handler jika memiliki metode close
                        if hasattr(handler, 'close'):
                            handler.close()
                        logger.removeHandler(handler)
                    except Exception:
                        pass
        
        # Force garbage collection untuk memastikan semua referensi dilepas
        gc.collect()
    except Exception:
        pass
    finally:
        # Kembalikan stderr asli
        restore_environment()
