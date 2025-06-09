"""
File: tests/dataset/preprocessor/conftest.py
Deskripsi: Fixtures untuk unit test preprocessor
"""
import os
import shutil
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from PIL import Image, ImageDraw

# Import mock classes - menggunakan import langsung untuk menghindari circular import
class MockPreprocessingEngine:
    """Mock untuk PreprocessingEngine"""
    def __init__(self, config):
        self.config = config
        self.preprocess_called = False
        self.validator = MockPreprocessingValidator(config)
        self.file_processor = MagicMock()
        self.file_scanner = MagicMock()
        self.path_resolver = MagicMock()
        self.cleanup_manager = MagicMock()
    
    def preprocess_split(self, *args, **kwargs):
        self.preprocess_called = True
        return {
            'status': 'success',
            'total_processed': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'invalid_samples': []
        }
    
    def cleanup(self):
        self.cleanup_manager.cleanup()


class MockPreprocessingValidator:
    """Mock untuk PreprocessingValidator"""
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or MagicMock()
        self.validate_called = False
        self.image_validator = MagicMock()
        self.label_validator = MagicMock()
        self.pair_validator = MagicMock()
    
    def validate_dataset(self, *args, **kwargs):
        self.validate_called = True
        return {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'invalid_samples': []
        }

# Direktori test
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
RAW_IMAGE_DIR = TEST_DATA_DIR / 'raw' / 'train' / 'images'
RAW_LABEL_DIR = TEST_DATA_DIR / 'raw' / 'train' / 'labels'
PREPROCESSED_IMAGE_DIR = TEST_DATA_DIR / 'preprocessed' / 'train' / 'images'
PREPROCESSED_LABEL_DIR = TEST_DATA_DIR / 'preprocessed' / 'train' / 'labels'

# Konfigurasi test
TEST_CONFIG = {
    'preprocessing': {
        'enabled': True,
        'validation': {
            'enabled': True,
            'move_invalid': False,
            'fix_issues': False
        },
        'normalization': {
            'method': 'minmax',
            'target_size': [640, 640],
            'preserve_aspect_ratio': False,
            'denormalize': False
        },
        'output': {
            'create_npy': True,
            'organize_by_split': True
        }
    },
    'file_naming': {
        'preprocessed_pattern': 'pre_{nominal}_{uuid}_{increment}',
        'preserve_uuid': True
    },
    'data': {
        'dir': str(TEST_DATA_DIR),
        'splits': {
            'train': str(TEST_DATA_DIR / 'raw' / 'train'),
            'valid': str(TEST_DATA_DIR / 'raw' / 'valid'),
            'test': str(TEST_DATA_DIR / 'raw' / 'test')
        },
        'output': {
            'preprocessed': str(TEST_DATA_DIR / 'preprocessed')
        }
    }
}

class MockProgressTracker:
    """
    Mock untuk ProgressTracker
    
    Digunakan untuk menangkap update progress dan status selama testing.
    Kelas ini menyimulasikan behavior ProgressTracker asli untuk keperluan testing.
    """
    def __init__(self):
        # Inisialisasi state tracker
        self.progress_updates = []  # Menyimpan semua update progress (progress, status)
        self.completed = False      # Flag penanda proses selesai
        self.error = None           # Menyimpan error jika terjadi
        self.messages = []          # Menyimpan pesan-pesan (tipe, pesan)
    
    def update(self, progress, status=None):
        """
        Simulasi update progress
        
        Args:
            progress (float): Nilai progress 0-100
            status (str, optional): Status saat ini. Defaults to None.
            
        Returns:
            bool: Selalu return True
        """
        # Validasi input
        if not isinstance(progress, (int, float)) or not (0 <= progress <= 100):
            raise ValueError("Progress harus berupa angka antara 0-100")
            
        self.progress_updates.append((float(progress), str(status) if status else None))
        return True
    
    def complete(self, message=None):
        """
        Tandai proses selesai
        
        Args:
            message (str, optional): Pesan penyelesaian. Defaults to None.
            
        Returns:
            bool: Selalu return True
        """
        self.completed = True
        if message:
            self.messages.append(('complete', str(message)))
        return True
    
    def error_occurred(self, error):
        """
        Catat error yang terjadi
        
        Args:
            error (Exception): Error yang terjadi
            
        Returns:
            bool: Selalu return True
        """
        if error is None:
            raise ValueError("Error tidak boleh None")
            
        self.error = error
        self.messages.append(('error', str(error)))
        return True
    
    def get_last_progress(self):
        """
        Ambil progress terakhir
        
        Returns:
            tuple: (progress, status) atau (None, None) jika belum ada
        """
        if not self.progress_updates:
            return None, None
        return self.progress_updates[-1]
    
    def was_progress_made(self):
        """
        Cek apakah ada progress yang tercatat
        
        Returns:
            bool: True jika ada progress, False jika tidak
        """
        return bool(self.progress_updates)
    
    def was_completed(self):
        """
        Cek apakah proses selesai
        
        Returns:
            bool: True jika selesai, False jika tidak
        """
        return self.completed
    
    def has_error(self):
        """
        Cek apakah terjadi error
        
        Returns:
            bool: True jika ada error, False jika tidak
        """
        return self.error is not None
    
    def reset(self):
        """
        Reset state tracker ke kondisi awal
        
        Returns:
            MockProgressTracker: Instance ini untuk method chaining
        """
        self.progress_updates = []
        self.completed = False
        self.error = None
        self.messages = []
        return self
    
    # Alias untuk kompatibilitas
    error = error_occurred

def create_test_image(filename, size=(800, 600), text=None):
    """Buat gambar test"""
    img = Image.new('RGB', size, color='white')
    if text:
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, fill='black')
    img.save(filename, 'PNG')
    return filename

def create_test_label(filename, num_objects=1):
    """Buat file label test dalam format YOLO"""
    with open(filename, 'w') as f:
        for i in range(num_objects):
            # Format: class_id x_center y_center width height
            f.write(f"0 0.5 0.5 0.2 0.2\n")
    return filename

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup environment untuk testing"""
    # Buat direktori jika belum ada
    os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
    os.makedirs(RAW_LABEL_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_LABEL_DIR, exist_ok=True)
    
    # Buat beberapa file test
    for i in range(5):
        img_path = RAW_IMAGE_DIR / f"test_{i}.png"
        label_path = RAW_LABEL_DIR / f"test_{i}.txt"
        
        create_test_image(img_path, text=f"Test Image {i}")
        create_test_label(label_path)
    
    yield  # Test berjalan di sini
    
    # Bersihkan setelah test selesai
    if os.path.exists(TEST_DATA_DIR / 'preprocessed'):
        shutil.rmtree(TEST_DATA_DIR / 'preprocessed')

@pytest.fixture
def test_config():
    """Fixture untuk konfigurasi test"""
    return TEST_CONFIG.copy()

@pytest.fixture
def progress_tracker():
    """Fixture untuk mock progress tracker"""
    return MockProgressTracker()

# Fixture preprocessor_service sudah dipindahkan ke test_preprocessor.py
# untuk menghindari circular import
