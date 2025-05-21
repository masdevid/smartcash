"""
File: smartcash/ui/setup/env_config/tests/test_helper.py
Deskripsi: Helper untuk unit testing
"""

import unittest
import warnings
import functools
import os
from unittest.mock import patch
from pathlib import Path

class WarningTestCase(unittest.TestCase):
    """Test case yang menangani warning"""
    
    def setUp(self):
        """Setup untuk test case"""
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

def ignore_layout_warnings(func):
    """Decorator untuk mengabaikan layout warnings"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            return func(*args, **kwargs)
    return wrapper

class MockColabEnvironment:
    """Context manager untuk mock environment Colab"""
    
    def __init__(self):
        self.patches = [
            patch('smartcash.common.utils.is_colab', return_value=True),
            patch('smartcash.common.constants.paths.COLAB_PATH', '/content'),
            patch('os.makedirs', side_effect=lambda path, **kwargs: None),
            patch('pathlib.Path.exists', side_effect=self._mock_path_exists),
            patch('pathlib.Path.mkdir', side_effect=lambda *args, **kwargs: None)
        ]
    
    def _mock_path_exists(self, path):
        """Mock untuk Path.exists"""
        # Direktori root project selalu ada
        if str(path) == '/content':
            return True
        # Untuk path lain, kembalikan False agar test dapat menguji pembuatan direktori
        return False
    
    def __enter__(self):
        """Start all patches"""
        for p in self.patches:
            p.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all patches"""
        for p in self.patches:
            p.stop()
