"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Test untuk download initializer
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.download.download_initializer import initialize_dataset_download_ui

class TestDownloadInitializer(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.base_dir = str(project_root)
        self.config_file = str(project_root / 'smartcash' / 'configs' / 'dataset_config.yaml')
        
        # Get config manager
        self.config_manager = get_config_manager(
            base_dir=self.base_dir,
            config_file=self.config_file
        )
        
        # Load config
        self.config = self.config_manager.load_config()
        
    def test_initialize_dataset_download_ui(self):
        """Test that the download UI initializes correctly"""
        try:
            # Initialize UI with config
            ui = initialize_dataset_download_ui({
                'base_dir': self.base_dir,
                'config_file': self.config_file
            })
            
            # Check that UI was created
            self.assertIsNotNone(ui)
            
            # Check that config was loaded
            self.assertIn('data', self.config)
            self.assertIn('roboflow', self.config['data'])
            
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main() 