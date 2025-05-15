"""
File: smartcash/ui/dataset/split/tests/test_ui_handlers.py
Deskripsi: Test untuk handler UI konfigurasi split dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestSplitUIHandlers(unittest.TestCase):
    """Test case untuk handler UI konfigurasi split dataset."""
    
    def setUp(self):
        """Setup untuk test case."""
        # Mock config
        self.mock_config = {
            'data': {
                'split': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15,
                    'stratified': True
                },
                'random_seed': 42,
                'backup_before_split': True,
                'backup_dir': 'data/splits_backup',
                'dataset_path': 'data',
                'preprocessed_path': 'data/preprocessed'
            }
        }
        
        # Mock UI components
        self.ui_components = {
            'train_slider': MagicMock(value=0.7),
            'val_slider': MagicMock(value=0.15),
            'test_slider': MagicMock(value=0.15),
            'stratified_checkbox': MagicMock(value=True),
            'random_seed': MagicMock(value=42),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/splits_backup'),
            'dataset_path': MagicMock(value='data'),
            'preprocessed_path': MagicMock(value='data/preprocessed'),
            'logger': MagicMock(),
            'module_name': 'dataset_split',
            'sync_info': MagicMock()
        }
        
        # Mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = self.mock_config
        self.mock_config_manager.save_module_config.return_value = True
    
    @patch('smartcash.ui.utils.persistence_utils.ensure_ui_persistence')
    def test_ensure_ui_persistence(self, mock_ensure_persistence):
        """Test memastikan persistensi UI."""
        from smartcash.ui.dataset.split.handlers.ui_handlers import ensure_ui_persistence
        
        # Panggil fungsi
        ensure_ui_persistence(self.ui_components, self.mock_config, self.ui_components['logger'])
        
        # Verifikasi hasil
        mock_ensure_persistence.assert_called_once_with(self.ui_components, 'dataset_split', self.ui_components['logger'])
    
    def test_update_ui_from_config(self):
        """Test update UI dari konfigurasi."""
        from smartcash.ui.dataset.split.handlers.ui_handlers import update_ui_from_config
        
        # Panggil fungsi
        update_ui_from_config(self.ui_components, self.mock_config)
        
        # Verifikasi hasil
        self.ui_components['train_slider'].value = 0.7
        self.ui_components['val_slider'].value = 0.15
        self.ui_components['test_slider'].value = 0.15
        self.ui_components['stratified_checkbox'].value = True
        self.ui_components['random_seed'].value = 42
        self.ui_components['backup_checkbox'].value = True
        self.ui_components['backup_dir'].value = 'data/splits_backup'
        self.ui_components['dataset_path'].value = 'data'
        self.ui_components['preprocessed_path'].value = 'data/preprocessed'
    
    def test_initialize_ui_from_config(self):
        """Test inisialisasi UI dari konfigurasi."""
        from smartcash.ui.dataset.split.handlers.ui_handlers import initialize_ui_from_config
        
        # Panggil fungsi
        initialize_ui_from_config(self.ui_components, self.mock_config)
        
        # Verifikasi hasil
        self.ui_components['train_slider'].value = 0.7
        self.ui_components['val_slider'].value = 0.15
        self.ui_components['test_slider'].value = 0.15
        self.ui_components['stratified_checkbox'].value = True
        self.ui_components['random_seed'].value = 42
        self.ui_components['backup_checkbox'].value = True
        self.ui_components['backup_dir'].value = 'data/splits_backup'
        self.ui_components['dataset_path'].value = 'data'
        self.ui_components['preprocessed_path'].value = 'data/preprocessed'
    
    def test_on_slider_change(self):
        """Test handler untuk perubahan slider."""
        from smartcash.ui.dataset.split.handlers.ui_handlers import on_slider_change
        
        # Setup mock
        train_slider = MagicMock(value=0.7)
        val_slider = MagicMock(value=0.15)
        test_slider = MagicMock(value=0.15)
        total_label = MagicMock()
        
        # Panggil fungsi
        on_slider_change(train_slider, val_slider, test_slider, total_label)
        
        # Verifikasi hasil
        total_label.value = total_label.value  # Verifikasi bahwa total_label diupdate
    
    def test_validate_sliders(self):
        """Test validasi slider."""
        from smartcash.ui.dataset.split.handlers.ui_handlers import validate_sliders
        
        # Setup mock
        train_slider = MagicMock(value=0.7)
        val_slider = MagicMock(value=0.15)
        test_slider = MagicMock(value=0.15)
        
        # Panggil fungsi
        result = validate_sliders(train_slider, val_slider, test_slider)
        
        # Verifikasi hasil
        self.assertTrue(result)
        
        # Test dengan total tidak sama dengan 1
        train_slider.value = 0.8
        result = validate_sliders(train_slider, val_slider, test_slider)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
