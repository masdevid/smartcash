"""
File: smartcash/ui/dataset/augmentation/tests/test_ui_components.py
Deskripsi: Test untuk komponen UI augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import ipywidgets as widgets
from IPython.display import display

class TestAugmentationUIComponents(unittest.TestCase):
    """Test case untuk komponen UI augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk test case."""
        # Mock config
        self.mock_config = {
            'augmentation': {
                # Parameter dasar
                'enabled': True,
                'num_variations': 3,
                'output_prefix': 'aug_test',
                'process_bboxes': True,
                'output_dir': 'data/augmented_test',
                'validate_results': True,
                'resume': False,
                'num_workers': 2,
                'balance_classes': True,
                'target_count': 500,
                'move_to_preprocessed': True,
                'target_split': 'train',
                
                # Jenis augmentasi yang didukung
                'types': ['combined', 'flip'],
                'available_types': [
                    'combined', 'flip', 'rotate', 'position', 'lighting'
                ],
                'available_splits': ['train', 'valid', 'test'],
                
                # Parameter augmentasi posisi
                'position': {
                    'fliplr': 0.6,
                    'degrees': 20,
                    'translate': 0.2,
                    'scale': 0.1,
                    'shear_max': 15
                },
                
                # Parameter augmentasi pencahayaan
                'lighting': {
                    'hsv_h': 0.03,
                    'hsv_s': 0.6,
                    'hsv_v': 0.5,
                    'contrast': [0.8, 1.2],
                    'brightness': [0.8, 1.2],
                    'blur': 0.3,
                    'noise': 0.2
                }
            }
        }
        
        # Mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = self.mock_config.get('augmentation')
        
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Mock display
        self.mock_display = MagicMock()
    
    @patch('smartcash.common.config.manager.get_config_manager')
    def test_create_augmentation_options(self, mock_get_config_manager):
        """Test pembuatan komponen augmentation options."""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
        
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        try:
            # Panggil fungsi
            result = create_augmentation_options(self.mock_config)
            
            # Verifikasi hasil
            self.assertIsInstance(result, widgets.VBox)
            self.assertEqual(len(result.children), 1)
            self.assertIsInstance(result.children[0], widgets.Tab)
            
            # Verifikasi bahwa tab memiliki 2 children (basic_options dan augmentation_types_box)
            self.assertEqual(len(result.children[0].children), 2)
            
            # Verifikasi bahwa get_config_manager dipanggil
            mock_get_config_manager.assert_called()
        except Exception as e:
            self.fail(f"create_augmentation_options gagal dengan error: {str(e)}")
    
    @patch('smartcash.common.config.manager.get_config_manager')
    def test_create_advanced_options(self, mock_get_config_manager):
        """Test pembuatan komponen advanced options."""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.components.advanced_options import create_advanced_options
        
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        try:
            # Panggil fungsi
            result = create_advanced_options(self.mock_config)
            
            # Verifikasi hasil
            self.assertIsInstance(result, widgets.VBox)
            self.assertEqual(len(result.children), 1)
            self.assertIsInstance(result.children[0], widgets.Tab)
            
            # Verifikasi bahwa tab memiliki 3 children (position_box, lighting_box, dan additional_box)
            self.assertEqual(len(result.children[0].children), 3)
            
            # Verifikasi bahwa get_config_manager dipanggil
            mock_get_config_manager.assert_called()
        except Exception as e:
            self.fail(f"create_advanced_options gagal dengan error: {str(e)}")
    
    def test_create_augmentation_ui(self):
        """Test pembuatan komponen UI augmentasi."""
        # Gunakan pendekatan alternatif untuk menguji create_augmentation_ui
        # karena terlalu banyak dependensi yang perlu di-mock
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        # Buat mock untuk get_config_manager
        with patch('smartcash.common.config.manager.get_config_manager') as mock_get_config_manager:
            # Setup mock
            mock_get_config_manager.return_value = self.mock_config_manager
            
            try:
                # Coba panggil fungsi dengan config yang valid
                result = create_augmentation_ui(None, {'augmentation': {}})
                
                # Verifikasi bahwa result adalah dictionary
                self.assertIsInstance(result, dict)
                self.assertIn('module_name', result)
                self.assertEqual(result['module_name'], 'augmentation')
            except Exception as e:
                # Jika terjadi error, pastikan itu bukan karena masalah dengan mock
                # tetapi karena masalah dengan implementasi UI yang sebenarnya
                self.fail(f"create_augmentation_ui gagal dengan error: {str(e)}")
    
    def test_create_and_display_augmentation_ui(self):
        """Test pembuatan dan tampilan UI augmentasi."""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.augmentation_initializer import create_and_display_augmentation_ui
        
        # Buat mock untuk fungsi-fungsi yang dipanggil
        with patch('smartcash.ui.dataset.augmentation.augmentation_initializer.initialize_augmentation_ui') as mock_initialize_augmentation_ui, \
             patch('smartcash.ui.dataset.augmentation.augmentation_initializer.display') as mock_display:
            
            # Setup mock
            mock_ui_components = {
                'ui': MagicMock(),
                'augmentation_options': MagicMock(),
                'advanced_options': MagicMock()
            }
            mock_initialize_augmentation_ui.return_value = mock_ui_components
            
            try:
                # Panggil fungsi
                result = create_and_display_augmentation_ui(None, self.mock_config)
                
                # Verifikasi hasil
                self.assertEqual(result, mock_ui_components)
                
                # Verifikasi bahwa initialize_augmentation_ui dipanggil
                mock_initialize_augmentation_ui.assert_called_with(None, self.mock_config)
                
                # Verifikasi bahwa display dipanggil
                mock_display.assert_called_with(mock_ui_components['ui'])
            except Exception as e:
                self.fail(f"create_and_display_augmentation_ui gagal dengan error: {str(e)}")
    
    def test_initialize_augmentation_ui(self):
        """Test inisialisasi UI augmentasi."""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_augmentation_ui
        
        # Buat mock untuk fungsi-fungsi yang dipanggil
        with patch('smartcash.ui.dataset.augmentation.augmentation_initializer.get_logger') as mock_get_logger, \
             patch('smartcash.ui.dataset.augmentation.components.augmentation_component.create_augmentation_ui') as mock_create_augmentation_ui, \
             patch('smartcash.ui.dataset.augmentation.augmentation_initializer.setup_handlers') as mock_setup_handlers, \
             patch('smartcash.ui.dataset.augmentation.handlers.persistence_handler.ensure_ui_persistence') as mock_ensure_ui_persistence, \
             patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_augmentation_info') as mock_update_augmentation_info:
            
            # Setup mock
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            mock_ui_components = {
                'ui': MagicMock(),
                'augmentation_options': MagicMock(),
                'advanced_options': MagicMock(),
                'update_ui_from_config': MagicMock()
            }
            mock_create_augmentation_ui.return_value = mock_ui_components
            mock_setup_handlers.return_value = mock_ui_components
            
            # Panggil fungsi
            try:
                result = initialize_augmentation_ui(None, self.mock_config)
                
                # Verifikasi hasil
                self.assertEqual(result, mock_ui_components)
                
                # Verifikasi bahwa semua fungsi dipanggil
                mock_get_logger.assert_called_once()
                mock_create_augmentation_ui.assert_called_once_with(None, self.mock_config)
                mock_setup_handlers.assert_called_once_with(mock_ui_components, None, self.mock_config)
                mock_update_augmentation_info.assert_called_once_with(mock_ui_components)
                mock_ensure_ui_persistence.assert_called_once_with(mock_ui_components)
                mock_ui_components['update_ui_from_config'].assert_called_once_with(mock_ui_components, self.mock_config)
            except Exception as e:
                self.fail(f"initialize_augmentation_ui gagal dengan error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
