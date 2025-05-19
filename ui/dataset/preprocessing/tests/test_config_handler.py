"""
File: smartcash/ui/dataset/preprocessing/tests/test_config_handler.py
Deskripsi: Unit test untuk config handler preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import ipywidgets as widgets
import os
import yaml

class TestConfigHandler(unittest.TestCase):
    """Test untuk config handler preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk config manager
        self.config_manager_patch = patch('smartcash.common.config.manager.get_config_manager')
        self.mock_config_manager = self.config_manager_patch.start()
        self.mock_config_manager.return_value = MagicMock()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'save_button': widgets.Button(description='Save'),
            'split_selector': widgets.Dropdown(
                options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
                value='All Splits'
            ),
            'preprocess_options': widgets.VBox([
                widgets.IntSlider(value=640, min=32, max=1280, description='Size:'),
                widgets.Checkbox(value=True, description='Normalize'),
                widgets.Checkbox(value=True, description='Preserve Aspect Ratio'),
                widgets.Checkbox(value=True, description='Cache'),
                widgets.IntSlider(value=4, min=1, max=16, description='Workers:')
            ]),
            'validation_options': widgets.VBox([
                widgets.Checkbox(value=True, description='Validate'),
                widgets.Checkbox(value=True, description='Fix Issues'),
                widgets.Checkbox(value=True, description='Move Invalid'),
                widgets.Text(value='data/invalid', description='Invalid Dir:')
            ])
        }
        
        # Mock config
        self.mock_config = {
            'data': {
                'dir': 'data'
            },
            'preprocessing': {
                'img_size': [640, 640],
                'output_dir': 'data/preprocessed',
                'splits': ['train', 'valid', 'test'],
                'enabled': True,
                'num_workers': 4,
                'normalization': {
                    'enabled': True,
                    'preserve_aspect_ratio': True
                },
                'validate': {
                    'enabled': True,
                    'fix_issues': True,
                    'move_invalid': True,
                    'invalid_dir': 'data/invalid'
                }
            }
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.config_manager_patch.stop()
    
    def test_update_config_from_ui(self):
        """Test untuk fungsi update_config_from_ui."""
        from smartcash.ui.dataset.preprocessing.utils.config_utils import update_config_from_ui
        
        # Panggil fungsi yang akan ditest
        result = update_config_from_ui(self.mock_ui_components, {})
        
        # Verifikasi hasil
        self.assertTrue('preprocessing' in result)
        self.assertTrue('data' in result)
        preproc_config = result['preprocessing']
        
        # Verifikasi nilai yang diambil dari UI
        self.assertEqual(preproc_config['img_size'], [640, 640])
        self.assertTrue(preproc_config['normalization']['enabled'])
        self.assertTrue(preproc_config['normalization']['preserve_aspect_ratio'])
        self.assertTrue(preproc_config['enabled'])
        self.assertEqual(preproc_config['num_workers'], 4)
        self.assertTrue(preproc_config['validate']['enabled'])
        self.assertTrue(preproc_config['validate']['fix_issues'])
        self.assertTrue(preproc_config['validate']['move_invalid'])
        self.assertEqual(preproc_config['validate']['invalid_dir'], 'data/invalid')
        self.assertEqual(preproc_config['splits'], ['train', 'valid', 'test'])
    
    @patch('os.path.exists')
    @patch('yaml.dump')
    def test_save_preprocessing_config(self, mock_yaml_dump, mock_exists):
        """Test untuk fungsi save_preprocessing_config."""
        from smartcash.ui.dataset.preprocessing.utils.config_utils import save_preprocessing_config
        
        # Setup mock
        mock_exists.return_value = True
        mock_open_patch = patch('builtins.open', mock_open())
        mock_open_obj = mock_open_patch.start()
        
        try:
            # Panggil fungsi yang akan ditest
            result = save_preprocessing_config(self.mock_config, 'configs/preprocessing_config.yaml')
            
            # Verifikasi hasil
            self.assertTrue(result)
            mock_open_obj.assert_called_with('configs/preprocessing_config.yaml', 'w')
            mock_yaml_dump.assert_called_once_with(self.mock_config, mock_open_obj.return_value.__enter__.return_value, default_flow_style=False)
            self.mock_config_manager.return_value.save_module_config.assert_called_once_with('preprocessing', self.mock_config)
        finally:
            mock_open_patch.stop()
    
    @patch('os.path.exists')
    @patch('yaml.safe_load')
    def test_load_preprocessing_config_from_file(self, mock_yaml_load, mock_exists):
        """Test untuk fungsi load_preprocessing_config dari file."""
        from smartcash.ui.dataset.preprocessing.utils.config_utils import load_preprocessing_config
        
        # Setup mock
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.mock_config
        mock_open_patch = patch('builtins.open', mock_open())
        mock_open_obj = mock_open_patch.start()
        
        # Setup config manager mock
        self.mock_config_manager.return_value.get_module_config.return_value = None
        
        try:
            # Panggil fungsi yang akan ditest
            result = load_preprocessing_config('configs/preprocessing_config.yaml', self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result, self.mock_config)
            mock_open_obj.assert_called_with('configs/preprocessing_config.yaml', 'r')
            self.mock_config_manager.return_value.get_module_config.assert_called_once_with('preprocessing')
            self.mock_config_manager.return_value.save_module_config.assert_called_once_with('preprocessing', self.mock_config)
        finally:
            mock_open_patch.stop()
    
    def test_load_preprocessing_config_from_config_manager(self):
        """Test untuk fungsi load_preprocessing_config dari config manager."""
        from smartcash.ui.dataset.preprocessing.utils.config_utils import load_preprocessing_config
        
        # Setup mock
        self.mock_config_manager.return_value.get_module_config.return_value = self.mock_config
        
        # Panggil fungsi yang akan ditest
        result = load_preprocessing_config('configs/preprocessing_config.yaml', self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_config)
        self.mock_config_manager.return_value.get_module_config.assert_called_once_with('preprocessing')
    
    def test_update_ui_from_config(self):
        """Test untuk fungsi update_ui_from_config."""
        from smartcash.ui.dataset.preprocessing.utils.config_utils import update_ui_from_config
        
        # Panggil fungsi yang akan ditest
        result = update_ui_from_config(self.mock_ui_components, self.mock_config)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.assertEqual(result['data_dir'], 'data')
        self.assertEqual(result['preprocessed_dir'], 'data/preprocessed')
        self.assertEqual(result['config'], self.mock_config)
        
        # Verifikasi nilai yang diupdate ke UI
        self.assertEqual(result['preprocess_options'].children[0].value, 640)
        self.assertTrue(result['preprocess_options'].children[1].value)
        self.assertTrue(result['preprocess_options'].children[2].value)
        self.assertTrue(result['preprocess_options'].children[3].value)
        self.assertEqual(result['preprocess_options'].children[4].value, 4)
        
        self.assertTrue(result['validation_options'].children[0].value)
        self.assertTrue(result['validation_options'].children[1].value)
        self.assertTrue(result['validation_options'].children[2].value)
        self.assertEqual(result['validation_options'].children[3].value, 'data/invalid')
        
        self.assertEqual(result['split_selector'].value, 'All Splits')

if __name__ == '__main__':
    unittest.main()
