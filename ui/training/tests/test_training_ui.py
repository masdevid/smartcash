"""
File: smartcash/ui/training/tests/test_training_ui.py
Deskripsi: Test suite untuk menguji fungsionalitas komponen UI training
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from typing import Dict, Any

# Import komponen training
from smartcash.ui.training.components.training_form import create_training_form
from smartcash.ui.training.components.training_layout import create_training_layout
from smartcash.ui.training.components.config_tabs import create_config_tabs
from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
from smartcash.ui.training.components.control_buttons import create_training_control_buttons as create_control_buttons
from smartcash.ui.training.components.fallback_component import create_fallback_layout
from smartcash.ui.training.training_init import TrainingInitializer


class TestTrainingComponents(unittest.TestCase):
    """Test untuk individual UI components"""
    
    def setUp(self):
        """Setup data pengujian dan mocks"""
        self.mock_config = {
            'model': {
                'name': 'yolov5-efficientnet-b4',
                'backbone': 'efficientnet-b4',
                'anchors': '5,6, 7,8, 9,10',
                'input_size': 640
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937
            },
            'data': {
                'train': '/path/to/train',
                'val': '/path/to/val',
                'classes': ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'rotate': 10,
                'shear': 2.0
            }
        }
        
        self.mock_metrics = {
            'train_loss': 0.453,
            'val_loss': 0.521,
            'map': 0.723,
            'f1': 0.789,
            'precision': 0.812,
            'recall': 0.765
        }
        
        # Patch untuk ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_widgets.HTML = MagicMock()
        self.mock_widgets.Button = MagicMock()
        self.mock_widgets.VBox = MagicMock()
        self.mock_widgets.HBox = MagicMock()
        self.mock_widgets.Tab = MagicMock()
        self.mock_widgets.Accordion = MagicMock()
        self.mock_widgets.Output = MagicMock()

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_config_tabs(self):
        """Test pembuatan config tabs"""
        result = create_config_tabs(self.mock_config)
        
        # Verifikasi tab dibuat dengan benar
        self.assertIn('tab_widget', result)
        self.assertIn('config_sections', result)
        self.mock_widgets.Tab.assert_called()

    def test_create_metrics_accordion(self):
        """Test pembuatan metrics accordion"""
        result = create_metrics_accordion()
        
        # Verifikasi accordion dibuat dengan benar
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_output', result)
        self.assertIn('metrics_output', result)
        self.mock_widgets.Output.assert_called()

    def test_create_control_buttons(self):
        """Test pembuatan control buttons"""
        result = create_control_buttons()
        
        # Verifikasi button handlers
        self.assertIn('start_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('validate_button', result)
        self.assertIn('cleanup_button', result)
        self.mock_widgets.Button.assert_called()

    def test_create_training_form(self):
        """Test pembuatan training form"""
        result = create_training_form(self.mock_config)
        
        # Verifikasi form components
        self.assertIn('control_buttons', result)
        self.assertIn('config_tabs', result)
        self.assertIn('metrics_section', result)

    def test_create_training_form_with_error(self):
        """Test pembuatan training form dengan error handling"""
        with patch('smartcash.ui.training.components.training_form.create_config_tabs', 
                  side_effect=Exception('Test error')):
            result = create_training_form(self.mock_config)
            
            # Verifikasi fallback digunakan
            self.assertIn('error_message', result)
            self.assertIn('fallback_container', result)

    def test_create_training_layout(self):
        """Test pembuatan training layout"""
        form_components = create_training_form(self.mock_config)
        result = create_training_layout(form_components)
        
        # Verifikasi layout sections
        self.assertIn('header_section', result)
        self.assertIn('info_section', result)
        self.assertIn('control_section', result)
        self.assertIn('progress_section', result)
        self.assertIn('metrics_section', result)
        self.assertIn('log_section', result)
        self.assertIn('main_container', result)


class TestTrainingInitializer(unittest.TestCase):
    """Test untuk training initializer"""
    
    def setUp(self):
        """Setup mocks untuk pengujian"""
        self.mock_config = {
            'model': {'name': 'yolov5-efficientnet-b4'},
            'training': {'batch_size': 16, 'epochs': 100},
            'data': {'classes': ['Rp1000', 'Rp2000']}
        }
        
        # Mock config manager sebagai class mock
        self.mock_config_manager_class = patch('smartcash.ui.training.training_init.ConfigManager').start()
        self.mock_config_manager_instance = self.mock_config_manager_class.return_value
        self.mock_config_manager_instance.get_config.return_value = self.mock_config
        self.mock_config_manager_instance.register_callback = MagicMock()
        
        # Mock training components
        self.mock_create_form = patch('smartcash.ui.training.training_init.create_training_form').start()
        self.mock_create_layout = patch('smartcash.ui.training.training_init.create_training_layout').start()
        
        # Mock display
        self.mock_display = patch('smartcash.ui.training.training_init.display').start()
        
        # Dummy UI components
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'config_tabs': {'tab_widget': MagicMock()},
            'metrics_section': {'metrics_accordion': MagicMock()},
            'progress_container': {'tracker': MagicMock()},
            'log_container': {'output': MagicMock()}
        }
        
        self.mock_create_form.return_value = {'config_tabs': self.mock_ui_components['config_tabs']}
        self.mock_create_layout.return_value = self.mock_ui_components

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_initialize(self):
        """Test inisialisasi training UI"""
        initializer = TrainingInitializer()
        result = initializer.initialize()
        
        # Verifikasi UI components
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_form.assert_called_once()
        self.mock_create_layout.assert_called_once()
        self.mock_display.assert_called_once_with(self.mock_ui_components['main_container'])

    def test_config_callback(self):
        """Test callback config update"""
        initializer = TrainingInitializer()
        initializer.initialize()
        
        # Simulasi callback config
        callback_fn = self.mock_config_manager_instance.register_callback.call_args[0][0]
        
        # Update config dan panggil callback
        new_config = {'model': {'name': 'updated-model'}}
        callback_fn(new_config)
        
        # Verifikasi UI diperbarui
        self.mock_create_form.assert_called_with(new_config)
        self.mock_display.assert_called()

    def test_prevent_duplicate_display(self):
        """Test pencegahan duplicate display"""
        initializer = TrainingInitializer()
        initializer.initialize()
        
        # Reset mock untuk pengujian kedua
        self.mock_display.reset_mock()
        
        # Coba inisialisasi lagi
        initializer.initialize()
        
        # Verifikasi display tidak dipanggil lagi
        self.mock_display.assert_not_called()


def run_tests():
    """Run test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tambahkan test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingInitializer))
    
    # Run tests dengan output lebih detail
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
