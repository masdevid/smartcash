"""
File: smartcash/ui/training/tests/test_components.py
Deskripsi: Test suite khusus untuk komponen UI training individual
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from typing import Dict, Any

# Import komponen UI
from smartcash.ui.training.components.config_tabs import create_config_tabs
from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
from smartcash.ui.training.components.control_buttons import create_training_control_buttons as create_control_buttons
from smartcash.ui.training.components.fallback_component import (
    create_fallback_component, create_fallback_layout, create_fallback_training_form
)


class TestConfigTabs(unittest.TestCase):
    """Test untuk komponen config tabs"""
    
    def setUp(self):
        """Setup data pengujian dan mocks"""
        self.test_config = {
            'model': {
                'name': 'yolov5-efficientnet-b4',
                'backbone': 'efficientnet-b4',
                'input_size': 640
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001
            }
        }
        
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_HTML = MagicMock()
        self.mock_VBox = MagicMock()
        self.mock_Tab = MagicMock()
        
        self.mock_widgets.HTML.return_value = self.mock_HTML
        self.mock_widgets.VBox.return_value = self.mock_VBox
        self.mock_widgets.Tab.return_value = self.mock_Tab
        
        # Mock tab_factory
        self.mock_create_tabs = patch('smartcash.ui.training.components.config_tabs.create_tabs').start()
        self.mock_create_tabs.return_value = self.mock_Tab
        
        # Prepare mocked Tab result
        self.mock_Tab.children = [self.mock_HTML, self.mock_HTML, self.mock_HTML, self.mock_HTML]

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_config_tabs_structure(self):
        """Test struktur config tabs"""
        result = create_config_tabs(self.test_config)
        
        # Verifikasi struktur result
        self.assertIn('tab_widget', result)
        self.assertIn('config_sections', result)
        
        # Verifikasi tab sections sesuai config
        self.assertEqual(len(result['config_sections']), len(self.test_config))
        
    def test_create_config_tabs_empty_config(self):
        """Test config tabs dengan config kosong"""
        result = create_config_tabs({})
        
        # Verifikasi masih membuat tab valid
        self.assertIn('tab_widget', result)
        self.assertIn('config_sections', result)
        self.assertEqual(len(result['config_sections']), 0)
        
    def test_create_config_tabs_nested_config(self):
        """Test config tabs dengan config nested"""
        nested_config = {
            'data': {
                'train': {
                    'path': '/path/to/train',
                    'annotations': '/path/to/annotations'
                },
                'val': {
                    'path': '/path/to/val',
                    'annotations': '/path/to/val_annotations'
                }
            }
        }
        
        result = create_config_tabs(nested_config)
        
        # Verifikasi nested config dihandle dengan benar
        self.assertIn('tab_widget', result)
        self.assertIn('config_sections', result)
        self.assertEqual(len(result['config_sections']), 1)  # Hanya 'data'


class TestMetricsAccordion(unittest.TestCase):
    """Test untuk komponen metrics accordion"""
    
    def setUp(self):
        """Setup data pengujian dan mocks"""
        self.test_metrics = {
            'train_loss': 0.453,
            'val_loss': 0.521,
            'map': 0.723,
            'f1': 0.789,
            'precision': 0.812,
            'recall': 0.765
        }
        
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_Accordion = MagicMock()
        self.mock_VBox = MagicMock()
        self.mock_Output = MagicMock()
        
        self.mock_widgets.Accordion.return_value = self.mock_Accordion
        self.mock_widgets.VBox.return_value = self.mock_VBox
        self.mock_widgets.Output.return_value = self.mock_Output
        
        # Mock accordion_factory
        self.mock_accordion_factory = patch('smartcash.ui.training.components.metrics_accordion.create_accordion').start()
        self.mock_accordion_factory.return_value = self.mock_Accordion

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_metrics_accordion_structure(self):
        """Test struktur metrics accordion"""
        result = create_metrics_accordion()
        
        # Verifikasi struktur result
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_outputs', result)
        
        # Verifikasi outputs dan chart
        self.assertIn('loss_output', result['chart_outputs'])
        self.assertIn('precision_recall_output', result['chart_outputs'])
        self.assertIn('map_output', result['chart_outputs'])
        
    def test_create_metrics_accordion_empty_metrics(self):
        """Test metrics accordion dengan metrics kosong"""
        result = create_metrics_accordion()
        
        # Verifikasi masih membuat accordion valid
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_output', result)
        self.assertIn('metrics_output', result)
        
    def test_create_metrics_accordion_custom_height(self):
        """Test metrics accordion dengan custom height"""
        custom_height = '400px'
        result = create_metrics_accordion(height=custom_height)
        
        # Verifikasi menggunakan custom height
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_output', result)
        self.assertIn('metrics_output', result)


class TestControlButtons(unittest.TestCase):
    """Test untuk komponen control buttons"""
    
    def setUp(self):
        """Setup mocks"""
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_Button = MagicMock()
        self.mock_HBox = MagicMock()
        self.mock_VBox = MagicMock()
        
        self.mock_widgets.Button.return_value = self.mock_Button
        self.mock_widgets.HBox.return_value = self.mock_HBox
        self.mock_widgets.VBox.return_value = self.mock_VBox
        
        # Mock action_buttons
        self.mock_action_buttons = patch('smartcash.ui.training.components.control_buttons.create_action_buttons').start()
        self.mock_action_buttons.return_value = {
            'download_button': self.mock_Button
        }

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_control_buttons_structure(self):
        """Test struktur control buttons"""
        result = create_control_buttons()
        
        # Verifikasi semua button dibuat
        self.assertIn('start_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('validate_button', result)
        self.assertIn('cleanup_button', result)
        self.assertIn('button_container', result)
        
    def test_create_control_buttons_click_handlers(self):
        """Test handlers untuk control buttons"""
        result = create_control_buttons()
        
        # Verifikasi click handlers di-set
        self.assertEqual(self.mock_Button.on_click.call_count, 5)  # 5 buttons


class TestFallbackComponent(unittest.TestCase):
    """Test untuk komponen fallback"""
    
    def setUp(self):
        """Setup mocks"""
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_HTML = MagicMock()
        self.mock_VBox = MagicMock()
        self.mock_HBox = MagicMock()
        
        self.mock_widgets.HTML.return_value = self.mock_HTML
        self.mock_widgets.VBox.return_value = self.mock_VBox
        self.mock_widgets.HBox.return_value = self.mock_HBox
        
        # Error message
        self.test_error = "Test error message"

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_fallback_component(self):
        """Test pembuatan fallback component"""
        result = create_fallback_component(self.test_error)
        
        # Verifikasi error message dan container
        self.assertIn('error_message', result)
        self.assertIn('fallback_container', result)
        
    def test_create_fallback_layout(self):
        """Test pembuatan fallback layout"""
        mock_components = {
            'error_message': self.mock_HTML
        }
        
        result = create_fallback_layout(mock_components, self.test_error)
        
        # Verifikasi layout container
        self.assertIn('header_section', result)
        self.assertIn('error_section', result)
        self.assertIn('main_container', result)
        
    def test_create_fallback_training_form(self):
        """Test pembuatan fallback training form"""
        result = create_fallback_training_form(self.test_error)
        
        # Verifikasi training form fallback
        self.assertIn('error_message', result)
        self.assertIn('fallback_container', result)


def run_tests():
    """Run test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tambahkan test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfigTabs))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsAccordion))
    suite.addTests(loader.loadTestsFromTestCase(TestControlButtons))
    suite.addTests(loader.loadTestsFromTestCase(TestFallbackComponent))
    
    # Run tests dengan output lebih detail
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
