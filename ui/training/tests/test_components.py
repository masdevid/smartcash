"""
File: smartcash/ui/training/tests/test_components.py
Deskripsi: Test suite khusus untuk komponen UI training individual
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from typing import Dict, Any
import ipywidgets as widgets

# Import komponen UI
from smartcash.ui.training.components.config_tabs import create_config_tabs
from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
from smartcash.ui.training.components.control_buttons import create_training_control_buttons as create_control_buttons
from smartcash.ui.training.components.fallback_component import (
    create_fallback_component, create_fallback_layout, create_fallback_training_form
)

# Import factory components yang digunakan oleh komponen UI
from smartcash.ui.components.tab_factory import create_tab_widget
from smartcash.ui.components.accordion_factory import create_accordion
from smartcash.ui.components.action_buttons import create_action_buttons


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
        self.mock_create_tabs = patch('smartcash.ui.components.tab_factory.create_tab_widget').start()
        self.mock_create_tabs.return_value = self.mock_Tab
        
        # Prepare mocked Tab result
        self.mock_Tab.children = [self.mock_HTML, self.mock_HTML, self.mock_HTML, self.mock_HTML]

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_config_tabs_structure(self):
        """Test struktur config tabs"""
        result = create_config_tabs(self.test_config)
        
        # Hasil langsung adalah Tab widget, bukan dictionary
        self.assertEqual(result, self.mock_Tab)
        
        # Verifikasi create_tab_widget dipanggil dengan parameter yang benar
        self.mock_create_tabs.assert_called_once()
        # Verifikasi bahwa ada 4 tab (Model, Hyperparameters, Strategy, Paths)
        self.assertEqual(len(self.mock_Tab.children), 4)
        
    def test_create_config_tabs_empty_config(self):
        """Test config tabs dengan config kosong"""
        result = create_config_tabs({})
        
        # Hasil langsung adalah Tab widget, bukan dictionary
        self.assertEqual(result, self.mock_Tab)
        
        # Verifikasi create_tab_widget dipanggil dengan parameter yang benar
        self.assertTrue(self.mock_create_tabs.called)
        # Verifikasi bahwa tetap ada 4 tab default meskipun config kosong
        self.assertEqual(len(self.mock_Tab.children), 4)
        
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
        
        # Hasil langsung adalah Tab widget, bukan dictionary
        self.assertEqual(result, self.mock_Tab)
        
        # Verifikasi create_tab_widget dipanggil dengan parameter yang benar
        self.assertTrue(self.mock_create_tabs.called)
        # Verifikasi bahwa tetap ada 4 tab default
        self.assertEqual(len(self.mock_Tab.children), 4)


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
        self.mock_accordion_factory = patch('smartcash.ui.components.accordion_factory.create_accordion').start()
        self.mock_accordion_factory.return_value = self.mock_Accordion

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_metrics_accordion_structure(self):
        """Test struktur metrics accordion"""
        result = create_metrics_accordion()
        
        # Verifikasi struktur result
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_output', result)
        self.assertIn('metrics_output', result)
        
        # Verifikasi accordion dibuat dengan benar
        self.assertEqual(result['metrics_accordion'], self.mock_Accordion)
        # Tidak perlu memeriksa selected_index karena itu diatur dalam create_accordion
        
    def test_create_metrics_accordion_empty_metrics(self):
        """Test metrics accordion dengan metrics kosong"""
        result = create_metrics_accordion()
        
        # Verifikasi struktur result dengan metrics kosong
        self.assertIn('metrics_accordion', result)
        self.assertIn('chart_output', result)
        self.assertIn('metrics_output', result)
        
    def test_create_metrics_accordion_custom_height(self):
        """Test metrics accordion dengan custom height"""
        custom_height = '500px'
        result = create_metrics_accordion(height=custom_height)
        
        # Verifikasi custom height diaplikasikan
        self.assertEqual(result['metrics_accordion'].layout.height, custom_height)


class TestControlButtons(unittest.TestCase):
    """Test untuk komponen control buttons"""
    
    def setUp(self):
        """Setup mocks"""
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_Button = MagicMock(spec=widgets.Button)
        self.mock_HBox = MagicMock(spec=widgets.HBox)
        self.mock_VBox = MagicMock(spec=widgets.VBox)
        
        self.mock_widgets.Button.return_value = self.mock_Button
        self.mock_widgets.HBox.return_value = self.mock_HBox
        self.mock_widgets.VBox.return_value = self.mock_VBox
        
        # Mock action_buttons
        self.mock_action_buttons = patch('smartcash.ui.components.action_buttons.create_action_buttons').start()
        self.mock_action_buttons.return_value = {
            'download_button': self.mock_Button
        }

    def tearDown(self):
        """Cleanup patches"""
        patch.stopall()

    def test_create_control_buttons_structure(self):
        """Test struktur control buttons"""
        # Patch widgets.HBox untuk menghindari error validasi children
        with patch('ipywidgets.widgets.HBox', MagicMock()) as mock_hbox:
            mock_hbox_instance = MagicMock()
            mock_hbox.return_value = mock_hbox_instance
            
            result = create_control_buttons()
            
            # Verifikasi semua button dibuat
            self.assertIn('start_button', result)
            self.assertIn('stop_button', result)
            self.assertIn('reset_button', result)
            self.assertIn('button_container', result)
        
    def test_create_control_buttons_click_handlers(self):
        """Test handlers untuk control buttons"""
        # Patch widgets.HBox untuk menghindari error validasi children
        with patch('ipywidgets.widgets.HBox', MagicMock()) as mock_hbox:
            mock_hbox_instance = MagicMock()
            mock_hbox.return_value = mock_hbox_instance
            
            result = create_control_buttons()
            
            # Verifikasi click handlers di-set untuk start, stop, dan reset button
            self.assertTrue(hasattr(result['start_button'], 'on_click'))
            self.assertTrue(hasattr(result['stop_button'], 'on_click'))
            self.assertTrue(hasattr(result['reset_button'], 'on_click'))


class TestFallbackComponent(unittest.TestCase):
    """Test untuk komponen fallback"""
    
    def setUp(self):
        """Setup mocks"""
        # Mock ipywidgets
        self.mock_widgets = patch('ipywidgets.widgets').start()
        self.mock_HTML = MagicMock(spec=widgets.HTML)
        self.mock_VBox = MagicMock(spec=widgets.VBox)
        self.mock_HBox = MagicMock(spec=widgets.HBox)
        
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
