"""
File: smartcash/tests/integration/test_evaluation_ui_integration.py
Deskripsi: Test integrasi untuk UI components evaluasi dengan handlers
"""

import os
import sys
import unittest
import numpy as np
import ipywidgets as widgets
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, Any, List

# Pastikan path root project ada di sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import UI components dan handlers
from smartcash.ui.evaluation.components.evaluation_components import (
    create_scenario_selection,
    create_test_data_selection,
    create_progress_tracking_container,
    create_results_container
)
from smartcash.ui.evaluation.handlers.evaluation_handler import (
    setup_evaluation_handlers,
    run_evaluation_process
)
from smartcash.ui.utils.button_state_manager import get_button_state_manager

class TestEvaluationUIIntegration(unittest.TestCase):
    """Test integrasi untuk UI components evaluasi dengan handlers"""
    
    def setUp(self):
        """Setup untuk test"""
        # Mock logger
        self.logger = MagicMock()
        
        # Mock config
        self.config = {
            'scenario': {
                'selected_scenario': 'test_scenario',
                'scenarios': {
                    'test_scenario': {
                        'name': 'Skenario Test',
                        'backbone': 'cspdarknet_s',
                        'augmentation_type': 'default'
                    },
                    'scenario_2': {
                        'name': 'Skenario 2',
                        'backbone': 'efficientnet_b4',
                        'augmentation_type': 'position'
                    }
                }
            },
            'model': {
                'img_size': 416,
                'use_custom_checkpoint': False
            },
            'test_data': {
                'batch_size': 4,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'evaluation': {
                'class_names': ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp75000', 'Rp100000']
            }
        }
        
        # Mock test folder
        self.test_folder = '/tmp/smartcash_test'
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(os.path.join(self.test_folder, 'images'), exist_ok=True)
    
    def tearDown(self):
        """Cleanup setelah test"""
        # Hapus dummy test data
        import shutil
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
    
    @patch('ipywidgets.Dropdown')
    def test_create_scenario_selection(self, mock_dropdown):
        """Test pembuatan UI component untuk pemilihan skenario"""
        # Setup mock
        mock_dropdown_instance = MagicMock()
        mock_dropdown.return_value = mock_dropdown_instance
        
        # Panggil fungsi
        result = create_scenario_selection(self.config)
        
        # Verifikasi
        self.assertIsNotNone(result)
        self.assertIn('scenario_dropdown', result)
        mock_dropdown.assert_called_once()
        
        # Verifikasi options dropdown
        call_args = mock_dropdown.call_args[1]
        self.assertIn('options', call_args)
        self.assertEqual(len(call_args['options']), 2)  # Dua skenario dalam config
    
    @patch('ipywidgets.Text')
    @patch('ipywidgets.Button')
    def test_create_test_data_selection(self, mock_button, mock_text):
        """Test pembuatan UI component untuk pemilihan test data"""
        # Setup mocks
        mock_text_instance = MagicMock()
        mock_button_instance = MagicMock()
        mock_text.return_value = mock_text_instance
        mock_button.return_value = mock_button_instance
        
        # Panggil fungsi
        result = create_test_data_selection()
        
        # Verifikasi
        self.assertIsNotNone(result)
        self.assertIn('test_data_path', result)
        self.assertIn('browse_button', result)
        mock_text.assert_called_once()
        mock_button.assert_called_once()
    
    @patch('ipywidgets.HTML')
    @patch('ipywidgets.FloatProgress')
    def test_create_progress_tracking_container(self, mock_progress, mock_html):
        """Test pembuatan UI component untuk progress tracking"""
        # Setup mocks
        mock_progress_instance = MagicMock()
        mock_html_instance = MagicMock()
        mock_progress.return_value = mock_progress_instance
        mock_html.return_value = mock_html_instance
        
        # Panggil fungsi
        result = create_progress_tracking_container()
        
        # Verifikasi
        self.assertIsNotNone(result)
        self.assertIn('progress_container', result)
        self.assertIn('progress_bar', result)
        self.assertIn('status_text', result)
        mock_progress.assert_called()
        mock_html.assert_called()
    
    @patch('ipywidgets.Output')
    def test_create_results_container(self, mock_output):
        """Test pembuatan UI component untuk hasil evaluasi"""
        # Setup mock
        mock_output_instance = MagicMock()
        mock_output.return_value = mock_output_instance
        
        # Panggil fungsi
        result = create_results_container()
        
        # Verifikasi
        self.assertIsNotNone(result)
        self.assertIn('result_area', result)
        mock_output.assert_called_once()
    
    @patch('smartcash.ui.utils.button_state_manager.get_button_state_manager')
    def test_setup_evaluation_handlers(self, mock_get_manager):
        """Test setup handlers untuk evaluation process"""
        # Setup mocks
        mock_button_manager = MagicMock()
        mock_get_manager.return_value = mock_button_manager
        
        mock_evaluate_button = MagicMock()
        
        # Setup UI components
        ui_components = {
            'evaluate_button': mock_evaluate_button,
            'logger': self.logger,
            'progress_container': MagicMock(),
            'progress_bar': MagicMock(),
            'status_text': MagicMock()
        }
        
        # Panggil fungsi
        with patch('smartcash.ui.evaluation.handlers.evaluation_handler.run_evaluation_process') as mock_run:
            result = setup_evaluation_handlers(ui_components, self.config)
            
            # Verifikasi
            self.assertEqual(result, ui_components)
            mock_get_manager.assert_called_once_with(ui_components)
            
            # Verifikasi button click handler
            on_click_handler = mock_evaluate_button.on_click.call_args[0][0]
            self.assertIsNotNone(on_click_handler)
    
    @patch('smartcash.model.utils.evaluation_pipeline.run_evaluation_pipeline')
    def test_integration_handlers_with_components(self, mock_run_pipeline):
        """Test integrasi antara UI components dan handlers"""
        # Setup mocks
        mock_run_pipeline.return_value = {
            'success': True,
            'metrics': {
                'map': 0.85,
                'overall_metrics': {
                    'precision': 0.9,
                    'recall': 0.8,
                    'f1_score': 0.85
                },
                'class_metrics': {
                    'Rp1000': {'precision': 0.9, 'recall': 0.8, 'f1_score': 0.85, 'ap': 0.85}
                }
            },
            'predictions': [{'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]), 'scores': np.array([0.9]), 'classes': np.array([0])}]
        }
        
        # Buat UI components
        with patch('ipywidgets.Dropdown') as mock_dropdown, \
             patch('ipywidgets.Text') as mock_text, \
             patch('ipywidgets.Button') as mock_button, \
             patch('ipywidgets.HTML') as mock_html, \
             patch('ipywidgets.FloatProgress') as mock_progress, \
             patch('ipywidgets.Output') as mock_output:
            
            # Setup mock instances
            mock_dropdown_instance = MagicMock(value='test_scenario')
            mock_text_instance = MagicMock(value=self.test_folder)
            mock_button_instance = MagicMock()
            mock_html_instance = MagicMock()
            mock_progress_instance = MagicMock()
            mock_output_instance = MagicMock()
            
            mock_dropdown.return_value = mock_dropdown_instance
            mock_text.return_value = mock_text_instance
            mock_button.return_value = mock_button_instance
            mock_html.return_value = mock_html_instance
            mock_progress.return_value = mock_progress_instance
            mock_output.return_value = mock_output_instance
            
            # Buat UI components
            scenario_components = create_scenario_selection(self.config)
            test_data_components = create_test_data_selection()
            progress_components = create_progress_tracking_container()
            results_components = create_results_container()
            
            # Gabungkan components
            ui_components = {
                **scenario_components,
                **test_data_components,
                **progress_components,
                **results_components,
                'logger': self.logger,
                'show_for_operation': MagicMock(),
                'error_operation': MagicMock(),
                'success_operation': MagicMock(),
                'warning_operation': MagicMock(),
                'update_progress': MagicMock()
            }
            
            # Setup handlers
            with patch('smartcash.ui.utils.button_state_manager.get_button_state_manager') as mock_get_manager:
                mock_button_manager = MagicMock()
                mock_button_manager.operation_context.return_value.__enter__ = MagicMock()
                mock_button_manager.operation_context.return_value.__exit__ = MagicMock()
                mock_get_manager.return_value = mock_button_manager
                
                # Setup handlers
                setup_evaluation_handlers(ui_components, self.config)
                
                # Simulasikan klik tombol evaluasi
                run_evaluation_process(ui_components, self.config, self.logger, mock_button_manager)
                
                # Verifikasi
                mock_run_pipeline.assert_called_once()
                ui_components['show_for_operation'].assert_called_once()
                ui_components['update_progress'].assert_called()
                ui_components['success_operation'].assert_called()

if __name__ == '__main__':
    unittest.main()
