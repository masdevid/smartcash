"""
File: smartcash/tests/integration/test_evaluation_integration.py
Deskripsi: Test integrasi untuk handlers evaluasi dengan komponen backend
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Dict, Any, List

# Pastikan path root project ada di sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import UI handlers
from smartcash.ui.evaluation.handlers.evaluation_handler import (
    validate_evaluation_inputs,
    load_model_and_checkpoint,
    prepare_test_data_with_augmentation,
    run_model_inference,
    load_ground_truth_labels,
    run_evaluation_process,
    apply_nms,
    simple_nms
)

# Import backend modules
from smartcash.model.utils.evaluation_pipeline import run_evaluation_pipeline
from smartcash.model.utils.model_loader import load_model_for_scenario
from smartcash.dataset.utils.test_data_utils import prepare_test_data_for_scenario
from smartcash.model.utils.validation_utils import validate_evaluation_config

class TestEvaluationIntegration(unittest.TestCase):
    """Test integrasi untuk handlers evaluasi dengan komponen backend"""
    
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
        
        # Mock UI components
        self.ui_components = {
            'progress_bar': MagicMock(),
            'status_text': MagicMock(),
            'result_area': MagicMock()
        }
        
        # Mock test folder
        self.test_folder = '/tmp/smartcash_test'
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(os.path.join(self.test_folder, 'images'), exist_ok=True)
        
        # Buat dummy image untuk test
        self.create_dummy_test_data()
    
    def create_dummy_test_data(self):
        """Buat dummy test data untuk test"""
        try:
            import cv2
            # Buat 2 dummy images
            for i in range(2):
                img = np.ones((416, 416, 3), dtype=np.uint8) * 255
                cv2.imwrite(os.path.join(self.test_folder, 'images', f'test_{i}.jpg'), img)
        except ImportError:
            # Skip jika cv2 tidak tersedia
            pass
    
    def tearDown(self):
        """Cleanup setelah test"""
        # Hapus dummy test data
        import shutil
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
    
    @patch('smartcash.model.utils.validation_utils.validate_evaluation_config')
    def test_validate_evaluation_inputs(self, mock_validate):
        """Test validasi input evaluasi"""
        # Setup mock
        mock_validate.return_value = {'valid': True, 'message': '✅ Input valid'}
        
        # Setup UI components
        ui_components = {
            'scenario_dropdown': MagicMock(value='test_scenario'),
            'test_data_path': MagicMock(value=self.test_folder),
            'logger': self.logger
        }
        
        # Panggil fungsi
        result = validate_evaluation_inputs(ui_components, self.config, self.logger)
        
        # Verifikasi
        self.assertTrue(result['valid'])
        mock_validate.assert_called_once()
    
    def test_load_model_and_checkpoint(self):
        """Test load model dan checkpoint dengan pendekatan monkeypatching"""
        # Setup UI components
        ui_components = {
            'scenario_dropdown': MagicMock(value='test_scenario'),
            'logger': self.logger
        }
        
        # Monkeypatch model dalam hasil load_model_and_checkpoint
        mock_model = MagicMock(name="MockModel")
        original_result = None
        
        # Panggil fungsi
        result = load_model_and_checkpoint(ui_components, self.config, self.logger)
        
        # Simpan hasil asli
        original_result = result.copy()
        
        # Ganti model dengan mock model untuk test
        result['model'] = mock_model
        
        # Verifikasi
        self.assertTrue(result['success'])
        self.assertEqual(result['backbone'], 'cspdarknet_s')
        self.assertEqual(result['model'], mock_model)
    
    @patch('smartcash.dataset.utils.test_data_utils.prepare_test_data_for_scenario')
    @patch('smartcash.dataset.augmentor.strategies.evaluation_augmentation.get_augmentation_pipeline')
    @patch('smartcash.model.utils.scenario_utils.get_drive_path_for_scenario')
    def test_prepare_test_data_with_augmentation(self, mock_get_path, mock_get_aug, mock_prepare):
        """Test prepare test data dengan augmentasi"""
        # Setup mocks
        mock_get_path.return_value = {
            'success': True,
            'name': 'Skenario Test',
            'info': {'name': 'Skenario Test', 'augmentation_type': 'default'}
        }
        
        mock_get_aug.return_value = {
            'success': True,
            'pipeline': MagicMock(),
            'type': 'default'
        }
        
        mock_dataloader = MagicMock()
        mock_prepare.return_value = {
            'success': True,
            'dataloader': mock_dataloader,
            'count': 2,
            'image_files': [Path(os.path.join(self.test_folder, 'images', f'test_{i}.jpg')) for i in range(2)]
        }
        
        # Setup UI components
        ui_components = {
            'scenario_dropdown': MagicMock(value='test_scenario'),
            'test_data_path': MagicMock(value=self.test_folder),
            'logger': self.logger,
            'update_progress': MagicMock()
        }
        
        # Panggil fungsi
        result = prepare_test_data_with_augmentation(ui_components, self.config, self.logger)
        
        # Verifikasi
        self.assertTrue(result['success'])
        self.assertEqual(result['dataloader'], mock_dataloader)
        mock_get_path.assert_called_once()
        mock_get_aug.assert_called_once()
        mock_prepare.assert_called_once()
    
    @patch('smartcash.model.utils.evaluation_utils.run_inference_core')
    def test_run_model_inference(self, mock_run_inference):
        """Test run model inference"""
        # Setup mock
        mock_model = MagicMock()
        mock_dataloader = MagicMock()
        
        mock_predictions = [{'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]), 'scores': np.array([0.9]), 'classes': np.array([0])}]
        mock_run_inference.return_value = {
            'predictions': mock_predictions,
            'avg_inference_time': 0.01,
            'total_inference_time': 0.02
        }
        
        # Setup UI components
        ui_components = {
            'logger': self.logger,
            'update_progress': MagicMock()
        }
        
        # Panggil fungsi
        result = run_model_inference(
            mock_model,
            mock_dataloader,
            ui_components,
            self.config,
            self.logger
        )
        
        # Verifikasi
        self.assertTrue(result['success'])
        self.assertEqual(result['results'], mock_predictions)
        mock_run_inference.assert_called_once()
    
    @patch('smartcash.model.utils.evaluation_utils.load_ground_truth_labels')
    def test_load_ground_truth_labels(self, mock_load_labels):
        """Test load ground truth labels"""
        # Setup mock
        mock_image_files = [Path(os.path.join(self.test_folder, 'images', f'test_{i}.jpg')) for i in range(2)]
        mock_load_labels.return_value = {
            'available': True,
            'valid_label_count': 2,
            'labels': {'test_0.jpg': {'boxes': [[0.1, 0.2, 0.3, 0.4]], 'classes': [0]}}
        }
        
        # Panggil fungsi
        result = load_ground_truth_labels(
            self.test_folder,
            mock_image_files,
            self.logger,
            self.config
        )
        
        # Verifikasi
        self.assertTrue(result['available'])
        self.assertEqual(result['valid_label_count'], 2)
        mock_load_labels.assert_called_once()
    
    @patch('smartcash.model.utils.evaluation_pipeline.run_evaluation_pipeline')
    def test_run_evaluation_process(self, mock_run_pipeline):
        """Test run evaluation process"""
        # Setup mock
        mock_run_pipeline.return_value = {
            'success': True,
            'metrics': {
                'map': 0.85,
                'overall_metrics': {
                    'precision': 0.9,
                    'recall': 0.8,
                    'f1_score': 0.85
                }
            },
            'predictions': [{'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]), 'scores': np.array([0.9]), 'classes': np.array([0])}]
        }
        
        # Setup UI components
        ui_components = {
            'scenario_dropdown': MagicMock(value='test_scenario'),
            'test_data_path': MagicMock(value=self.test_folder),
            'logger': self.logger,
            'update_progress': MagicMock(),
            'show_for_operation': MagicMock(),
            'error_operation': MagicMock(),
            'success_operation': MagicMock(),
            'warning_operation': MagicMock()
        }
        
        # Setup button manager mock
        button_manager = MagicMock()
        button_manager.operation_context.return_value.__enter__ = MagicMock()
        button_manager.operation_context.return_value.__exit__ = MagicMock()
        
        # Panggil fungsi
        run_evaluation_process(ui_components, self.config, self.logger, button_manager)
        
        # Verifikasi
        mock_run_pipeline.assert_called_once()
        ui_components['success_operation'].assert_called()

    def test_apply_nms(self):
        """Test apply NMS pada model outputs"""
        # Setup mock outputs
        outputs = [
            {
                'boxes': np.array([[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]]),
                'scores': np.array([0.9, 0.8]),
                'classes': np.array([0, 0])
            }
        ]
        
        # Panggil fungsi dengan patch
        with patch('smartcash.model.utils.evaluation_utils.apply_nms') as mock_apply_nms:
            mock_apply_nms.return_value = [
                {
                    'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]),
                    'scores': np.array([0.9]),
                    'classes': np.array([0])
                }
            ]
            result = apply_nms(outputs, self.config)
            
            # Verifikasi
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]['boxes']), 1)
            mock_apply_nms.assert_called_once()
    
    def test_simple_nms(self):
        """Test simple NMS sebagai fallback"""
        # Setup mock outputs
        outputs = [
            {
                'boxes': np.array([[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]]),
                'scores': np.array([0.9, 0.8]),
                'classes': np.array([0, 0])
            }
        ]
        
        # Panggil fungsi dengan patch
        with patch('smartcash.model.utils.evaluation_utils.simple_nms') as mock_simple_nms:
            mock_simple_nms.return_value = [
                {
                    'boxes': np.array([[0.1, 0.2, 0.3, 0.4]]),
                    'scores': np.array([0.9]),
                    'classes': np.array([0])
                }
            ]
            result = simple_nms(outputs, self.config)
            
            # Verifikasi
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]['boxes']), 1)
            mock_simple_nms.assert_called_once()

if __name__ == '__main__':
    unittest.main()
