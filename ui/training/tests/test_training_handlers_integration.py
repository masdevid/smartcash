"""
File: smartcash/ui/training/tests/test_training_handlers_integration.py
Deskripsi: Pengujian integrasi untuk handlers training yang terintegrasi dengan TrainingServiceManager
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from typing import Dict, Any

# Import modul yang akan diuji
from smartcash.ui.training.handlers.start_training_handler import handle_start_training
from smartcash.ui.training.handlers.stop_training_handler import handle_stop_training
from smartcash.ui.training.handlers.reset_training_handler import handle_reset_training, _initialize_empty_chart
from smartcash.ui.training.handlers.validation_handler import validate_model_before_training
from smartcash.ui.training.services.training_service_manager import TrainingServiceManager


class TestTrainingHandlersIntegration(unittest.TestCase):
    """Pengujian integrasi untuk handlers training dengan TrainingServiceManager"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
            'training_service': MagicMock(),
            'model_manager': MagicMock(),
            'status_panel': MagicMock(),
            'metrics_chart': MagicMock(),
            'epoch_progress': MagicMock(),
            'batch_progress': MagicMock(),
            'validation_progress': MagicMock(),
            'logger': MagicMock(),
            'metrics_history': {'train_loss': [], 'val_loss': [], 'epochs': []},
            'stop_button': MagicMock()
        }
        
        # Mock config
        self.mock_config = {
            'training': {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }
        
        # Mock training manager
        self.mock_training_manager = MagicMock(spec=TrainingServiceManager)
        self.mock_training_manager.start_training = MagicMock()
        self.mock_training_manager.stop_training = MagicMock()
        self.mock_training_manager.reset_training_state = MagicMock()
        self.mock_training_manager.validate_training_readiness = MagicMock(return_value=True)
        
        # Tambahkan training manager ke UI components
        self.mock_ui_components['training_manager'] = self.mock_training_manager
    
    @patch('smartcash.ui.training.handlers.start_training_handler._execute_training_process')
    @patch('smartcash.ui.training.handlers.start_training_handler.set_state')
    def test_handle_start_training_with_manager(self, mock_set_state, mock_execute):
        """Test handle_start_training menggunakan training manager jika tersedia"""
        # Panggil function yang diuji
        handle_start_training(self.mock_ui_components, self.mock_config)
        
        # Verifikasi state diubah
        mock_set_state.assert_called_once_with(active=True, stop_requested=False)
        
        # Verifikasi execute training process dipanggil dengan training service
        mock_execute.assert_called_once_with(
            self.mock_ui_components, 
            self.mock_ui_components['training_service'], 
            self.mock_config
        )
    
    def test_handle_start_training_fallback(self):
        """Test handle_start_training fallback ke training service jika manager tidak tersedia"""
        # Hapus training manager dari UI components
        ui_components_no_manager = self.mock_ui_components.copy()
        ui_components_no_manager.pop('training_manager')
        
        # Mock dataset
        mock_dataset = MagicMock()
        ui_components_no_manager['model_manager'].get_dataset = MagicMock(return_value=mock_dataset)
        
        # Panggil function yang diuji
        handle_start_training(ui_components_no_manager, self.mock_config)
        
        # Verifikasi training service digunakan langsung
        ui_components_no_manager['training_service'].start_training.assert_called_once()
    
    @patch('smartcash.ui.training.handlers.stop_training_handler.get_state')
    @patch('smartcash.ui.training.handlers.stop_training_handler.set_state')
    def test_handle_stop_training_with_manager(self, mock_set_state, mock_get_state):
        """Test handle_stop_training menggunakan training manager jika tersedia"""
        # Setup mock untuk get_state
        mock_get_state.return_value = {'active': True}
        
        # Panggil function yang diuji
        handle_stop_training(self.mock_ui_components)
        
        # Verifikasi state diubah
        mock_set_state.assert_called_once_with(stop_requested=True)
        
        # Verifikasi training manager digunakan
        self.mock_training_manager.stop_training.assert_called_once()
        # Verifikasi training service tidak digunakan langsung
        self.mock_ui_components['training_service'].stop_training.assert_not_called()
    
    @patch('smartcash.ui.training.handlers.stop_training_handler.get_state')
    @patch('smartcash.ui.training.handlers.stop_training_handler.set_state')
    def test_handle_stop_training_fallback(self, mock_set_state, mock_get_state):
        """Test handle_stop_training fallback ke training service jika manager tidak tersedia"""
        # Setup mock untuk get_state
        mock_get_state.return_value = {'active': True}
        
        # Hapus training manager dari UI components
        ui_components_no_manager = self.mock_ui_components.copy()
        ui_components_no_manager.pop('training_manager')
        
        # Panggil function yang diuji
        handle_stop_training(ui_components_no_manager)
        
        # Verifikasi state diubah
        mock_set_state.assert_called_once_with(stop_requested=True)
        
        # Verifikasi training service digunakan langsung
        ui_components_no_manager['training_service'].stop_training.assert_called_once()
    
    def test_handle_reset_training_with_manager(self):
        """Test handle_reset_training menggunakan training manager jika tersedia"""
        # Panggil function yang diuji
        handle_reset_training(self.mock_ui_components)
        
        # Verifikasi training manager digunakan
        self.mock_training_manager.reset_training_state.assert_called_once()
        # Verifikasi training service tidak digunakan langsung
        self.mock_ui_components['training_service'].reset_training_state.assert_not_called()
    
    def test_handle_reset_training_fallback(self):
        """Test handle_reset_training fallback ke training service jika manager tidak tersedia"""
        # Hapus training manager dari UI components
        ui_components_no_manager = self.mock_ui_components.copy()
        ui_components_no_manager.pop('training_manager')
        
        # Panggil function yang diuji
        handle_reset_training(ui_components_no_manager)
        
        # Verifikasi training service digunakan langsung
        ui_components_no_manager['training_service'].reset_training_state.assert_called_once()
    
    @patch('smartcash.ui.training.utils.training_chart_utils.initialize_empty_training_chart')
    def test_initialize_empty_chart(self, mock_initialize_chart):
        """Test _initialize_empty_chart menggunakan fungsi dari training_chart_utils"""
        # Panggil function yang diuji
        _initialize_empty_chart(self.mock_ui_components)
        
        # Verifikasi initialize_empty_training_chart dipanggil
        mock_initialize_chart.assert_called_once_with(self.mock_ui_components)
        
        # Verifikasi metrics history direset
        self.assertEqual(self.mock_ui_components['metrics_history'], 
                         {'train_loss': [], 'val_loss': [], 'epochs': []})
        
        # Verifikasi progress bars direset
        for progress_key in ['epoch_progress', 'batch_progress', 'validation_progress']:
            progress_bar = self.mock_ui_components[progress_key]
            progress_bar.value = 0
    
    def test_validate_model_before_training_with_manager(self):
        """Test validate_model_before_training menggunakan training manager"""
        # Panggil function yang diuji
        result = validate_model_before_training(self.mock_ui_components, self.mock_config)
        
        # Verifikasi training manager digunakan
        self.mock_training_manager.validate_training_readiness.assert_called_once()
        # Verifikasi hasil sesuai dengan hasil dari training manager
        self.assertTrue(result)
    
    def test_validate_model_before_training_create_manager(self):
        """Test validate_model_before_training membuat training manager jika belum ada"""
        # Hapus training manager dari UI components
        ui_components_no_manager = self.mock_ui_components.copy()
        ui_components_no_manager.pop('training_manager')
        
        # Mock TrainingServiceManager
        with patch('smartcash.ui.training.services.TrainingServiceManager') as mock_manager_class:
            # Setup mock untuk manager yang dibuat
            mock_manager = MagicMock()
            mock_manager.validate_training_readiness = MagicMock(return_value=True)
            mock_manager_class.return_value = mock_manager
            
            # Panggil function yang diuji
            result = validate_model_before_training(ui_components_no_manager, self.mock_config)
            
            # Verifikasi TrainingServiceManager dibuat
            mock_manager_class.assert_called_once_with(ui_components_no_manager, self.mock_config)
            # Verifikasi training manager ditambahkan ke UI components
            self.assertEqual(ui_components_no_manager['training_manager'], mock_manager)
            # Verifikasi validate_training_readiness dipanggil
            mock_manager.validate_training_readiness.assert_called_once()
            # Verifikasi hasil sesuai dengan hasil dari training manager
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
