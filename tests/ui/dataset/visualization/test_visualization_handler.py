"""
File: tests/ui/dataset/visualization/test_visualization_handler.py
Deskripsi: Unit test untuk visualization_handler
"""

import pytest
from unittest.mock import patch, MagicMock
from smartcash.ui.dataset.visualization.handlers.visualization_handler import DatasetVisualizationHandler

@patch('smartcash.dataset.preprocessor.list_available_datasets')
@patch('smartcash.dataset.preprocessor.get_preprocessing_stats')
def test_load_dataset_success(mock_get_stats, mock_list_datasets):
    """Test pemuatan dataset berhasil"""
    # Setup
    mock_list_datasets.return_value = ['dataset1', 'dataset2']
    mock_get_stats.return_value = {'total_images': 100}
    
    handler = DatasetVisualizationHandler()
    result = handler.load_dataset('dataset1')
    
    assert result is True
    assert handler.current_dataset == 'dataset1'
    assert handler.preprocessing_stats == {'total_images': 100}

@patch('smartcash.dataset.preprocessor.list_available_datasets')
def test_load_dataset_not_found(mock_list_datasets):
    """Test pemuatan dataset yang tidak ditemukan"""
    mock_list_datasets.return_value = ['dataset1', 'dataset2']
    
    handler = DatasetVisualizationHandler()
    result = handler.load_dataset('dataset3')
    
    assert result is False

@patch('smartcash.dataset.preprocessor.list_available_datasets')
@patch('smartcash.dataset.preprocessor.get_preprocessing_stats')
def test_load_dataset_exception(mock_get_stats, mock_list_datasets):
    """Test exception saat memuat dataset"""
    mock_list_datasets.return_value = ['dataset1']
    mock_get_stats.side_effect = Exception("Test error")
    
    handler = DatasetVisualizationHandler()
    result = handler.load_dataset('dataset1')
    
    assert result is False
    assert handler.preprocessing_stats == {}
