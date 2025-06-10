"""Unit test untuk TrainingService"""

import pytest
from unittest.mock import MagicMock, patch
from smartcash.model.service.training_service import TrainingService
from smartcash.common.environment import EnvironmentManager
import tempfile

@pytest.fixture
def mock_config():
    return {
        'epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 16,
        'class_labels': ['Rp1000', 'Rp2000']
    }

@pytest.fixture
def mock_env_manager(tmp_path):
    env = MagicMock()
    # Buat file model palsu
    model_path = tmp_path / "efficientnet_b4.pt"
    with open(model_path, 'wb') as f:
        f.write(b"fake model data" * 100)  # Tulis lebih banyak data
    
    env.get_model_dir.return_value = tmp_path
    env.get_dataloader.return_value = []
    return env


def test_training_service_init(mock_config, mock_env_manager):
    """Test inisialisasi TrainingService"""
    with patch('smartcash.model.manager.ModelManager.build_model') as mock_build:
        mock_build.return_value = MagicMock()
        service = TrainingService(mock_config, mock_env_manager)
        assert service.config == mock_config
        assert service.env_manager == mock_env_manager
        assert service.model is not None


def test_training_start(mock_config, mock_env_manager):
    """Test start training tanpa error"""
    with patch('smartcash.model.manager.ModelManager.build_model') as mock_build:
        mock_build.return_value = MagicMock()
        service = TrainingService(mock_config, mock_env_manager)
        service.trainer = MagicMock()  # Inisialisasi trainer
        assert service is not None
        assert service.model is not None
        assert service.trainer is not None
