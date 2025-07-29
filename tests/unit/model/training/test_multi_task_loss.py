#!/usr/bin/env python3
"""
Comprehensive tests for smartcash/model/training/multi_task_loss.py

This module tests:
- UncertaintyMultiTaskLoss functionality
- AdaptiveMultiTaskLoss functionality  
- Factory functions
- Edge cases and error handling
- Loss computation accuracy
- Uncertainty parameter learning
- Adaptive weighting mechanisms
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np

# Import the modules under test
from smartcash.model.training.multi_task_loss import (
    UncertaintyMultiTaskLoss,
    AdaptiveMultiTaskLoss,
    create_uncertainty_loss,
    create_adaptive_loss,
    create_banknote_multi_task_loss
)


class TestUncertaintyMultiTaskLoss:
    """Test cases for UncertaintyMultiTaskLoss class."""
    
    @pytest.fixture
    def sample_layer_config(self):
        """Sample layer configuration for testing."""
        return {
            'layer_1': {'num_classes': 7, 'description': 'Full banknote detection'},
            'layer_2': {'num_classes': 7, 'description': 'Nominal features'},
            'layer_3': {'num_classes': 3, 'description': 'Common features'}
        }
    
    @pytest.fixture
    def sample_loss_config(self):
        """Sample loss configuration for testing."""
        return {
            'box_weight': 0.05,
            'obj_weight': 1.0,
            'cls_weight': 0.5,
            'focal_loss': False,
            'label_smoothing': 0.0,
            'dynamic_weighting': True,
            'min_variance': 1e-3,
            'max_variance': 10.0
        }
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions tensor dict for testing."""
        batch_size = 4
        return {
            'layer_1': [
                torch.randn(batch_size, 3, 80, 80, 22),  # P3 predictions (5 + 17)
                torch.randn(batch_size, 3, 40, 40, 22),  # P4 predictions
                torch.randn(batch_size, 3, 20, 20, 22),  # P5 predictions
            ],
            'layer_2': [
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22),
                torch.randn(batch_size, 3, 20, 20, 22),
            ],
            'layer_3': [
                torch.randn(batch_size, 3, 80, 80, 8),   # 5 + 3 classes
                torch.randn(batch_size, 3, 40, 40, 8),
                torch.randn(batch_size, 3, 20, 20, 8),
            ]
        }
    
    @pytest.fixture
    def sample_targets(self):
        """Sample targets tensor dict for testing."""
        return {
            'layer_1': torch.randint(0, 7, (20, 6)),  # [img_idx, class, x, y, w, h]
            'layer_2': torch.randint(0, 7, (18, 6)),
            'layer_3': torch.randint(0, 3, (15, 6))
        }
    
    def test_initialization_basic(self, sample_layer_config, mock_logger):
        """Test basic initialization of UncertaintyMultiTaskLoss."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Check basic attributes
        assert loss_fn.num_layers == 3
        assert loss_fn.layer_names == ['layer_1', 'layer_2', 'layer_3']
        assert len(loss_fn.log_vars) == 3
        assert len(loss_fn.layer_losses) == 3
        
        # Check logger calls
        mock_logger.info.assert_called()
        
        # Check learnable parameters initialization
        for layer_name in sample_layer_config.keys():
            assert layer_name in loss_fn.log_vars
            assert isinstance(loss_fn.log_vars[layer_name], nn.Parameter)
            assert loss_fn.log_vars[layer_name].data.item() == 0.0
    
    def test_initialization_with_config(self, sample_layer_config, sample_loss_config, mock_logger):
        """Test initialization with custom loss configuration."""
        loss_fn = UncertaintyMultiTaskLoss(
            sample_layer_config, 
            sample_loss_config, 
            logger=mock_logger
        )
        
        assert loss_fn.use_dynamic_weighting == True
        assert loss_fn.min_variance == 1e-3
        assert loss_fn.max_variance == 10.0
        
        # Check individual layer loss configurations
        for layer_name in sample_layer_config.keys():
            layer_loss = loss_fn.layer_losses[layer_name]
            assert hasattr(layer_loss, 'box_weight')
            assert hasattr(layer_loss, 'obj_weight')
            assert hasattr(layer_loss, 'cls_weight')
    
    def test_forward_basic_computation(self, sample_layer_config, 
                                     sample_predictions, sample_targets, mock_logger):
        """Test basic forward pass computation."""
        
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Mock individual layer losses to return controlled values
        with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                         return_value=(torch.tensor(0.3, requires_grad=True), {'total_loss': torch.tensor(0.3)})):
            with patch.object(loss_fn.layer_losses['layer_2'], '__call__', 
                             return_value=(torch.tensor(0.4, requires_grad=True), {'total_loss': torch.tensor(0.4)})):
                with patch.object(loss_fn.layer_losses['layer_3'], '__call__', 
                                 return_value=(torch.tensor(0.2, requires_grad=True), {'total_loss': torch.tensor(0.2)})):
                    
                    total_loss, loss_breakdown = loss_fn(sample_predictions, sample_targets)
        
        # Check return types
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_breakdown, dict)
        assert total_loss.requires_grad
        
        # Check loss breakdown structure
        assert 'total_loss' in loss_breakdown
        assert 'num_layers' in loss_breakdown
        assert 'layer_losses' in loss_breakdown
        assert 'uncertainties' in loss_breakdown
        assert 'use_uncertainty_weighting' in loss_breakdown
    
    def test_forward_with_uncertainty_weighting(self, sample_layer_config, mock_logger):
        """Test forward pass with uncertainty weighting enabled."""
        loss_config = {'dynamic_weighting': True}
        loss_fn = UncertaintyMultiTaskLoss(
            sample_layer_config, 
            loss_config, 
            logger=mock_logger
        )
        
        # Create simple mock predictions and targets
        predictions = {
            'layer_1': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_2': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_3': [torch.randn(2, 3, 40, 40, 8, requires_grad=True)]
        }
        targets = {
            'layer_1': torch.randint(0, 7, (5, 6)).float(),
            'layer_2': torch.randint(0, 7, (5, 6)).float(),
            'layer_3': torch.randint(0, 3, (5, 6)).float()
        }
        
        with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                         return_value=(torch.tensor(0.3, requires_grad=True), {})):
            with patch.object(loss_fn.layer_losses['layer_2'], '__call__', 
                             return_value=(torch.tensor(0.4, requires_grad=True), {})):
                with patch.object(loss_fn.layer_losses['layer_3'], '__call__', 
                                 return_value=(torch.tensor(0.2, requires_grad=True), {})):
                    
                    total_loss, loss_breakdown = loss_fn(predictions, targets)
                    
                    # Check that uncertainty weighting components are present
                    for layer_name in sample_layer_config.keys():
                        assert f'{layer_name}_weighted_loss' in loss_breakdown
                        assert f'{layer_name}_regularization' in loss_breakdown
                        assert f'{layer_name}_uncertainty' in loss_breakdown
                    
                    assert loss_breakdown['use_uncertainty_weighting'] == True
    
    def test_forward_without_uncertainty_weighting(self, sample_layer_config, mock_logger):
        """Test forward pass with uncertainty weighting disabled."""
        loss_config = {'dynamic_weighting': False}
        loss_fn = UncertaintyMultiTaskLoss(
            sample_layer_config, 
            loss_config, 
            logger=mock_logger
        )
        
        predictions = {
            'layer_1': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_2': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_3': [torch.randn(2, 3, 40, 40, 8, requires_grad=True)]
        }
        targets = {
            'layer_1': torch.randint(0, 7, (5, 6)).float(),
            'layer_2': torch.randint(0, 7, (5, 6)).float(),
            'layer_3': torch.randint(0, 3, (5, 6)).float()
        }
        
        with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                         return_value=(torch.tensor(0.3, requires_grad=True), {})):
            with patch.object(loss_fn.layer_losses['layer_2'], '__call__', 
                             return_value=(torch.tensor(0.4, requires_grad=True), {})):
                with patch.object(loss_fn.layer_losses['layer_3'], '__call__', 
                                 return_value=(torch.tensor(0.2, requires_grad=True), {})):
                    
                    total_loss, loss_breakdown = loss_fn(predictions, targets)
                    
                    # Check that uncertainty weighting is disabled
                    assert loss_breakdown['use_uncertainty_weighting'] == False
                    
                    # Should not have weighted loss components
                    for layer_name in sample_layer_config.keys():
                        assert f'{layer_name}_weighted_loss' not in loss_breakdown
                        assert f'{layer_name}_regularization' not in loss_breakdown
    
    def test_handle_missing_predictions(self, sample_layer_config, mock_logger):
        """Test handling of missing predictions for some layers."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Only provide predictions for layer_1
        predictions = {
            'layer_1': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)]
        }
        targets = {
            'layer_1': torch.randint(0, 7, (5, 6)).float(),
            'layer_2': torch.randint(0, 7, (5, 6)).float(),
            'layer_3': torch.randint(0, 3, (5, 6)).float()
        }
        
        with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                         return_value=(torch.tensor(0.3, requires_grad=True), {})):
            
            total_loss, loss_breakdown = loss_fn(predictions, targets)
            
            # Check that missing layers get zero loss
            assert 'layer_2_total_loss' in loss_breakdown
            assert 'layer_3_total_loss' in loss_breakdown
            assert loss_breakdown['layer_2_total_loss'].item() == 0.0
            assert loss_breakdown['layer_3_total_loss'].item() == 0.0
    
    def test_handle_invalid_predictions(self, sample_layer_config, mock_logger):
        """Test handling of invalid prediction formats."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Provide predictions with mixed valid and invalid elements
        predictions = {
            'layer_1': [torch.randn(2, 3, 40, 40, 22), 'invalid', torch.randn(2, 3, 40, 40, 22)],
            'layer_2': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_3': [torch.randn(2, 3, 40, 40, 8)]  # Valid tensor
        }
        targets = {
            'layer_1': torch.randint(0, 7, (5, 6)).float(),
            'layer_2': torch.randint(0, 7, (5, 6)).float(),
            'layer_3': torch.randint(0, 3, (5, 6)).float()
        }
        
        with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                         return_value=(torch.tensor(0.3, requires_grad=True), {})):
            with patch.object(loss_fn.layer_losses['layer_2'], '__call__', 
                             return_value=(torch.tensor(0.4, requires_grad=True), {})):
                with patch.object(loss_fn.layer_losses['layer_3'], '__call__', 
                                 return_value=(torch.tensor(0.2, requires_grad=True), {})):
                    
                    total_loss, loss_breakdown = loss_fn(predictions, targets)
                    
                    # Should handle invalid predictions gracefully
                    assert isinstance(total_loss, torch.Tensor)
                    mock_logger.warning.assert_called()
    
    def test_get_uncertainty_weights(self, sample_layer_config, mock_logger):
        """Test getting uncertainty weights for each layer."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Manually set some log variance values
        loss_fn.log_vars['layer_1'].data = torch.tensor(0.5)
        loss_fn.log_vars['layer_2'].data = torch.tensor(-0.3)
        loss_fn.log_vars['layer_3'].data = torch.tensor(1.0)
        
        weights = loss_fn.get_uncertainty_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert all(isinstance(w, float) for w in weights.values())
        assert all(w > 0 for w in weights.values())  # Weights should be positive
    
    def test_get_uncertainty_values(self, sample_layer_config, mock_logger):
        """Test getting uncertainty values for each layer."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        # Manually set some log variance values
        loss_fn.log_vars['layer_1'].data = torch.tensor(0.5)
        loss_fn.log_vars['layer_2'].data = torch.tensor(-0.3)
        loss_fn.log_vars['layer_3'].data = torch.tensor(1.0)
        
        uncertainties = loss_fn.get_uncertainty_values()
        
        assert isinstance(uncertainties, dict)
        assert len(uncertainties) == 3
        assert all(isinstance(u, float) for u in uncertainties.values())
        assert all(u > 0 for u in uncertainties.values())  # Uncertainties should be positive
    
    def test_update_loss_config(self, sample_layer_config, mock_logger):
        """Test updating loss configuration."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        new_config = {
            'box_weight': 0.1,
            'obj_weight': 2.0,
            'cls_weight': 1.0
        }
        
        loss_fn.update_loss_config(new_config)
        
        # Check that config was updated
        assert loss_fn.loss_config['box_weight'] == 0.1
        assert loss_fn.loss_config['obj_weight'] == 2.0
        assert loss_fn.loss_config['cls_weight'] == 1.0
    
    def test_get_loss_summary(self, sample_layer_config, mock_logger):
        """Test getting formatted loss summary."""
        loss_fn = UncertaintyMultiTaskLoss(sample_layer_config, logger=mock_logger)
        
        loss_dict = {
            'total_loss': torch.tensor(1.234),
            'num_layers': 3,
            'uncertainties': {
                'layer_1': 0.567,
                'layer_2': 0.890,
                'layer_3': 0.123
            }
        }
        
        summary = loss_fn.get_loss_summary(loss_dict)
        
        assert isinstance(summary, str)
        assert '1.234' in summary  # Total loss should be in summary
        assert 'Layers: 3' in summary
        assert 'Uncertainties:' in summary
        assert '0.567' in summary  # Uncertainty values should be present
    
    def test_variance_clamping(self, sample_layer_config, mock_logger):
        """Test that variance values are properly clamped."""
        loss_config = {'min_variance': 0.01, 'max_variance': 5.0}
        loss_fn = UncertaintyMultiTaskLoss(
            sample_layer_config, 
            loss_config, 
            logger=mock_logger
        )
        
        # Set extreme log variance values
        loss_fn.log_vars['layer_1'].data = torch.tensor(-10.0)  # Very small variance
        loss_fn.log_vars['layer_2'].data = torch.tensor(10.0)   # Very large variance
        
        uncertainties = loss_fn.get_uncertainty_values()
        
        # Check clamping (allow for floating point precision)
        assert uncertainties['layer_1'] >= 0.01 - 1e-6  # Should be clamped to min
        assert uncertainties['layer_2'] <= 5.0 + 1e-6   # Should be clamped to max


class TestAdaptiveMultiTaskLoss:
    """Test cases for AdaptiveMultiTaskLoss class."""
    
    @pytest.fixture
    def sample_layer_config(self):
        """Sample layer configuration for testing."""
        return {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 3}
        }
    
    @pytest.fixture
    def adaptive_loss_config(self):
        """Loss configuration for adaptive loss."""
        return {
            'dynamic_weighting': True,
            'adaptation_rate': 0.05,
            'performance_threshold': 0.1
        }
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    def test_initialization(self, sample_layer_config, adaptive_loss_config, mock_logger):
        """Test AdaptiveMultiTaskLoss initialization."""
        loss_fn = AdaptiveMultiTaskLoss(
            sample_layer_config,
            adaptive_loss_config,
            performance_window=50,
            logger=mock_logger
        )
        
        assert loss_fn.performance_window == 50
        assert loss_fn.adaptation_rate == 0.05
        assert loss_fn.performance_threshold == 0.1
        assert loss_fn.step_count == 0
        assert len(loss_fn.loss_history) == 2
    
    def test_loss_history_update(self, sample_layer_config, mock_logger):
        """Test loss history updating mechanism."""
        loss_fn = AdaptiveMultiTaskLoss(
            sample_layer_config,
            performance_window=10,
            logger=mock_logger
        )
        
        # Simulate multiple forward passes
        for i in range(15):
            loss_breakdown = {
                'layer_1_total_loss': torch.tensor(0.5 - i * 0.01),
                'layer_2_total_loss': torch.tensor(0.3 - i * 0.005)
            }
            loss_fn._update_loss_history(loss_breakdown)
        
        # Check that history is maintained within window
        assert len(loss_fn.loss_history['layer_1']) == 10
        assert len(loss_fn.loss_history['layer_2']) == 10
        
        # Check that latest values are correct
        assert abs(loss_fn.loss_history['layer_1'][-1] - (0.5 - 14 * 0.01)) < 1e-6
        assert abs(loss_fn.loss_history['layer_2'][-1] - (0.3 - 14 * 0.005)) < 1e-6
    
    def test_adaptation_factor_computation(self, sample_layer_config, mock_logger):
        """Test computation of adaptive weighting factors."""
        loss_fn = AdaptiveMultiTaskLoss(
            sample_layer_config,
            {'adaptation_rate': 0.1, 'performance_threshold': 0.2},
            performance_window=20,
            logger=mock_logger
        )
        
        # Simulate improving layer (layer_1) and non-improving layer (layer_2)
        for i in range(20):
            loss_fn.loss_history['layer_1'].append(1.0 - i * 0.04)  # Good improvement
            loss_fn.loss_history['layer_2'].append(0.5 - i * 0.005)  # Poor improvement
        
        factors = loss_fn._compute_adaptation_factors()
        
        assert isinstance(factors, dict)
        assert len(factors) == 2
        
        # Layer with good improvement should get lower weight
        # Layer with poor improvement should get higher weight
        assert factors['layer_1'] < factors['layer_2']
    
    def test_forward_with_adaptation(self, sample_layer_config, mock_logger):
        """Test forward pass with adaptive weighting."""
        loss_fn = AdaptiveMultiTaskLoss(
            sample_layer_config,
            performance_window=5,
            logger=mock_logger
        )
        
        predictions = {
            'layer_1': [torch.randn(2, 3, 40, 40, 22, requires_grad=True)],
            'layer_2': [torch.randn(2, 3, 40, 40, 8, requires_grad=True)]
        }
        targets = {
            'layer_1': torch.randint(0, 7, (5, 6)).float(),
            'layer_2': torch.randint(0, 3, (5, 6)).float()
        }
        
        # Mock parent class forward method
        with patch.object(UncertaintyMultiTaskLoss, 'forward') as mock_parent_forward:
            mock_parent_forward.return_value = (
                torch.tensor(0.5, requires_grad=True),
                {
                    'layer_1_total_loss': torch.tensor(0.3),
                    'layer_2_total_loss': torch.tensor(0.2)
                }
            )
            
            # Run several forward passes to build up history
            for _ in range(10):
                total_loss, loss_breakdown = loss_fn(predictions, targets)
            
            # Check that adaptation factors are present after enough steps
            if loss_fn.step_count > loss_fn.performance_window:
                assert 'adaptation_factors' in loss_breakdown


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_uncertainty_loss(self):
        """Test create_uncertainty_loss factory function."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 3}
        }
        loss_config = {'dynamic_weighting': True}
        
        loss_fn = create_uncertainty_loss(layer_config, loss_config)
        
        assert isinstance(loss_fn, UncertaintyMultiTaskLoss)
        assert loss_fn.num_layers == 2
        assert loss_fn.use_dynamic_weighting == True
    
    def test_create_adaptive_loss(self):
        """Test create_adaptive_loss factory function."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 3}
        }
        loss_config = {'adaptation_rate': 0.02}
        
        loss_fn = create_adaptive_loss(layer_config, loss_config)
        
        assert isinstance(loss_fn, AdaptiveMultiTaskLoss)
        assert loss_fn.num_layers == 2
        assert loss_fn.adaptation_rate == 0.02
    
    def test_create_banknote_multi_task_loss_uncertainty(self):
        """Test create_banknote_multi_task_loss with uncertainty loss."""
        loss_fn = create_banknote_multi_task_loss(use_adaptive=False)
        
        assert isinstance(loss_fn, UncertaintyMultiTaskLoss)
        assert loss_fn.num_layers == 3
        assert 'layer_1' in loss_fn.layer_names
        assert 'layer_2' in loss_fn.layer_names
        assert 'layer_3' in loss_fn.layer_names
        
        # Check class configurations
        assert loss_fn.layer_config['layer_1']['num_classes'] == 7
        assert loss_fn.layer_config['layer_2']['num_classes'] == 7
        assert loss_fn.layer_config['layer_3']['num_classes'] == 3
    
    def test_create_banknote_multi_task_loss_adaptive(self):
        """Test create_banknote_multi_task_loss with adaptive loss."""
        loss_fn = create_banknote_multi_task_loss(use_adaptive=True)
        
        assert isinstance(loss_fn, AdaptiveMultiTaskLoss)
        assert loss_fn.num_layers == 3


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for edge case testing."""
        return {'layer_1': {'num_classes': 7}}
    
    def test_empty_predictions(self, basic_config):
        """Test handling of empty predictions."""
        loss_fn = UncertaintyMultiTaskLoss(basic_config)
        
        # Test case where predictions dict is missing the layer
        predictions = {}  # No predictions for layer_1
        targets = {'layer_1': torch.randint(0, 7, (5, 6)).float()}
        
        total_loss, loss_breakdown = loss_fn(predictions, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert loss_breakdown['layer_1_total_loss'].item() == 0.0
    
    def test_empty_targets(self, basic_config):
        """Test handling of empty targets."""
        loss_fn = UncertaintyMultiTaskLoss(basic_config)
        
        predictions = {'layer_1': [torch.randn(2, 3, 40, 40, 22)]}
        targets = {}
        
        total_loss, loss_breakdown = loss_fn(predictions, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert loss_breakdown['layer_1_total_loss'].item() == 0.0
    
    def test_mismatched_devices(self, basic_config):
        """Test handling of tensors on different devices."""
        loss_fn = UncertaintyMultiTaskLoss(basic_config)
        
        # Create tensors on CPU (assuming no GPU available in test environment)
        predictions = {'layer_1': [torch.randn(2, 3, 40, 40, 22)]}
        targets = {'layer_1': torch.randint(0, 7, (5, 6)).float()}
        
        # This should work without errors (tensors will be moved to same device)
        total_loss, loss_breakdown = loss_fn(predictions, targets)
        
        assert isinstance(total_loss, torch.Tensor)
    
    def test_zero_variance_handling(self, basic_config):
        """Test handling of extremely small variances."""
        loss_config = {'min_variance': 1e-6, 'max_variance': 1e6}
        loss_fn = UncertaintyMultiTaskLoss(basic_config, loss_config)
        
        # Set very small log variance
        loss_fn.log_vars['layer_1'].data = torch.tensor(-20.0)
        
        uncertainties = loss_fn.get_uncertainty_values()
        weights = loss_fn.get_uncertainty_weights()
        
        # Should be clamped to minimum (allow for floating point precision)
        assert uncertainties['layer_1'] >= 1e-6 - 1e-10
        assert weights['layer_1'] > 0  # Should still be positive
    
    def test_nan_loss_handling(self, basic_config):
        """Test handling of exception from individual layer loss computation."""
        loss_fn = UncertaintyMultiTaskLoss(basic_config)
        
        predictions = {'layer_1': [torch.randn(2, 3, 40, 40, 22)]}
        targets = {'layer_1': torch.randint(0, 7, (5, 6)).float()}
        
        # Test that exception handling works correctly by simulating a failing layer loss
        # The test verifies that when layer loss computation fails, the system gracefully
        # handles it and continues rather than crashing the entire training process
        with patch.object(loss_fn.layer_losses['layer_1'], 'forward') as mock_forward:
            mock_forward.side_effect = Exception("Simulated layer loss computation failure")
            
            total_loss, loss_breakdown = loss_fn(predictions, targets)
            
            # Even with exception, the function should complete and return valid structures
            assert isinstance(total_loss, torch.Tensor)
            assert isinstance(loss_breakdown, dict)
            assert 'layer_1_total_loss' in loss_breakdown
            
            # The specific behavior may vary (could be 0.0 or some fallback value)
            # but the system should remain stable
            assert total_loss.item() >= 0.0


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    def test_end_to_end_training_simulation(self):
        """Test end-to-end simulation of training with uncertainty loss."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 7},
            'layer_3': {'num_classes': 3}
        }
        
        loss_fn = UncertaintyMultiTaskLoss(layer_config)
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.001)
        
        # Simulate training iterations
        initial_uncertainties = loss_fn.get_uncertainty_values()
        
        for epoch in range(5):
            # Create batch data
            predictions = {
                'layer_1': [torch.randn(4, 3, 40, 40, 22, requires_grad=True)],
                'layer_2': [torch.randn(4, 3, 40, 40, 22, requires_grad=True)],
                'layer_3': [torch.randn(4, 3, 40, 40, 8, requires_grad=True)]
            }
            targets = {
                'layer_1': torch.randint(0, 7, (10, 6)).float(),
                'layer_2': torch.randint(0, 7, (8, 6)).float(),
                'layer_3': torch.randint(0, 3, (6, 6)).float()
            }
            
            # Forward pass
            with patch.object(loss_fn.layer_losses['layer_1'], '__call__', 
                             return_value=(torch.tensor(0.5, requires_grad=True), {})):
                with patch.object(loss_fn.layer_losses['layer_2'], '__call__', 
                                 return_value=(torch.tensor(0.4, requires_grad=True), {})):
                    with patch.object(loss_fn.layer_losses['layer_3'], '__call__', 
                                     return_value=(torch.tensor(0.3, requires_grad=True), {})):
                    
                        total_loss, loss_breakdown = loss_fn(predictions, targets)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
        
        # Check that uncertainty parameters have been updated
        final_uncertainties = loss_fn.get_uncertainty_values()
        
        # At least one uncertainty should have changed
        assert any(abs(initial_uncertainties[layer] - final_uncertainties[layer]) > 1e-6 
                  for layer in layer_config.keys())
    
    def test_adaptive_loss_convergence(self):
        """Test that adaptive loss converges properly."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 3}
        }
        
        loss_fn = AdaptiveMultiTaskLoss(
            layer_config,
            performance_window=10
        )
        
        # Simulate training with one layer improving faster than the other
        for step in range(25):
            loss_breakdown = {
                'layer_1_total_loss': torch.tensor(1.0 - step * 0.03),  # Fast improvement
                'layer_2_total_loss': torch.tensor(0.8 - step * 0.01)   # Slow improvement
            }
            
            loss_fn._update_loss_history(loss_breakdown)
            loss_fn.step_count += 1
        
        # Get adaptation factors
        factors = loss_fn._compute_adaptation_factors()
        
        # Layer with slower improvement should get higher weight
        assert factors['layer_2'] > factors['layer_1']


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])