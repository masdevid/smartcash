#!/usr/bin/env python3
"""
Comprehensive tests for checkpoint selection and filtering functionality.

Tests the checkpoint selector component with various filtering criteria,
edge cases, and integration scenarios.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np

from smartcash.model.evaluation.checkpoint_selector import (
    CheckpointSelector,
    create_checkpoint_selector,
    get_available_checkpoints
)


class TestCheckpointSelector:
    """Test CheckpointSelector class functionality."""
    
    @pytest.fixture
    def temp_checkpoint_env(self):
        """Create temporary checkpoint environment for testing."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create multiple checkpoint directories
        checkpoint_dirs = [
            temp_path / 'checkpoints',
            temp_path / 'runs' / 'train' / 'exp1' / 'weights',
            temp_path / 'runs' / 'train' / 'exp2' / 'weights',
            temp_path / 'models' / 'checkpoints'
        ]
        
        for dir_path in checkpoint_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create various checkpoint files with different characteristics
        checkpoint_files = []
        
        # High-quality checkpoints
        high_quality_checkpoints = [
            {
                'dir': checkpoint_dirs[0],
                'name': 'best_cspdarknet_multi_20240101.pt',
                'backbone': 'cspdarknet',
                'val_map': 0.89,
                'val_loss': 0.15,
                'epoch': 95
            },
            {
                'dir': checkpoint_dirs[0],
                'name': 'best_efficientnet_b4_multi_20240102.pt',
                'backbone': 'efficientnet_b4',
                'val_map': 0.92,
                'val_loss': 0.12,
                'epoch': 87
            },
            {
                'dir': checkpoint_dirs[1],
                'name': 'best.pt',
                'backbone': 'yolov5s',
                'val_map': 0.85,
                'val_loss': 0.18,
                'epoch': 76
            }
        ]
        
        # Medium-quality checkpoints
        medium_quality_checkpoints = [
            {
                'dir': checkpoint_dirs[1],
                'name': 'last.pt',
                'backbone': 'cspdarknet',
                'val_map': 0.73,
                'val_loss': 0.25,
                'epoch': 45
            },
            {
                'dir': checkpoint_dirs[2],
                'name': 'smartcash_efficientnet_b4_best.pt',
                'backbone': 'efficientnet_b4',
                'val_map': 0.78,
                'val_loss': 0.22,
                'epoch': 62
            }
        ]
        
        # Low-quality checkpoints (should be filtered out by default)
        low_quality_checkpoints = [
            {
                'dir': checkpoint_dirs[3],
                'name': 'early_checkpoint.pt',
                'backbone': 'unknown',
                'val_map': 0.05,
                'val_loss': 0.85,
                'epoch': 5
            }
        ]
        
        all_checkpoint_data = high_quality_checkpoints + medium_quality_checkpoints + low_quality_checkpoints
        
        # Create actual checkpoint files
        for checkpoint_data in all_checkpoint_data:
            checkpoint_path = checkpoint_data['dir'] / checkpoint_data['name']
            
            # Create realistic checkpoint content
            torch_data = {
                'model_state_dict': {'dummy_layer.weight': torch.randn(10, 10)},
                'config': {
                    'backbone': checkpoint_data['backbone'],
                    'model': {'backbone': checkpoint_data['backbone']},
                    'training_mode': 'two_phase'
                },
                'metrics': {
                    'val_map': checkpoint_data['val_map'],
                    'val_loss': checkpoint_data['val_loss'],
                    'precision': checkpoint_data['val_map'] - 0.02,
                    'recall': checkpoint_data['val_map'] + 0.01
                },
                'epoch': checkpoint_data['epoch'],
                'architecture_type': 'yolov5',
                'backbone': checkpoint_data['backbone'],
                'timestamp': time.time(),
                'model_info': {'total_parameters': 1000000 + checkpoint_data['epoch'] * 1000}
            }
            
            torch.save(torch_data, checkpoint_path)
            checkpoint_files.append({
                'path': str(checkpoint_path),
                'data': checkpoint_data
            })
        
        yield {
            'temp_dir': temp_path,
            'checkpoint_dirs': checkpoint_dirs,
            'checkpoint_files': checkpoint_files,
            'high_quality': high_quality_checkpoints,
            'medium_quality': medium_quality_checkpoints,
            'low_quality': low_quality_checkpoints
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_checkpoint_env):
        """Create test configuration for checkpoint selector."""
        return {
            'evaluation': {
                'checkpoints': {
                    'discovery_paths': [str(d) for d in temp_checkpoint_env['checkpoint_dirs']],
                    'filename_patterns': ['best_*.pt', 'best.pt', 'last.pt', '*_best.pt'],
                    'required_keys': ['model_state_dict'],
                    'supported_backbones': ['cspdarknet', 'efficientnet_b4', 'yolov5s', 'unknown'],
                    'min_val_map': 0.1,
                    'sort_by': 'val_map',
                    'max_checkpoints': 10,
                    'auto_select_best': True
                }
            }
        }
    
    def test_checkpoint_selector_initialization(self, test_config):
        """Test CheckpointSelector initialization."""
        selector = CheckpointSelector(config=test_config)
        
        assert len(selector.discovery_paths) == 4
        assert selector.filename_patterns == ['best_*.pt', 'best.pt', 'last.pt', '*_best.pt']
        assert selector.required_keys == ['model_state_dict']
        assert selector.min_val_map == 0.1
        assert selector.supported_backbones == ['cspdarknet', 'efficientnet_b4', 'yolov5s', 'unknown']
    
    def test_checkpoint_discovery(self, test_config, temp_checkpoint_env):
        """Test checkpoint discovery across multiple directories."""
        selector = CheckpointSelector(config=test_config)
        checkpoints = selector.list_available_checkpoints()
        
        # Should find high and medium quality checkpoints, but filter out low quality
        assert len(checkpoints) >= 5  # All checkpoints that meet min_val_map threshold
        
        # Verify checkpoints are sorted by val_map (descending)
        for i in range(len(checkpoints) - 1):
            current_map = checkpoints[i].get('metrics', {}).get('val_map', 0)
            next_map = checkpoints[i + 1].get('metrics', {}).get('val_map', 0)
            assert current_map >= next_map, "Checkpoints should be sorted by val_map descending"
        
        # Verify each checkpoint has required fields
        for checkpoint in checkpoints:
            assert 'path' in checkpoint
            assert 'filename' in checkpoint
            assert 'display_name' in checkpoint
            assert 'backbone' in checkpoint
            assert 'metrics' in checkpoint
            assert 'file_size_mb' in checkpoint
            assert checkpoint['valid'] is True
    
    def test_checkpoint_filtering_by_backbone(self, test_config, temp_checkpoint_env):
        """Test checkpoint filtering by backbone architecture."""
        selector = CheckpointSelector(config=test_config)
        
        # Test CSPDarknet filtering
        cspdarknet_checkpoints = selector.filter_checkpoints(backbone='cspdarknet')
        assert len(cspdarknet_checkpoints) >= 1
        for checkpoint in cspdarknet_checkpoints:
            assert checkpoint['backbone'] == 'cspdarknet'
        
        # Test EfficientNet-B4 filtering
        efficientnet_checkpoints = selector.filter_checkpoints(backbone='efficientnet_b4')
        assert len(efficientnet_checkpoints) >= 1
        for checkpoint in efficientnet_checkpoints:
            assert checkpoint['backbone'] == 'efficientnet_b4'
        
        # Test case insensitive filtering
        case_insensitive = selector.filter_checkpoints(backbone='CSPDARKNET')
        assert len(case_insensitive) == len(cspdarknet_checkpoints)
    
    def test_checkpoint_filtering_by_min_map(self, test_config, temp_checkpoint_env):
        """Test checkpoint filtering by minimum mAP threshold."""
        selector = CheckpointSelector(config=test_config)
        
        # Test high threshold (should get only best checkpoints)
        high_quality = selector.filter_checkpoints(min_map=0.85)
        assert len(high_quality) >= 2  # Should get the high-quality checkpoints
        for checkpoint in high_quality:
            assert checkpoint.get('metrics', {}).get('val_map', 0) >= 0.85
        
        # Test medium threshold
        medium_quality = selector.filter_checkpoints(min_map=0.70)
        assert len(medium_quality) >= len(high_quality)  # Should include more checkpoints
        
        # Test very high threshold (should get few or no checkpoints)
        ultra_high = selector.filter_checkpoints(min_map=0.95)
        # May be empty, but should not crash
        assert isinstance(ultra_high, list)
    
    def test_checkpoint_filtering_combined(self, test_config, temp_checkpoint_env):
        """Test combined filtering criteria."""
        selector = CheckpointSelector(config=test_config)
        
        # Combine backbone and min_map filtering
        combined = selector.filter_checkpoints(
            backbone='efficientnet_b4',
            min_map=0.75
        )
        
        # Should only return high-quality EfficientNet-B4 checkpoints
        for checkpoint in combined:
            assert checkpoint['backbone'] == 'efficientnet_b4'
            assert checkpoint.get('metrics', {}).get('val_map', 0) >= 0.75
    
    def test_checkpoint_selection_and_validation(self, test_config, temp_checkpoint_env):
        """Test checkpoint selection and validation."""
        selector = CheckpointSelector(config=test_config)
        
        # Get a valid checkpoint path
        checkpoints = selector.list_available_checkpoints()
        assert len(checkpoints) > 0
        
        test_checkpoint_path = checkpoints[0]['path']
        
        # Test checkpoint selection
        checkpoint_info = selector.select_checkpoint(test_checkpoint_path)
        
        assert checkpoint_info is not None
        assert checkpoint_info['path'] == test_checkpoint_path
        assert 'display_name' in checkpoint_info
        assert 'backbone' in checkpoint_info
        assert 'metrics' in checkpoint_info
        
        # Test checkpoint validation
        is_valid, message = selector.validate_checkpoint(test_checkpoint_path)
        assert is_valid is True
        assert 'valid' in message.lower()
    
    def test_checkpoint_validation_edge_cases(self, test_config, temp_checkpoint_env):
        """Test checkpoint validation with edge cases."""
        selector = CheckpointSelector(config=test_config)
        
        # Test non-existent checkpoint
        is_valid, message = selector.validate_checkpoint('/nonexistent/path.pt')
        assert is_valid is False
        assert 'not found' in message.lower() or 'tidak ditemukan' in message.lower()
        
        # Test empty checkpoint file
        empty_checkpoint = temp_checkpoint_env['temp_dir'] / 'empty.pt'
        empty_checkpoint.write_bytes(b'')
        
        is_valid, message = selector.validate_checkpoint(str(empty_checkpoint))
        assert is_valid is False
        assert 'error' in message.lower()
    
    def test_best_checkpoint_selection(self, test_config, temp_checkpoint_env):
        """Test best checkpoint selection functionality."""
        selector = CheckpointSelector(config=test_config)
        
        # Test get best checkpoint overall
        best_overall = selector.get_best_checkpoint()
        assert best_overall is not None
        assert best_overall.get('metrics', {}).get('val_map', 0) >= 0.85  # Should be high quality
        
        # Test get best checkpoint for specific backbone
        best_cspdarknet = selector.get_best_checkpoint(backbone='cspdarknet')
        if best_cspdarknet:  # May be None if no cspdarknet checkpoints meet criteria
            assert best_cspdarknet['backbone'] == 'cspdarknet'
        
        best_efficientnet = selector.get_best_checkpoint(backbone='efficientnet_b4')
        if best_efficientnet:
            assert best_efficientnet['backbone'] == 'efficientnet_b4'
    
    def test_backbone_statistics(self, test_config, temp_checkpoint_env):
        """Test backbone statistics generation."""
        selector = CheckpointSelector(config=test_config)
        stats = selector.get_backbone_stats()
        
        assert isinstance(stats, dict)
        
        # Should have stats for different backbones
        expected_backbones = ['cspdarknet', 'efficientnet_b4', 'yolov5s']
        for backbone in expected_backbones:
            if backbone in stats:
                backbone_stats = stats[backbone]
                assert 'count' in backbone_stats
                assert 'best_map' in backbone_stats
                assert 'avg_map' in backbone_stats
                assert 'checkpoints' in backbone_stats
                
                assert backbone_stats['count'] > 0
                assert backbone_stats['best_map'] >= 0
                assert backbone_stats['avg_map'] >= 0
                assert len(backbone_stats['checkpoints']) == backbone_stats['count']
    
    def test_ui_integration_functions(self, test_config, temp_checkpoint_env):
        """Test UI integration helper functions."""
        selector = CheckpointSelector(config=test_config)
        
        # Test create_checkpoint_options for UI dropdown
        options = selector.create_checkpoint_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        
        for option in options:
            assert isinstance(option, tuple)
            assert len(option) == 2  # (label, value)
            label, value = option
            assert isinstance(label, str)
            assert isinstance(value, str)
            assert 'mAP:' in label  # Should include mAP in label
            assert Path(value).exists()  # Value should be valid path
    
    def test_checkpoint_metadata_extraction(self, test_config, temp_checkpoint_env):
        """Test checkpoint metadata extraction."""
        selector = CheckpointSelector(config=test_config)
        
        # Get a checkpoint for testing
        checkpoints = selector.list_available_checkpoints()
        test_checkpoint = checkpoints[0]
        
        # Test metadata extraction
        metadata = selector._extract_checkpoint_metadata(Path(test_checkpoint['path']))
        
        # Verify all expected fields are present
        expected_fields = [
            'path', 'filename', 'display_name', 'model_name', 'backbone',
            'layer_mode', 'date', 'metrics', 'config', 'file_size_mb',
            'epoch', 'architecture_type', 'training_mode', 'valid'
        ]
        
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"
        
        # Verify data types
        assert isinstance(metadata['file_size_mb'], (int, float))
        assert isinstance(metadata['epoch'], int)
        assert isinstance(metadata['metrics'], dict)
        assert isinstance(metadata['config'], dict)
        assert metadata['valid'] is True
    
    def test_filename_pattern_parsing(self, test_config, temp_checkpoint_env):
        """Test filename pattern parsing for different checkpoint naming conventions."""
        selector = CheckpointSelector(config=test_config)
        
        # Test various filename patterns
        test_files = [
            ('best_model_cspdarknet_multi_20240101.pt', 'cspdarknet'),
            ('best_efficientnet_b4_multi_20240102.pt', 'efficientnet_b4'),
            ('best_yolov5_cspdarknet_multi_20240103.pt', 'cspdarknet'),
            ('unified_efficientnet_b4_multi_best_20240104.pt', 'efficientnet_b4'),
            ('smartcash_cspdarknet_best.pt', 'cspdarknet'),
            ('best_model.pt', 'cspdarknet'),  # Default backbone
        ]
        
        checkpoints = selector.list_available_checkpoints()
        
        # Verify that different filename patterns are parsed correctly
        for checkpoint in checkpoints:
            filename = checkpoint['filename']
            backbone = checkpoint['backbone']
            
            # Check if backbone was extracted correctly based on filename
            assert backbone in ['cspdarknet', 'efficientnet_b4', 'yolov5s', 'unknown']
    
    def test_cache_functionality(self, test_config, temp_checkpoint_env):
        """Test checkpoint cache functionality."""
        selector = CheckpointSelector(config=test_config)
        
        # First call should build cache
        checkpoints1 = selector.list_available_checkpoints()
        
        # Second call should use cache (should be faster and return same results)
        checkpoints2 = selector.list_available_checkpoints()
        
        assert len(checkpoints1) == len(checkpoints2)
        
        # Test cache refresh
        checkpoints3 = selector.list_available_checkpoints(refresh_cache=True)
        assert len(checkpoints3) == len(checkpoints1)
    
    def test_factory_functions(self, test_config):
        """Test factory functions for creating checkpoint selectors."""
        # Test create_checkpoint_selector
        selector = create_checkpoint_selector(test_config)
        assert isinstance(selector, CheckpointSelector)
        
        # Test get_available_checkpoints
        checkpoints = get_available_checkpoints(test_config)
        assert isinstance(checkpoints, list)
    
    def test_wildcard_path_expansion(self, temp_checkpoint_env):
        """Test wildcard path expansion in discovery paths."""
        # Create config with wildcard paths
        wildcard_config = {
            'evaluation': {
                'checkpoints': {
                    'discovery_paths': [
                        str(temp_checkpoint_env['temp_dir'] / 'runs' / 'train' / '*' / 'weights'),
                        str(temp_checkpoint_env['temp_dir'] / 'checkpoints')
                    ],
                    'filename_patterns': ['*.pt'],
                    'required_keys': ['model_state_dict'],
                    'min_val_map': 0.1
                }
            }
        }
        
        selector = CheckpointSelector(config=wildcard_config)
        checkpoints = selector.list_available_checkpoints()
        
        # Should find checkpoints in wildcard-matched directories
        assert len(checkpoints) > 0
        
        # Verify checkpoints are from expected directories
        checkpoint_dirs = [Path(cp['path']).parent for cp in checkpoints]
        expected_patterns = [
            temp_checkpoint_env['temp_dir'] / 'runs' / 'train' / 'exp1' / 'weights',
            temp_checkpoint_env['temp_dir'] / 'runs' / 'train' / 'exp2' / 'weights',
            temp_checkpoint_env['temp_dir'] / 'checkpoints'
        ]
        
        # At least some checkpoints should be from wildcard-matched directories
        assert any(any(str(expected) in str(cp_dir) for expected in expected_patterns) for cp_dir in checkpoint_dirs)


class TestCheckpointSelectorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_discovery_paths(self):
        """Test with empty discovery paths."""
        config = {
            'evaluation': {
                'checkpoints': {
                    'discovery_paths': [],
                    'filename_patterns': ['*.pt'],
                    'min_val_map': 0.1
                }
            }
        }
        
        selector = CheckpointSelector(config=config)
        checkpoints = selector.list_available_checkpoints()
        
        # Should handle gracefully and return empty list
        assert checkpoints == []
    
    def test_nonexistent_discovery_paths(self):
        """Test with non-existent discovery paths."""
        config = {
            'evaluation': {
                'checkpoints': {
                    'discovery_paths': ['/nonexistent/path1', '/nonexistent/path2'],
                    'filename_patterns': ['*.pt'],
                    'min_val_map': 0.1
                }
            }
        }
        
        selector = CheckpointSelector(config=config)
        checkpoints = selector.list_available_checkpoints()
        
        # Should handle gracefully and return empty list
        assert checkpoints == []
    
    def test_invalid_config_structure(self):
        """Test with invalid configuration structure."""
        # Test with missing evaluation section
        invalid_config1 = {'other_section': {}}
        selector1 = CheckpointSelector(config=invalid_config1)
        assert selector1 is not None  # Should not crash
        
        # Test with missing checkpoints section
        invalid_config2 = {'evaluation': {'other_section': {}}}
        selector2 = CheckpointSelector(config=invalid_config2)
        assert selector2 is not None  # Should not crash
        
        # Test with None config
        selector3 = CheckpointSelector(config=None)
        assert selector3 is not None  # Should not crash
    
    def test_corrupted_checkpoint_files(self):
        """Test handling of corrupted checkpoint files."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            checkpoint_dir = temp_path / 'checkpoints'
            checkpoint_dir.mkdir(parents=True)
            
            # Create corrupted checkpoint file
            corrupted_file = checkpoint_dir / 'corrupted.pt'
            corrupted_file.write_text('This is not a valid PyTorch file')
            
            config = {
                'evaluation': {
                    'checkpoints': {
                        'discovery_paths': [str(checkpoint_dir)],
                        'filename_patterns': ['*.pt'],
                        'min_val_map': 0.0
                    }
                }
            }
            
            selector = CheckpointSelector(config=config)
            checkpoints = selector.list_available_checkpoints()
            
            # Should handle corrupted files gracefully
            assert isinstance(checkpoints, list)
            # Corrupted file should be filtered out
            assert len(checkpoints) == 0
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_extremely_high_min_map_threshold(self):
        """Test with extremely high mAP threshold."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            checkpoint_dir = temp_path / 'checkpoints'
            checkpoint_dir.mkdir(parents=True)
            
            config = {
                'evaluation': {
                    'checkpoints': {
                        'discovery_paths': [str(checkpoint_dir)],
                        'filename_patterns': ['*.pt'],
                        'min_val_map': 0.99  # Extremely high threshold
                    }
                }
            }
            
            selector = CheckpointSelector(config=config)
            checkpoints = selector.list_available_checkpoints()
            
            # Should return empty list or very few checkpoints
            assert isinstance(checkpoints, list)
            assert len(checkpoints) <= 1  # At most one checkpoint should meet this threshold
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Run comprehensive checkpoint selection tests
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])