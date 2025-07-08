"""
File: tests/unit/ui/dataset/preprocess/test_operation_summary.py
Description: Tests for operation summary components
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.dataset.preprocess.components.operation_summary import (
    create_operation_summary, create_operation_results_summary,
    _create_class_badges, _get_class_color, _generate_summary_html,
    _create_preprocess_results_summary, _create_check_results_summary,
    _create_cleanup_results_summary, _create_generic_results_summary,
    _format_split_stats
)
from smartcash.ui.dataset.preprocess.constants import BANKNOTE_CLASSES


class TestOperationSummary:
    """Test operation summary creation and functionality"""
    
    def test_create_operation_summary_default(self):
        """Test creating operation summary with default config"""
        summary = create_operation_summary()
        
        assert isinstance(summary, widgets.VBox)
        assert len(summary.children) == 1
        assert isinstance(summary.children[0], widgets.HTML)
        
        # Check that update method is attached
        assert hasattr(summary, 'update_summary')
        assert callable(summary.update_summary)
        assert hasattr(summary, 'summary_widget')
    
    def test_create_operation_summary_with_config(self):
        """Test creating operation summary with custom config"""
        config = {
            'preprocessing': {
                'normalization': {
                    'preset': 'yolov5l',
                    'method': 'zscore'
                },
                'target_splits': ['train', 'test'],
                'batch_size': 64,
                'validation': {'enabled': True},
                'cleanup_target': 'both'
            },
            'data': {
                'dir': 'custom_data',
                'preprocessed_dir': 'custom_output'
            }
        }
        
        summary = create_operation_summary(config)
        html_content = summary.children[0].value
        
        # Check that custom config values are displayed
        assert 'yolov5l' in html_content
        assert 'zscore' not in html_content  # Method is not directly shown in display
        assert 'train, test' in html_content
        assert '64' in html_content
        assert 'Aktif' in html_content  # Validation enabled
        assert 'both' in html_content
        assert 'custom_data' in html_content
        assert 'custom_output' in html_content
    
    def test_create_operation_summary_preset_sizes(self):
        """Test operation summary shows correct sizes for different presets"""
        configs = [
            ({'preprocessing': {'normalization': {'preset': 'yolov5s'}}}, '640x640'),
            ({'preprocessing': {'normalization': {'preset': 'yolov5l'}}}, '832x832'),
            ({'preprocessing': {'normalization': {'preset': 'yolov5x'}}}, '1024x1024'),
            ({'preprocessing': {'normalization': {'preset': 'custom'}}}, '640x640')  # Default
        ]
        
        for config, expected_size in configs:
            summary = create_operation_summary(config)
            html_content = summary.children[0].value
            assert expected_size in html_content
    
    def test_update_summary_method(self):
        """Test updating summary with new configuration"""
        summary = create_operation_summary()
        
        new_config = {
            'preprocessing': {
                'normalization': {'preset': 'yolov5x'},
                'batch_size': 128,
                'target_splits': ['train', 'valid', 'test']
            },
            'data': {
                'dir': 'new_data',
                'preprocessed_dir': 'new_output'
            }
        }
        
        summary.update_summary(new_config)
        html_content = summary.summary_widget.value
        
        # Check that new values are reflected
        assert 'yolov5x' in html_content
        assert '1024x1024' in html_content  # yolov5x size
        assert '128' in html_content
        assert 'train, valid, test' in html_content
        assert 'new_data' in html_content
        assert 'new_output' in html_content


class TestClassBadges:
    """Test banknote class badge creation"""
    
    def test_create_class_badges(self):
        """Test creating banknote class badges"""
        badges_html = _create_class_badges()
        
        assert isinstance(badges_html, str)
        assert len(badges_html) > 0
        
        # Check that all banknote classes are represented
        for class_id, class_info in BANKNOTE_CLASSES.items():
            display_name = class_info['display']
            assert display_name in badges_html
        
        # Check HTML structure
        assert '<span' in badges_html
        assert 'background-color:' in badges_html
        assert 'border-radius:' in badges_html
    
    def test_get_class_color(self):
        """Test getting colors for banknote classes"""
        colors = []
        for class_id in range(len(BANKNOTE_CLASSES)):
            color = _get_class_color(class_id)
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
            colors.append(color)
        
        # Check that different classes get different colors (mostly)
        assert len(set(colors)) >= len(BANKNOTE_CLASSES) // 2
    
    def test_get_class_color_wrap_around(self):
        """Test color assignment wraps around for large class IDs"""
        color_0 = _get_class_color(0)
        color_7 = _get_class_color(7)  # Should wrap around
        
        assert color_0 == color_7  # Should be the same due to modulo


class TestGenerateSummaryHTML:
    """Test HTML generation for summary"""
    
    def test_generate_summary_html_complete(self):
        """Test generating complete summary HTML"""
        config = {
            'preprocessing': {
                'normalization': {
                    'preset': 'yolov5l',
                    'method': 'minmax'
                },
                'target_splits': ['train', 'valid'],
                'batch_size': 32,
                'validation': {'enabled': False},
                'cleanup_target': 'preprocessed'
            },
            'data': {
                'dir': 'test_data',
                'preprocessed_dir': 'test_output'
            }
        }
        
        html = _generate_summary_html(config)
        
        assert isinstance(html, str)
        assert len(html) > 0
        
        # Check structure
        assert '<div' in html
        assert 'Konfigurasi Preprocessing' in html
        assert 'Normalisasi' in html
        assert 'Processing' in html
        assert 'Classes yang Didukung' in html
        
        # Check content
        assert 'yolov5l' in html
        assert '832x832' in html
        assert 'train, valid' in html
        assert '32' in html
        assert 'Minimal' in html  # Validation disabled
        assert 'preprocessed' in html
        assert 'test_data' in html
        assert 'test_output' in html
    
    def test_generate_summary_html_minimal(self):
        """Test generating summary HTML with minimal config"""
        config = {}
        
        html = _generate_summary_html(config)
        
        # Should still generate valid HTML with defaults
        assert isinstance(html, str)
        assert len(html) > 0
        assert 'yolov5s' in html  # Default preset
        assert '640x640' in html  # Default size


class TestOperationResultsSummary:
    """Test operation results summary creation"""
    
    def test_create_operation_results_summary_no_results(self):
        """Test creating results summary with no results"""
        summary = create_operation_results_summary()
        
        assert isinstance(summary, widgets.VBox)
        assert len(summary.children) == 1
        assert isinstance(summary.children[0], widgets.HTML)
    
    def test_create_operation_results_summary_preprocess(self):
        """Test creating preprocess results summary"""
        results = {
            'operation': 'preprocess',
            'success': True,
            'stats': {
                'total_files': 100,
                'processed_files': 95
            },
            'configuration': {
                'normalization_preset': 'yolov5l',
                'target_splits': ['train', 'valid']
            }
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Preprocessing Completed' in html_content
        assert '100' in html_content  # Total files
        assert '95' in html_content   # Processed files
        assert '95.0%' in html_content  # Success rate
        assert 'yolov5l' in html_content
        assert 'train, valid' in html_content
    
    def test_create_operation_results_summary_check(self):
        """Test creating check results summary"""
        results = {
            'operation': 'check',
            'success': True,
            'service_ready': True,
            'file_statistics': {
                'train': {
                    'raw_images': 50,
                    'preprocessed_files': 25,
                    'augmented_files': 10,
                    'sample_files': 5
                },
                'valid': {
                    'raw_images': 20,
                    'preprocessed_files': 10,
                    'augmented_files': 5,
                    'sample_files': 2
                }
            }
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Dataset Ready' in html_content
        assert '70' in html_content  # Total raw (50 + 20)
        assert '35' in html_content  # Total preprocessed (25 + 10)
        assert 'Yes' in html_content  # Ready for processing
        assert 'train: 50 raw, 25 processed' in html_content
        assert 'valid: 20 raw, 10 processed' in html_content
    
    def test_create_operation_results_summary_check_not_ready(self):
        """Test creating check results summary when not ready"""
        results = {
            'operation': 'check',
            'success': True,
            'service_ready': False,
            'file_statistics': {
                'train': {'raw_images': 0, 'preprocessed_files': 0}
            }
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Dataset Not Ready' in html_content
        assert '⚠️' in html_content
        assert 'No' in html_content  # Not ready for processing
    
    def test_create_operation_results_summary_cleanup(self):
        """Test creating cleanup results summary"""
        results = {
            'operation': 'cleanup',
            'success': True,
            'files_removed': 25,
            'cleanup_target': 'preprocessed',
            'affected_splits': ['train', 'valid']
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Cleanup Completed' in html_content
        assert '25' in html_content
        assert 'preprocessed' in html_content
        assert 'Success' in html_content
        assert 'train, valid' in html_content
    
    def test_create_operation_results_summary_generic(self):
        """Test creating generic results summary"""
        results = {
            'operation': 'unknown',
            'success': True,
            'message': 'Custom operation completed successfully'
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Unknown Results' in html_content
        assert '✅' in html_content
        assert 'Custom operation completed successfully' in html_content
    
    def test_create_operation_results_summary_generic_failure(self):
        """Test creating generic results summary for failure"""
        results = {
            'operation': 'test',
            'success': False,
            'message': 'Operation failed with error'
        }
        
        summary = create_operation_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Test Results' in html_content
        assert '❌' in html_content
        assert 'Operation failed with error' in html_content


class TestResultsSummaryHelpers:
    """Test helper functions for results summary"""
    
    def test_create_preprocess_results_summary(self):
        """Test preprocess-specific results summary"""
        results = {
            'stats': {'total_files': 200, 'processed_files': 180},
            'configuration': {
                'normalization_preset': 'yolov5x',
                'target_splits': ['train', 'test', 'valid']
            }
        }
        
        summary = _create_preprocess_results_summary(results)
        html_content = summary.children[0].value
        
        assert '200' in html_content
        assert '180' in html_content
        assert '90.0%' in html_content  # 180/200
        assert 'yolov5x' in html_content
        assert 'train, test, valid' in html_content
    
    def test_create_preprocess_results_summary_division_by_zero(self):
        """Test preprocess results summary with zero total files"""
        results = {
            'stats': {'total_files': 0, 'processed_files': 0},
            'configuration': {'normalization_preset': 'yolov5s', 'target_splits': []}
        }
        
        summary = _create_preprocess_results_summary(results)
        html_content = summary.children[0].value
        
        # Should handle division by zero gracefully
        assert 'N/A' in html_content or '0' in html_content
    
    def test_create_check_results_summary(self):
        """Test check-specific results summary"""
        results = {
            'service_ready': True,
            'file_statistics': {
                'train': {'raw_images': 100, 'preprocessed_files': 50},
                'valid': {'raw_images': 50, 'preprocessed_files': 25}
            }
        }
        
        summary = _create_check_results_summary(results)
        html_content = summary.children[0].value
        
        assert '150' in html_content  # Total raw
        assert '75' in html_content   # Total preprocessed
        assert 'Dataset Ready' in html_content
    
    def test_create_cleanup_results_summary(self):
        """Test cleanup-specific results summary"""
        results = {
            'files_removed': 42,
            'cleanup_target': 'both',
            'affected_splits': ['train', 'valid', 'test']
        }
        
        summary = _create_cleanup_results_summary(results)
        html_content = summary.children[0].value
        
        assert '42' in html_content
        assert 'both' in html_content
        assert 'train, valid, test' in html_content
        assert 'Success' in html_content
    
    def test_create_generic_results_summary(self):
        """Test generic results summary"""
        results = {
            'success': True,
            'message': 'Test operation completed',
            'operation': 'test'
        }
        
        summary = _create_generic_results_summary(results)
        html_content = summary.children[0].value
        
        assert 'Test Results' in html_content
        assert 'Test operation completed' in html_content
        assert '✅' in html_content
    
    def test_format_split_stats_empty(self):
        """Test formatting split statistics with empty data"""
        result = _format_split_stats({})
        assert result == "No data available"
    
    def test_format_split_stats_none(self):
        """Test formatting split statistics with None data"""
        result = _format_split_stats(None)
        assert result == "No data available"
    
    def test_format_split_stats_valid(self):
        """Test formatting split statistics with valid data"""
        file_stats = {
            'train': {'raw_images': 100, 'preprocessed_files': 80},
            'valid': {'raw_images': 50, 'preprocessed_files': 40},
            'test': {'raw_images': 25, 'preprocessed_files': 20}
        }
        
        result = _format_split_stats(file_stats)
        
        assert '• train: 100 raw, 80 processed' in result
        assert '• valid: 50 raw, 40 processed' in result
        assert '• test: 25 raw, 20 processed' in result
        assert '<br>' in result
    
    def test_format_split_stats_missing_keys(self):
        """Test formatting split statistics with missing keys"""
        file_stats = {
            'train': {},  # Missing keys
            'valid': {'raw_images': 30}  # Missing preprocessed_files
        }
        
        result = _format_split_stats(file_stats)
        
        assert '• train: 0 raw, 0 processed' in result
        assert '• valid: 30 raw, 0 processed' in result


class TestOperationSummaryIntegration:
    """Test integration of operation summary components"""
    
    def test_summary_and_results_integration(self):
        """Test that operation summary and results work together"""
        config = {
            'preprocessing': {
                'normalization': {'preset': 'yolov5l'},
                'target_splits': ['train', 'valid'],
                'batch_size': 64
            }
        }
        
        # Create operation summary
        op_summary = create_operation_summary(config)
        
        # Create results summary
        results = {
            'operation': 'preprocess',
            'success': True,
            'stats': {'total_files': 100, 'processed_files': 100},
            'configuration': {
                'normalization_preset': 'yolov5l',
                'target_splits': ['train', 'valid']
            }
        }
        results_summary = create_operation_results_summary(results)
        
        # Both should be valid widgets
        assert isinstance(op_summary, widgets.VBox)
        assert isinstance(results_summary, widgets.VBox)
        
        # Both should contain HTML content
        assert isinstance(op_summary.children[0], widgets.HTML)
        assert isinstance(results_summary.children[0], widgets.HTML)
        
        # Content should be consistent
        op_html = op_summary.children[0].value
        results_html = results_summary.children[0].value
        
        assert 'yolov5l' in op_html
        assert 'yolov5l' in results_html
        assert 'train, valid' in op_html
        assert 'train, valid' in results_html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])