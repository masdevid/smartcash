"""
Unit tests for metrics_analyzer.py
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


class TestMetricsAnalyzer:
    """Test cases for MetricsAnalyzer class."""
    
    @pytest.fixture(autouse=True)
    def setup_analyzer(self, sample_metrics_tracker):
        """Set up a metrics analyzer with sample data."""
        from smartcash.model.training.visualization.metrics_analyzer import MetricsAnalyzer
        
        # Add some test data to the tracker
        tracker = sample_metrics_tracker
        current_time = datetime.now().timestamp()
        
        # Add 5 epochs of training data
        for i in range(5):
            tracker.epoch_metrics.append({
                'epoch': i,
                'phase': 'training',
                'timestamp': current_time - (4 - i) * 60,  # 1 minute apart
                'train_loss': 1.0 - (i * 0.1),  # Decreasing loss
                'val_loss': 1.1 - (i * 0.08),   # Decreasing but with different rate
                'learning_rate': 0.001,
                'phase_num': 1
            })
            
            # Add layer metrics
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if f'{layer}_{metric}' not in tracker.layer_metrics[layer]:
                        tracker.layer_metrics[layer][metric] = []
                    tracker.layer_metrics[layer][metric].append(0.8 + (i * 0.05))  # Increasing metrics
        
        # Add a phase transition
        tracker.phase_transitions.append({
            'epoch': 2,
            'from_phase': 1,
            'to_phase': 2,
            'timestamp': current_time - 180  # 3 minutes ago
        })
        
        self.analyzer = MetricsAnalyzer(tracker, verbose=True)
        
    def test_analyze_training_progress(self):
        """Test the main analysis method."""
        analysis = self.analyzer.analyze_training_progress()
        
        # Check basic structure
        assert 'training_duration' in analysis
        assert 'best_epoch' in analysis
        assert 'final_metrics' in analysis
        assert 'convergence_analysis' in analysis
        assert 'phase_transitions' in analysis
        assert 'layer_performance' in analysis
        
        # Check some specific values
        assert analysis['final_metrics']['epoch'] == 4
        assert analysis['phase_transitions'][0]['from_phase'] == 1
        assert analysis['phase_transitions'][0]['to_phase'] == 2
        
    def test_calculate_training_duration(self):
        """Test training duration calculation."""
        duration = self.analyzer._calculate_training_duration()
        
        assert 'start_time' in duration
        assert 'end_time' in duration
        assert 'duration_seconds' in duration
        assert 'duration_minutes' in duration
        assert 'duration_hours' in duration
        
        # Should be approximately 4 minutes (4 intervals of 1 minute)
        assert 3.9 < duration['duration_seconds'] / 60 < 4.1
        
    def test_find_best_epoch(self):
        """Test finding the best epoch based on validation loss."""
        best_epoch = self.analyzer._find_best_epoch()
        
        # With decreasing val_loss, best epoch should be the last one
        assert best_epoch['epoch'] == 5  # 0-based index + 1
        assert best_epoch['phase'] == 'training'
        
    def test_get_final_metrics(self):
        """Test extraction of final metrics."""
        final_metrics = self.analyzer._get_final_metrics()
        
        assert final_metrics['epoch'] == 4  # 0-based index
        assert final_metrics['phase'] == 'training'
        assert final_metrics['train_loss'] == pytest.approx(0.6)  # 1.0 - (4 * 0.1)
        assert final_metrics['learning_rate'] == 0.001
        
        # Check layer metrics
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                assert f'{layer}_{metric}' in final_metrics
                assert final_metrics[f'{layer}_{metric}'] == pytest.approx(1.0)  # 0.8 + (4 * 0.05)
    
    def test_analyze_convergence(self):
        """Test convergence analysis."""
        # Add more epochs to test convergence
        for i in range(5, 15):  # Add 10 more epochs
            self.tracker.epoch_metrics.append({
                'epoch': i,
                'phase': 'training' if i % 2 == 0 else 'validation',
                'timestamp': i * 60,
                'train_loss': max(0.1, 0.5 - (i * 0.01)),  # Gradually decreasing loss
                'val_loss': max(0.15, 0.6 - (i * 0.008)),  # Gradually decreasing
                'learning_rate': 0.001 * (0.9 ** i),
                'phase_num': 2
            })
        
        convergence = self.analyzer._analyze_convergence()
        
        # Check that all expected keys are present
        for key in ['final_train_loss', 'final_val_loss', 'train_converged', 
                   'val_converged', 'final_train_slope', 'final_val_slope']:
            assert key in convergence
        
        # With our test data, training should be converging
        assert isinstance(convergence['train_converged'], bool)
        assert isinstance(convergence['val_converged'], bool)
        
        # Loss should be decreasing
        assert convergence['final_train_loss'] < 0.5
        assert convergence['final_val_loss'] < 0.6
        
    def test_analyze_phase_transitions(self):
        """Test phase transition analysis."""
        transitions = self.analyzer._analyze_phase_transitions()
        
        assert len(transitions) == 1
        assert transitions[0]['epoch'] == 2
        assert transitions[0]['from_phase'] == 1
        assert transitions[0]['to_phase'] == 2
        
    def test_analyze_layer_performance(self):
        """Test layer performance analysis."""
        # Add more epochs with varying metrics
        for i in range(5, 10):
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    # Add some variation to metrics
                    base_value = 0.7 + (i * 0.05) + (0.01 * hash(layer) % 0.1)
                    self.tracker.layer_metrics[layer][metric].append(base_value)
        
        performance = self.analyzer._analyze_layer_performance()
        
        # Check structure
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            assert layer in performance
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                assert metric in performance[layer]
                
                # Check all required statistics are present
                for stat in ['final', 'max', 'min', 'mean', 'std']:
                    assert stat in performance[layer][metric]
                
                # Check values are within expected ranges
                assert 0 <= performance[layer][metric]['min'] <= 1.0
                assert 0 <= performance[layer][metric]['max'] <= 1.0
                assert 0 <= performance[layer][metric]['final'] <= 1.0
                assert 0 <= performance[layer][metric]['mean'] <= 1.0
                assert performance[layer][metric]['std'] >= 0
    
    @patch('pathlib.Path.mkdir')
    @patch('json.dump')
    @patch('builtins.open', new_callable=lambda: MagicMock())
    def test_save_analysis_report(self, mock_open, mock_json_dump, mock_mkdir):
        """Test saving analysis report to file."""
        from pathlib import Path
        
        report_path = self.analyzer.save_analysis_report(
            session_dir=Path('/tmp/test_session'),
            session_id='test123'
        )
        
        # Should create directory and save file
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(Path('/tmp/test_session/training_analysis.json'), 'w')
        
        # Check the report structure
        args, _ = mock_json_dump.call_args
        report = args[0]
        
        assert report['session_id'] == 'test123'
        assert 'timestamp' in report
        assert 'analysis' in report
        assert 'training_duration' in report['analysis']
        assert 'best_epoch' in report['analysis']
