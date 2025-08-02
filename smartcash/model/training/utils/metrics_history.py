#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/metrics_history.py

JSON-based metrics history recording system for training visualization.
Stores epoch-wise metrics per phase for easy chart generation.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Single epoch metrics record."""
    epoch: int
    phase: int
    timestamp: str
    train_loss: float
    val_loss: float
    learning_rate: float
    
    # YOLO detection metrics
    val_map50: Optional[float] = None
    val_map50_95: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    val_f1: Optional[float] = None
    val_accuracy: Optional[float] = None
    
    # Multi-task metrics (Phase 2)
    total_loss: Optional[float] = None
    layer_1_uncertainty: Optional[float] = None
    layer_2_uncertainty: Optional[float] = None
    layer_3_uncertainty: Optional[float] = None
    
    # Layer-specific metrics
    layer_1_accuracy: Optional[float] = None
    layer_1_precision: Optional[float] = None
    layer_1_recall: Optional[float] = None
    layer_1_f1: Optional[float] = None
    layer_2_accuracy: Optional[float] = None
    layer_2_precision: Optional[float] = None
    layer_2_recall: Optional[float] = None
    layer_2_f1: Optional[float] = None
    layer_3_accuracy: Optional[float] = None
    layer_3_precision: Optional[float] = None
    layer_3_recall: Optional[float] = None
    layer_3_f1: Optional[float] = None
    
    # Additional metrics for debugging
    additional_metrics: Optional[Dict[str, float]] = None


class MetricsHistoryRecorder:
    """
    JSON-based metrics history recorder for training visualization.
    
    Stores metrics per epoch and phase for easy chart generation
    and visualization manager access.
    """
    
    def __init__(self, output_dir: str = "training_logs", session_id: Optional[str] = None):
        """
        Initialize metrics history recorder.
        
        Args:
            output_dir: Directory to store metrics history files
            session_id: Optional session ID for file naming (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        
        # File paths for different data
        self.metrics_file = self.output_dir / f"metrics_history_{session_id}.json"
        self.phase_summary_file = self.output_dir / f"phase_summary_{session_id}.json"
        self.latest_file = self.output_dir / "latest_metrics.json"
        
        # In-memory storage for current session
        self.history: List[EpochMetrics] = []
        self.phase_summaries: Dict[int, Dict[str, Any]] = {}
        
        # Load existing data if available
        self._load_existing_data()
        
        logger.info(f"MetricsHistoryRecorder initialized for session {session_id}")
        logger.info(f"Metrics file: {self.metrics_file}")
    
    def _load_existing_data(self):
        """Load existing metrics data if files exist."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.history = [EpochMetrics(**record) for record in data]
                logger.info(f"Loaded {len(self.history)} existing metric records")
            
            if self.phase_summary_file.exists():
                with open(self.phase_summary_file, 'r') as f:
                    self.phase_summaries = json.load(f)
                logger.info(f"Loaded phase summaries for phases: {list(self.phase_summaries.keys())}")
                
        except Exception as e:
            logger.warning(f"Failed to load existing metrics data: {e}")
            self.history = []
            self.phase_summaries = {}
    
    def record_epoch(self, epoch: int, phase: int, metrics: Dict[str, float]):
        """
        Record metrics for a single epoch.
        
        Args:
            epoch: Epoch number
            phase: Training phase (1 or 2)
            metrics: Dictionary of metric names and values
        """
        try:
            # Extract core metrics with defaults
            train_loss = metrics.get('train_loss', 0.0)
            val_loss = metrics.get('val_loss', 0.0)
            learning_rate = metrics.get('learning_rate', 0.0)
            
            # Create epoch record
            epoch_record = EpochMetrics(
                epoch=epoch,
                phase=phase,
                timestamp=datetime.now().isoformat(),
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=learning_rate
            )
            
            # Fill YOLO detection metrics
            yolo_metrics = [
                'val_map50', 'val_map50_95', 'val_precision', 
                'val_recall', 'val_f1', 'val_accuracy'
            ]
            for metric in yolo_metrics:
                if metric in metrics:
                    setattr(epoch_record, metric, metrics[metric])
            
            # Fill multi-task metrics (mainly Phase 2)
            multitask_metrics = [
                'total_loss', 'layer_1_uncertainty', 'layer_2_uncertainty', 'layer_3_uncertainty'
            ]
            for metric in multitask_metrics:
                if metric in metrics:
                    setattr(epoch_record, metric, metrics[metric])
            
            # Fill layer-specific metrics
            layer_metrics = [
                'layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1',
                'layer_2_accuracy', 'layer_2_precision', 'layer_2_recall', 'layer_2_f1',
                'layer_3_accuracy', 'layer_3_precision', 'layer_3_recall', 'layer_3_f1'
            ]
            for metric in layer_metrics:
                if metric in metrics:
                    setattr(epoch_record, metric, metrics[metric])
            
            # Store additional metrics that don't fit standard schema
            additional = {}
            standard_metrics = set(['epoch', 'phase'] + yolo_metrics + multitask_metrics + layer_metrics + 
                                 ['train_loss', 'val_loss', 'learning_rate'])
            for key, value in metrics.items():
                if key not in standard_metrics and isinstance(value, (int, float)):
                    additional[key] = float(value)
            
            if additional:
                epoch_record.additional_metrics = additional
            
            # Add to history
            self.history.append(epoch_record)
            
            # Save immediately to disk
            self._save_metrics()
            self._save_latest()
            
            logger.debug(f"Recorded epoch {epoch} metrics for phase {phase}")
            
        except Exception as e:
            logger.error(f"Failed to record epoch {epoch} metrics: {e}")
    
    def finalize_phase(self, phase: int, final_metrics: Dict[str, float]):
        """
        Finalize a training phase and record summary.
        
        Args:
            phase: Phase number (1 or 2)
            final_metrics: Final metrics for the phase
        """
        try:
            # Get phase epochs
            phase_epochs = [record for record in self.history if record.phase == phase]
            
            if not phase_epochs:
                logger.warning(f"No epochs found for phase {phase}")
                return
            
            # Calculate phase statistics
            phase_summary = {
                'phase': phase,
                'start_time': phase_epochs[0].timestamp,
                'end_time': phase_epochs[-1].timestamp,
                'total_epochs': len(phase_epochs),
                'best_val_loss': min(e.val_loss for e in phase_epochs),
                'final_val_loss': phase_epochs[-1].val_loss,
                'final_metrics': final_metrics
            }
            
            # Add best mAP if available
            map_scores = [e.val_map50 for e in phase_epochs if e.val_map50 is not None]
            if map_scores:
                phase_summary['best_map50'] = max(map_scores)
                phase_summary['final_map50'] = phase_epochs[-1].val_map50
            
            self.phase_summaries[phase] = phase_summary
            self._save_phase_summary()
            
            logger.info(f"Finalized phase {phase}: {len(phase_epochs)} epochs, "
                       f"best val_loss: {phase_summary['best_val_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to finalize phase {phase}: {e}")
    
    def get_phase_data(self, phase: int) -> List[Dict[str, Any]]:
        """
        Get all metrics data for a specific phase.
        
        Args:
            phase: Phase number
            
        Returns:
            List of epoch metrics for the phase
        """
        phase_data = [asdict(record) for record in self.history if record.phase == phase]
        return phase_data
    
    def get_metric_series(self, metric_name: str, phase: Optional[int] = None) -> Dict[str, List]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            phase: Optional phase filter
            
        Returns:
            Dictionary with 'epochs', 'values', and 'phases' lists
        """
        epochs = []
        values = []
        phases = []
        
        for record in self.history:
            if phase is not None and record.phase != phase:
                continue
                
            # Get metric value
            metric_value = getattr(record, metric_name, None)
            if metric_value is None and record.additional_metrics:
                metric_value = record.additional_metrics.get(metric_name)
            
            if metric_value is not None:
                epochs.append(record.epoch)
                values.append(metric_value)
                phases.append(record.phase)
        
        return {
            'epochs': epochs,
            'values': values,
            'phases': phases
        }
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """
        Export data in format optimized for visualization manager.
        
        Returns:
            Dictionary with structured data for charts
        """
        export_data = {
            'session_id': self.session_id,
            'total_epochs': len(self.history),
            'phases': list(self.phase_summaries.keys()),
            'phase_summaries': self.phase_summaries,
            'metrics': {}
        }
        
        # Generate series for key metrics
        key_metrics = [
            'train_loss', 'val_loss', 'val_map50', 'val_precision', 
            'val_recall', 'learning_rate'
        ]
        
        for metric in key_metrics:
            export_data['metrics'][metric] = self.get_metric_series(metric)
        
        # Add phase-specific metrics
        for phase in [1, 2]:
            phase_data = self.get_phase_data(phase)
            export_data[f'phase_{phase}_data'] = phase_data
        
        return export_data
    
    def _save_metrics(self):
        """Save metrics history to JSON file."""
        try:
            data = [asdict(record) for record in self.history]
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _save_phase_summary(self):
        """Save phase summaries to JSON file."""
        try:
            with open(self.phase_summary_file, 'w') as f:
                json.dump(self.phase_summaries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save phase summary: {e}")
    
    def _save_latest(self):
        """Save latest metrics for quick access."""
        try:
            if self.history:
                latest_data = {
                    'latest_epoch': asdict(self.history[-1]),
                    'session_id': self.session_id,
                    'total_epochs': len(self.history),
                    'file_paths': {
                        'metrics': str(self.metrics_file),
                        'phase_summary': str(self.phase_summary_file)
                    }
                }
                with open(self.latest_file, 'w') as f:
                    json.dump(latest_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save latest metrics: {e}")


def create_metrics_recorder(output_dir: str = "training_logs", 
                           session_id: Optional[str] = None) -> MetricsHistoryRecorder:
    """
    Factory function to create metrics history recorder.
    
    Args:
        output_dir: Directory to store metrics files
        session_id: Optional session ID for file naming
        
    Returns:
        MetricsHistoryRecorder instance
    """
    return MetricsHistoryRecorder(output_dir, session_id)