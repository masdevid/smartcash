#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/smartcash/model/training/visualization_manager.py

Comprehensive training visualization and metrics tracking for SmartCash multi-layer detection model.

This module provides advanced visualization capabilities including:
- Training curves and loss tracking
- Confusion matrices for each detection layer
- Per-layer metrics comparison
- Phase transition analysis
- Learning rate schedules
- Comprehensive training dashboards

Features:
- Real-time metrics tracking during training
- Automatic chart generation with matplotlib/seaborn
- Support for multi-layer detection architectures
- Phase-aware visualization for multi-phase training
- Export capabilities for training analysis
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
from pathlib import Path

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from smartcash.common.logger import get_logger
from smartcash.model.training.charts.chart_generator import ChartGenerator
from smartcash.model.training.charts.chart_utils import ChartUtils

class VisualizationHelper:
    """Helper class for common visualization operations."""
    
    @staticmethod
    def get_default_layer_metrics() -> Dict[str, List[str]]:
        """Get default metrics for each layer."""
        return {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
    
    @staticmethod
    def initialize_layer_metrics(num_classes_per_layer: Dict[str, int]) -> Dict[str, Dict[str, List]]:
        """Initialize metrics structure for all layers."""
        return {
            layer: VisualizationHelper.get_default_layer_metrics()
            for layer in num_classes_per_layer.keys()
        }
    
    @staticmethod
    def extract_metrics_from_dict(metrics: Dict[str, Any], layer_patterns: Dict[str, List[str]]) -> Dict[str, Dict[str, List]]:
        """Extract metrics from a dictionary based on layer patterns."""
        extracted_metrics = {}
        for layer, metric_names in layer_patterns.items():
            layer_metrics = VisualizationHelper.get_default_layer_metrics()
            for metric_name in metric_names:
                if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                    metric_type = metric_name.split('_', 2)[-1]
                    metric_key_map = {
                        'accuracy': 'accuracy',
                        'precision': 'precision', 
                        'recall': 'recall',
                        'f1': 'f1_score'
                    }
                    if metric_type in metric_key_map:
                        storage_key = metric_key_map[metric_type]
                        layer_metrics[storage_key].append(metrics[metric_name])
            extracted_metrics[layer] = layer_metrics
        return extracted_metrics
    
    @staticmethod
    def generate_chart_with_error_handling(func):
        """Decorator to handle common chart generation errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if args[0].verbose:  # Assuming first arg is self
                    args[0].logger.warning(f"⚠️ Error generating chart: {str(e)}")
                return None
        return wrapper

from smartcash.common.logger import get_logger
from smartcash.model.training.charts.chart_generator import ChartGenerator
from smartcash.model.training.charts.chart_utils import ChartUtils

class VisualizationMetricsTracker:
    """
    Enhanced metrics tracker with confusion matrix and visualization capabilities.
    
    This class handles comprehensive tracking of training metrics across multiple
    detection layers, supporting the SmartCash multi-layer architecture with
    layer_1 (banknote detection), layer_2 (denomination features), and 
    layer_3 (common features).
    """
    
    def __init__(self, num_classes_per_layer: Dict[str, int], class_names: Dict[str, List[str]] = None, 
                 save_dir: str = "data/visualization", verbose: bool = False):
        """
        Initialize the comprehensive metrics tracker.
        
        Args:
            num_classes_per_layer: Dictionary mapping layer names to number of classes
            class_names: Optional mapping of layer name lists to class name lists
            save_dir: Directory to save visualization outputs
            verbose: Enable verbose logging
        """
        self.num_classes_per_layer = num_classes_per_layer
        self.class_names = class_names or self._generate_default_class_names()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize logger - using instance logger as per QWEN.md guidelines
        self.logger = get_logger(f'{self.__class__.__module__}.{self.__class__.__name__}')
        
        # Initialize chart generator
        self.chart_generator = ChartGenerator(save_dir=save_dir, verbose=verbose)
        
        # Metrics storage
        self.epoch_metrics = []
        self.confusion_matrices = defaultdict(list)  # Per layer
        self.predictions_history = defaultdict(list)
        self.ground_truth_history = defaultdict(list)
        
        # Training curves data
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.phase_transitions = []  # Track phase changes
        
        # Per-layer metrics
        self.layer_metrics = VisualizationHelper.initialize_layer_metrics(num_classes_per_layer)
        
        if verbose:
            self.logger.info("📊 Comprehensive metrics tracker initialized")
            self.logger.info(f"   • Layers: {list(num_classes_per_layer.keys())}")
            self.logger.info(f"   • Classes per layer: {num_classes_per_layer}")
            self.logger.info(f"   • Save directory: {self.save_dir}")
    
    def _generate_default_class_names(self) -> Dict[str, List[str]]:
        """
        Generate default class names based on SmartCash MODEL_ARC.md specifications.
        
        Returns:
            Dictionary mapping layer names to class name lists
        """
        return {
            'layer_1': ['001', '002', '005', '010', '020', '050', '100'],  # Banknote denominations
            'layer_2': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],  # Denomination features
            'layer_3': ['l3_sign', 'l3_text', 'l3_thread']  # Common features
        }
    
    def update_metrics(self, epoch: int, phase: str, metrics: Dict[str, Any], 
                      predictions: Dict[str, np.ndarray] = None, 
                      ground_truth: Dict[str, np.ndarray] = None, phase_num: int = None):
        try:
            # Add phase information
            metrics['phase_num'] = phase_num or 1
            metrics['phase_name'] = phase
            
            # Add epoch number
            metrics['epoch'] = epoch
            
            # Store metrics
            self.epoch_metrics.append(metrics)
            
            # Update loss tracking
            if 'train_loss' in metrics:
                self.train_losses.append(metrics['train_loss'])
            if 'val_loss' in metrics:
                self.val_losses.append(metrics['val_loss'])
            
            # Update phase transitions
            if phase_num and phase_num != self.current_phase:
                self.phase_transitions.append({
                    'epoch': epoch,
                    'from_phase': self.epoch_metrics[-2]['phase'],
                    'to_phase': phase
                })
                if self.verbose:
                    self.logger.info(f"📍 Phase transition at epoch {epoch}: {self.epoch_metrics[-2]['phase']} → {phase}")
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error updating metrics: {str(e)}")
    
    def _extract_layer_metrics_from_epoch_data(self, metrics: Dict[str, Any], epoch: int):
        """
        Extract layer-specific metrics from epoch metrics data.
        
        Args:
            metrics: Dictionary of epoch metrics
            epoch: Current epoch number
        """
        try:
            # Look for layer-specific metrics patterns
            layer_patterns = {
                'layer_1': ['layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1'],
                'layer_2': ['layer_2_accuracy', 'layer_2_precision', 'layer_2_recall', 'layer_2_f1'],
                'layer_3': ['layer_3_accuracy', 'layer_3_precision', 'layer_3_recall', 'layer_3_f1']
            }
            
            # Extract layer-specific metrics
            extracted_metrics = VisualizationHelper.extract_metrics_from_dict(metrics, layer_patterns)
            
            # Update our metrics storage
            for layer, metrics_dict in extracted_metrics.items():
                if layer in self.layer_metrics:
                    for metric_type, values in metrics_dict.items():
                        self.layer_metrics[layer][metric_type].extend(values)
            
            # Also look for global validation metrics that can be used as fallback
            global_metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
            global_values = {}
            for metric in global_metrics:
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    metric_type = metric.replace('val_', '')
                    if metric_type == 'f1':
                        metric_type = 'f1_score'
                    global_values[metric_type] = metrics[metric]
            
            # If no layer-specific metrics found, use global metrics as fallback for all layers
            if global_values:
                layers_with_data = any(
                    any(len(self.layer_metrics[layer][metric_type]) > 0 
                        for metric_type in ['accuracy', 'precision', 'recall', 'f1_score'])
                    for layer in self.layer_metrics.keys()
                )
                
                # Only use global fallback if no layer-specific data exists yet
                if not layers_with_data:
                    for layer in self.layer_metrics.keys():
                        for metric_type, value in global_values.items():
                            if metric_type in self.layer_metrics[layer]:
                                self.layer_metrics[layer][metric_type].append(value)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error extracting layer metrics from epoch data: {str(e)}")
    
    def _calculate_layer_confusion_matrix(self, layer: str, predictions: np.ndarray, ground_truth: np.ndarray, epoch: int):
        """
        Calculate confusion matrix for a specific layer.
        
        Args:
            layer: Layer name (e.g., 'layer_1', 'layer_2', 'layer_3')
            predictions: Model predictions for this layer
            ground_truth: Ground truth labels for this layer
            epoch: Current epoch number
        """
        try:
            # Convert predictions and ground truth to class indices
            pred_classes = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
            true_classes = np.argmax(ground_truth, axis=-1) if ground_truth.ndim > 1 else ground_truth
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_classes.flatten(), pred_classes.flatten(), 
                                labels=range(self.num_classes_per_layer[layer]))
            
            # Store confusion matrix
            cm_data = {
                'epoch': epoch,
                'matrix': cm,
                'accuracy': np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            }
            self.confusion_matrices[layer].append(cm_data)
            
            # Calculate per-class metrics
            if np.sum(cm) > 0:
                precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-10)
                recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                # Store layer metrics
                self.layer_metrics[layer]['precision'].append(np.mean(precision))
                self.layer_metrics[layer]['recall'].append(np.mean(recall))
                self.layer_metrics[layer]['f1_score'].append(np.mean(f1))
                self.layer_metrics[layer]['accuracy'].append(cm_data['accuracy'])
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error calculating confusion matrix for {layer}: {str(e)}")
    
    def generate_comprehensive_charts(self, session_id: str = None, phase_num: int = None) -> Dict[str, str]:
        """
        Generate comprehensive research-focused training charts and confusion matrices.
        
        Args:
            session_id: Unique identifier for this training session
            phase_num: Training phase number for research context
            
        Returns:
            Dictionary mapping chart types to file paths
        """
        if not VISUALIZATION_AVAILABLE:
            if self.verbose:
                self.logger.warning("⚠️ Visualization libraries not available. Skipping chart generation.")
            return {}
        
        session_id = session_id or f"training_{int(time.time())}"
        session_dir = self.save_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        generated_charts = {}
        
        try:
            # 1. Training curves
            training_curves_path = self.generate_training_curves_chart(session_id)
            if training_curves_path:
                generated_charts['training_curves'] = str(training_curves_path)
            
            # 2. Confusion matrices for each layer
            for layer in self.num_classes_per_layer.keys():
                cm_path = self.generate_confusion_matrix_chart(layer, session_id)
                if cm_path:
                    generated_charts[f'confusion_matrix_{layer}'] = str(cm_path)
            
            # 3. Per-layer metrics comparison
            layer_metrics_path = self._generate_layer_metrics_chart(session_dir)
            if layer_metrics_path:
                generated_charts['layer_metrics'] = str(layer_metrics_path)
            
            # 4. Research-focused phase analysis with hypothesis testing context
            phase_analysis_path = self._generate_research_phase_analysis_chart(session_dir, phase_num)
            if phase_analysis_path:
                generated_charts['research_phase_analysis'] = str(phase_analysis_path)
            
            # 5. Learning rate schedule
            lr_schedule_path = self._generate_lr_schedule_chart(session_dir)
            if lr_schedule_path:
                generated_charts['lr_schedule'] = str(lr_schedule_path)
            
            # 6. Research-focused comprehensive dashboard
            dashboard_path = self._generate_research_dashboard(session_dir, phase_num)
            if dashboard_path:
                generated_charts['research_dashboard'] = str(dashboard_path)
                
            # 7. Phase 2 specific: mAP@0.5 trends chart
            if phase_num == 2:
                map_trends_path = self._generate_map_trends_chart(session_dir)
                if map_trends_path:
                    generated_charts['detection_map_trends'] = str(map_trends_path)
            
            # Save metrics summary
            self.save_metrics_summary(session_dir, session_id)
            
            if self.verbose:
                self.logger.info(f"📊 Generated {len(generated_charts)} visualization charts")
                for chart_type, path in generated_charts.items():
                    self.logger.info(f"   • {chart_type}: {Path(path).name}")
            
            return generated_charts
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Error generating charts: {str(e)}")
            return {}
    
    @VisualizationHelper.generate_chart_with_error_handling
    def generate_confusion_matrix_chart(self, layer_name: str, session_id: str) -> Optional[str]:
        """
        Generate confusion matrix chart for a specific layer.
        
        Args:
            layer_name: Name of the layer (layer_1, layer_2, etc.)
            session_id: Unique identifier for this training session
            
        Returns:
            Path to generated chart image or None if generation failed
        """
        if not self.confusion_matrices[layer_name]:
            return None
            
        latest_cm_data = self.confusion_matrices[layer_name][-1]
        return self.chart_generator.generate_confusion_matrix_chart(
            latest_cm_data['matrix'],
            self.class_names[layer_name],
            latest_cm_data['accuracy'],
            session_id,
            layer_name
        )
    
    @VisualizationHelper.generate_chart_with_error_handling
    def _generate_layer_metrics_chart(self, session_dir: Path) -> Optional[str]:
        """Generate per-layer metrics comparison chart."""
        if not self.layer_metrics:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Layer Metrics Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Check if we're using global fallback (all layers have identical values)
        using_global_fallback = self._check_if_using_global_fallback()
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Define different line styles and markers for better visibility when overlapping
            layer_styles = {
                'layer_1': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
                'layer_2': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'}, 
                'layer_3': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'}
            }
            
            lines_plotted = 0
            for layer, layer_data in self.layer_metrics.items():
                metric_values = layer_data.get(metric, [])
                if metric_values:
                    epochs = range(1, len(metric_values) + 1)
                    style = layer_styles.get(layer, {'color': 'black', 'linestyle': '-', 'marker': 'o'})
                    
                    ax.plot(epochs, metric_values, 
                           label=layer, 
                           linewidth=2.5 if using_global_fallback else 2,
                           linestyle=style['linestyle'],
                           marker=style['marker'], 
                           markersize=6 if using_global_fallback else 4,
                           color=style['color'],
                           alpha=0.8)
                    lines_plotted += 1
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if lines_plotted > 0:
                ax.legend()
                ax.set_ylim(0, 1)
        
        chart_path = session_dir / 'layer_metrics_comparison.png'
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        return str(chart_path)
    
    def _check_if_using_global_fallback(self) -> bool:
        """
        Check if we're using global fallback metrics (all layers have identical values).
        
        Returns:
            True if all layers have identical metric values, False otherwise
        """
        try:
            if not self.layer_metrics:
                return False
            
            # Get all layers and check if they have identical values for each metric type
            layers = list(self.layer_metrics.keys())
            if len(layers) <= 1:
                return False
            
            metric_types = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric_type in metric_types:
                # Get values for each layer
                layer_values = []
                for layer in layers:
                    values = self.layer_metrics[layer].get(metric_type, [])
                    if values:
                        layer_values.append(values)
                
                # If we have values from multiple layers, check if they're identical
                if len(layer_values) > 1:
                    first_values = layer_values[0]
                    for other_values in layer_values[1:]:
                        if len(first_values) != len(other_values):
                            return False
                        # Check if values are approximately equal (within small tolerance)
                        for v1, v2 in zip(first_values, other_values):
                            if abs(v1 - v2) > 1e-6:  # Not identical
                                return False
                    
                    # If we reach here, this metric type has identical values across layers
                    # Continue checking other metric types
                    continue
            
            # If we've checked all metric types and found identical values, it's global fallback
            return True
            
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Error checking global fallback: {e}")
            return False
    
    def _generate_phase_analysis_chart(self, session_dir: Path) -> Optional[str]:
        """Generate phase transition analysis chart."""
        try:
            if not self.phase_transitions or not self.train_losses:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            epochs = range(1, len(self.train_losses) + 1)
            ax.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
            
            if self.val_losses:
                ax.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', linewidth=2, label='Validation Loss')
            
            # Mark phase transitions
            colors = ['orange', 'green', 'purple']
            for idx, transition in enumerate(self.phase_transitions):
                color = colors[idx % len(colors)]
                ax.axvline(x=transition['epoch'], color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax.text(transition['epoch'], max(self.train_losses) * 0.9, 
                       f"{transition['from_phase']} → {transition['to_phase']}", 
                       rotation=90, verticalalignment='bottom', fontsize=10, color=color)
            
            ax.set_title('Training Progress with Phase Transitions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            chart_path = session_dir / 'phase_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating phase analysis chart: {str(e)}")
            return None
    
    def _generate_lr_schedule_chart(self, session_dir: Path) -> Optional[str]:
        """Generate learning rate schedule chart."""
        try:
            if not self.learning_rates:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(self.learning_rates) + 1)
            ax.plot(epochs, self.learning_rates, 'g-', linewidth=2, marker='o', markersize=3)
            
            # Mark phase transitions
            for transition in self.phase_transitions:
                ax.axvline(x=transition['epoch'], color='orange', linestyle='--', alpha=0.7)
            
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            chart_path = session_dir / 'learning_rate_schedule.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating LR schedule chart: {str(e)}")
            return None
    
    def _generate_research_dashboard(self, session_dir: Path, phase_num: int = None) -> Optional[str]:
        """Generate a research-focused comprehensive training dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Research-focused title
            phase_context = self._get_dashboard_title_context(phase_num)
            fig.suptitle(f'SmartCash Research Dashboard - {phase_context}', fontsize=18, fontweight='bold')
            
            # Training curves (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            if self.train_losses:
                epochs = range(1, len(self.train_losses) + 1)
                ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if self.val_losses:
                ax1.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Research-focused metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            research_trends = self._extract_all_research_trends()
            if research_trends:
                for metric_name, values in research_trends.items():
                    if values and len(values) > 0:
                        epochs = range(1, len(values) + 1)
                        ax2.plot(epochs, values, label=self._get_metric_display_name(metric_name), linewidth=2)
                ax2.set_title('Research-Focused Metrics Trends')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
            else:
                # Fallback to layer metrics if research trends not available
                for layer, layer_data in self.layer_metrics.items():
                    accuracy_values = layer_data.get('accuracy', [])
                    if accuracy_values:
                        epochs = range(1, len(accuracy_values) + 1)
                        ax2.plot(epochs, accuracy_values, label=f'{layer} Accuracy', linewidth=2)
                ax2.set_title('Per-Layer Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Confusion matrices (middle row)
            cm_positions = [gs[1, 0], gs[1, 1], gs[1, 2]]
            for idx, (layer, position) in enumerate(zip(self.num_classes_per_layer.keys(), cm_positions)):
                if layer in self.confusion_matrices and self.confusion_matrices[layer]:
                    ax = fig.add_subplot(position)
                    latest_cm = self.confusion_matrices[layer][-1]['matrix']
                    sns.heatmap(latest_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                    ax.set_title(f'CM - {layer.upper()}')
            
            # Phase transitions and summary stats (bottom)
            ax3 = fig.add_subplot(gs[2, :])
            if self.train_losses:
                epochs = range(1, len(self.train_losses) + 1)
                ax3.plot(epochs, self.train_losses, 'b-', alpha=0.7, linewidth=1)
                
                # Mark phase transitions
                for transition in self.phase_transitions:
                    ax3.axvline(x=transition['epoch'], color='red', linestyle='--', linewidth=2)
                    ax3.text(transition['epoch'], max(self.train_losses) * 0.8, 
                           f"Phase {transition['from_phase']} → {transition['to_phase']}", 
                           rotation=90, verticalalignment='bottom', fontsize=8)
            
            ax3.set_title('Training Timeline with Phase Transitions')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Training Loss')
            ax3.grid(True, alpha=0.3)
            
            # Add research-focused summary text
            summary_text = self._generate_research_summary_text()
            fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            chart_path = session_dir / f'research_dashboard_phase_{phase_num or "unknown"}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating research dashboard: {str(e)}")
            return None
    
    def _generate_research_phase_analysis_chart(self, session_dir: Path, phase_num: int = None) -> Optional[str]:
        """Generate research-focused phase transition analysis chart."""
        try:
            if not self.phase_transitions or not self.train_losses:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Top plot: Loss curves with research context
            ax1.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
            if self.val_losses:
                ax1.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', linewidth=2, label='Validation Loss')
            
            # Mark phase transitions with research context
            colors = ['orange', 'green', 'purple']
            for idx, transition in enumerate(self.phase_transitions):
                color = colors[idx % len(colors)]
                ax1.axvline(x=transition['epoch'], color=color, linestyle='--', linewidth=2, alpha=0.8)
                
                # Add research context to transition labels
                context = self._get_transition_research_context(transition)
                ax1.text(transition['epoch'], max(self.train_losses) * 0.9, 
                       context, rotation=90, verticalalignment='bottom', fontsize=10, color=color)
            
            ax1.set_title('Research Training Progress with Phase Transitions', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Research metrics progression
            research_trends = self._extract_research_accuracy_trends()
            for metric_name, values in research_trends.items():
                if values:
                    metric_epochs = range(1, len(values) + 1)
                    display_name = self._get_metric_display_name(metric_name)
                    ax2.plot(metric_epochs, values, linewidth=2, marker='o', markersize=3, label=display_name)
            
            # Mark phase transitions on metrics plot too
            for transition in self.phase_transitions:
                ax2.axvline(x=transition['epoch'], color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_title('Research Metrics Progression', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy / Performance')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = session_dir / f'research_phase_analysis_phase_{phase_num or "unknown"}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating research phase analysis chart: {str(e)}")
            return None
    
    def _generate_map_trends_chart(self, session_dir: Path) -> Optional[str]:
        """Generate mAP@0.5 trends chart for Phase 2 additional detection information."""
        try:
            # Extract mAP trends
            map_values = []
            for epoch_data in self.epoch_metrics:
                map_val = epoch_data.get('val_detection_map50')
                if map_val is not None and map_val > 0:
                    map_values.append(map_val)
            
            if not map_values:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(map_values) + 1)
            ax.plot(epochs, map_values, 'g-', linewidth=2, marker='o', markersize=4, label='Detection mAP@0.5')
            
            # Add trend line
            if len(map_values) > 1:
                z = np.polyfit(epochs, map_values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
            
            ax.set_title('Phase 2 Additional Detection Information: mAP@0.5 Trends', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP@0.5')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)  # mAP is between 0 and 1
            
            chart_path = session_dir / 'detection_map50_trends.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating mAP trends chart: {str(e)}")
            return None
    
    def _get_dashboard_title_context(self, phase_num: int = None) -> str:
        """Get dashboard title context based on phase."""
        if phase_num == 1:
            return "Phase 1: Single-Layer Denomination Detection"
        elif phase_num == 2:
            return "Phase 2: Multi-Layer Hierarchical Detection"
        else:
            return "Multi-Layer Banknote Detection Research"
    
    def _get_transition_research_context(self, transition: Dict) -> str:
        """Get research context for phase transitions."""
        from_phase = transition.get('from_phase', '')
        to_phase = transition.get('to_phase', '')
        
        if 'phase_1' in to_phase or '1' in to_phase:
            return 'Start\nDenomination\nDetection'
        elif 'phase_2' in to_phase or '2' in to_phase:
            return 'Begin\nHierarchical\nApproach'
        else:
            return f'{from_phase}\n→\n{to_phase}'
    
    def _get_research_context(self, phase_num: int) -> str:
        """Get research context description for the given phase."""
        if phase_num == 1:
            return "single_layer_denomination_detection"
        elif phase_num == 2:
            return "multi_layer_hierarchical_detection"
        else:
            return "unknown_phase"
    
    def _extract_research_accuracy_trends(self) -> Dict[str, List[float]]:
        """Extract research-focused accuracy trends from epoch metrics."""
        return ChartUtils.extract_research_accuracy_trends(self.epoch_metrics)

    def _extract_secondary_metrics_trends(self) -> Dict[str, List[float]]:
        """Extract secondary research metrics trends."""
        return ChartUtils.extract_secondary_metrics_trends(self.epoch_metrics)

    def _extract_all_research_trends(self) -> Dict[str, List[float]]:
        """Extract all available research trends for comprehensive visualization."""
        return ChartUtils.extract_all_research_trends(self.epoch_metrics)

    def _get_metric_display_name(self, metric_name: str) -> str:
        """Convert internal metric names to user-friendly display names."""
        return self.chart_generator._get_metric_display_name(metric_name)

    def _extract_research_summary_metrics(self) -> str:
        """Extract key research metrics for summary display."""
        return ChartUtils.get_research_summary_metrics(self.epoch_metrics)
    
    def save_metrics_summary(self, session_dir: Path, session_id: str):
        """
        Save comprehensive research-focused metrics summary to JSON.
        
        Args:
            session_dir: Directory to save the summary
            session_id: Unique identifier for this training session
        """
        self.chart_generator.save_metrics_summary(
            session_dir,
            session_id,
            self.epoch_metrics,
            self.phase_transitions,
            self.layer_metrics,
            self.confusion_matrices
        )
    

    
    def _get_contextual_chart_title(self) -> str:
        """Generate contextual chart title based on training configuration."""
        # Determine backbone and phase info
        backbone = "Unknown Backbone"
        total_epochs = len(self.epoch_metrics)
        
        # Try to extract backbone from saved metrics
        if self.epoch_metrics:
            # Look for backbone info in metrics or derive from class configuration
            for record in self.epoch_metrics:
                if 'backbone' in record:
                    backbone = record['backbone'].upper()
                    break
        
        # Determine phase information
        phases_used = set()
        for record in self.epoch_metrics:
            if 'phase_num' in record:
                phases_used.add(record['phase_num'])
        
        if len(phases_used) > 1:
            phase_info = f"Two-Phase Training ({min(phases_used)} → {max(phases_used)})"
        elif phases_used:
            phase_info = f"Single-Phase Training (Phase {list(phases_used)[0]})"
        else:
            phase_info = "Training Progress"
        
        return f"SmartCash {backbone} - {phase_info} ({total_epochs} epochs)"
    
    def _plot_enhanced_loss_curves(self, ax, epochs):
        """Plot loss curves with enhanced scaling for small decreasing values."""
        has_data = False
        
        if self.train_losses:
            ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
            has_data = True
            
        if self.val_losses:
            ax.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
            has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Loss Data Available', transform=ax.transAxes, ha='center', va='center', fontsize=12)
            return
            
        # Enhanced scaling for small decreasing values
        all_losses = []
        if self.train_losses:
            all_losses.extend(self.train_losses)
        if self.val_losses:
            all_losses.extend(self.val_losses)
        
        if all_losses:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            loss_range = max_loss - min_loss
            
            # If loss range is very small (< 0.1), use linear scale with padding
            # If loss range is moderate, use linear scale
            # If loss range is large, consider log scale
            if loss_range < 0.1:
                # Small decreasing losses - add padding to show curve clearly
                padding = loss_range * 0.1 if loss_range > 0 else 0.01
                ax.set_ylim(min_loss - padding, max_loss + padding)
            elif min_loss > 0 and max_loss / min_loss > 10:
                # Large range - use log scale
                ax.set_yscale('log')
        
        # Styling
        ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add best loss annotations
        if self.val_losses:
            best_val_epoch = np.argmin(self.val_losses) + 1
            best_val_loss = min(self.val_losses)
            ax.annotate(f'Best Val: {best_val_loss:.4f}@{best_val_epoch}', 
                       xy=(best_val_epoch, best_val_loss),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=10)
    
    def _plot_loss_breakdown_chart(self, ax, epochs):
        """Plot detailed loss breakdown components."""
        # Extract loss breakdown data from epoch metrics
        loss_components = {}
        
        # Look for loss breakdown components with correct field names
        loss_breakdown_fields = [
            'train_box_loss', 'train_obj_loss', 'train_cls_loss',
            'val_box_loss', 'val_obj_loss', 'val_cls_loss'
        ]
        
        for record in self.epoch_metrics:
            for field in loss_breakdown_fields:
                if field in record and isinstance(record[field], (int, float)) and record[field] is not None:
                    if field not in loss_components:
                        loss_components[field] = []
                    loss_components[field].append(record[field])
            
            # Also check for legacy loss breakdown fields as fallback
            for key, value in record.items():
                if 'loss' in key.lower() and key not in ['train_loss', 'val_loss', 'total_loss'] and isinstance(value, (int, float)) and value is not None:
                    if key not in loss_components:
                        loss_components[key] = []
                    loss_components[key].append(value)
        
        if not loss_components:
            ax.text(0.5, 0.5, 'No Loss Breakdown Data\nAvailable', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            ax.set_title('Loss Breakdown Components', fontsize=14, fontweight='bold')
            return
        
        # Plot each loss component
        colors = plt.cm.Set3(np.linspace(0, 1, len(loss_components)))
        
        for (component, values), color in zip(loss_components.items(), colors):
            if len(values) > 0:
                # Ensure we have correct epochs for this component
                # Use only the number of epochs that match the values length
                component_epochs = list(range(1, len(values) + 1))
                ax.plot(component_epochs, values, label=self._format_loss_component_name(component), 
                       linewidth=2, marker='o', markersize=3, color=color, alpha=0.8)
        
        ax.set_title('Loss Breakdown Components', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_merged_research_metrics(self, ax, epochs):
        """Plot merged research accuracy trends and secondary metrics."""
        # Merge accuracy trends and secondary metrics
        research_data = self._extract_research_accuracy_trends()
        secondary_data = self._extract_secondary_metrics_trends()
        
        all_metrics = {}
        all_metrics.update(research_data)
        all_metrics.update(secondary_data)
        
        if not all_metrics:
            ax.text(0.5, 0.5, 'No Research Metrics\nData Available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            ax.set_title('Research Metrics Trends', fontsize=14, fontweight='bold')
            return
        
        # Plot metrics with different styles for accuracy vs other metrics
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
        
        for (metric_name, values), color in zip(all_metrics.items(), colors):
            if values and len(values) > 0:
                metric_epochs = epochs[:len(values)]
                
                # Use different styles for different metric types
                if 'accuracy' in metric_name.lower():
                    linestyle = '-'
                    marker = 'o'
                    linewidth = 2.5
                elif 'precision' in metric_name.lower() or 'recall' in metric_name.lower():
                    linestyle = '--'
                    marker = 's'
                    linewidth = 2
                elif 'f1' in metric_name.lower():
                    linestyle = '-.'
                    marker = 'D'
                    linewidth = 2
                else:
                    linestyle = ':'
                    marker = '^'
                    linewidth = 1.5
                
                ax.plot(metric_epochs, values, 
                       label=self._get_metric_display_name(metric_name),
                       linestyle=linestyle, marker=marker, markersize=3,
                       linewidth=linewidth, color=color, alpha=0.8)
        
        ax.set_title('Research Metrics Trends', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_ylim(0, 1.05)  # Most research metrics are 0-1
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _format_loss_component_name(self, component_name: str) -> str:
        """Format loss component names for display."""
        # Map component names to display names
        display_names = {
            'train_box_loss': 'Train Box Loss',
            'train_obj_loss': 'Train Objectness',
            'train_cls_loss': 'Train Classification',
            'val_box_loss': 'Val Box Loss',
            'val_obj_loss': 'Val Objectness',
            'val_cls_loss': 'Val Classification',
            # Legacy support
            'train_bbox_loss': 'Train Bounding Box',
            'val_bbox_loss': 'Val Bounding Box'
        }
        
        # Use direct mapping if available
        if component_name in display_names:
            return display_names[component_name]
        
        # Fallback to generic formatting
        name = component_name.replace('train_', '').replace('val_', '')
        name = name.replace('_loss', '').replace('loss', '')
        name = name.replace('_', ' ').title()
        
        # Handle special cases
        replacements = {
            'Box': 'Box Loss',
            'Obj': 'Objectness',
            'Cls': 'Classification', 
            'Bbox': 'Bounding Box',
            'Total': 'Total Loss'
        }
        
        for old, new in replacements.items():
            if old in name:
                name = name.replace(old, new)
        
        return name


def create_visualization_manager(num_classes_per_layer: Dict[str, int], 
                               class_names: Dict[str, List[str]] = None,
                               save_dir: str = "data/visualization",
                               verbose: bool = False) -> VisualizationMetricsTracker:
    """
    Factory function to create a research-focused visualization manager for training.
    
    This creates a visualization manager optimized for SmartCash research with:
    - Research-focused metric tracking (denomination accuracy, hierarchical performance)
    - Phase-appropriate visualizations (Phase 1: single-layer, Phase 2: multi-layer + mAP)
    - Backward compatibility for existing UI components
    
    Args:
        num_classes_per_layer: Dictionary mapping layer names to number of classes
        class_names: Optional mapping of layer names to class name lists
        save_dir: Directory to save visualization outputs
        verbose: Enable verbose logging
        
    Returns:
        Configured VisualizationMetricsTracker instance with research focus
    """
    tracker = VisualizationMetricsTracker(
        num_classes_per_layer=num_classes_per_layer,
        class_names=class_names,
        save_dir=save_dir,
        verbose=verbose
    )
    
    if verbose:
        # Use the tracker's logger for consistency
        tracker.logger.info("🔬 Research-focused visualization manager created")
        tracker.logger.info("   • Phase 1 focus: Single-layer denomination detection")
        tracker.logger.info("   • Phase 2 focus: Multi-layer hierarchical detection + mAP@0.5")
        tracker.logger.info("   • Backward compatibility: Legacy UI metrics supported")
    
    return tracker