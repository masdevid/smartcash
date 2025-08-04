#!/usr/bin/env python3
"""
File: smartcash/visualization/training_charts.py

Training metrics visualization using JSON metrics history.
Creates per-phase charts and combined views for training analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Set style for better-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingChartsVisualizer:
    """
    Creates training visualization charts from JSON metrics history.
    
    Provides per-phase and combined views for training analysis.
    """
    
    def __init__(self, metrics_file: str, output_dir: str = "charts"):
        """
        Initialize visualizer with metrics file.
        
        Args:
            metrics_file: Path to JSON metrics history file
            output_dir: Directory to save chart images
        """
        self.metrics_file = Path(metrics_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_data = []
        self.phase_data = {}
        
        self._load_metrics()
        logger.info(f"Loaded {len(self.metrics_data)} epoch records")
    
    def _load_metrics(self):
        """Load metrics data from JSON file."""
        try:
            if not self.metrics_file.exists():
                logger.error(f"Metrics file not found: {self.metrics_file}")
                return
            
            with open(self.metrics_file, 'r') as f:
                self.metrics_data = json.load(f)
            
            # Group by phase
            for record in self.metrics_data:
                phase = record['phase']
                if phase not in self.phase_data:
                    self.phase_data[phase] = []
                self.phase_data[phase].append(record)
            
            logger.info(f"Loaded data for phases: {list(self.phase_data.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            self.metrics_data = []
            self.phase_data = {}
    
    def create_loss_charts(self, save: bool = True) -> Dict[str, Any]:
        """
        Create enhanced loss progression charts per phase with improved scaling.
        
        Args:
            save: Whether to save charts to disk
            
        Returns:
            Dictionary with chart data and file paths
        """
        if not self.metrics_data:
            logger.warning("No metrics data available")
            return {}
        
        # Create contextual title
        contextual_title = self._get_contextual_title()
        
        # Create figure with subplots for each phase
        num_phases = len(self.phase_data)
        fig, axes = plt.subplots(1, num_phases, figsize=(8*num_phases, 6))
        if num_phases == 1:
            axes = [axes]
        
        fig.suptitle(contextual_title, fontsize=16, fontweight='bold')
        
        chart_data = {}
        
        for i, (phase, data) in enumerate(self.phase_data.items()):
            ax = axes[i]
            
            # Extract data
            epochs = [record['epoch'] for record in data]
            train_losses = [record['train_loss'] for record in data]
            val_losses = [record['val_loss'] for record in data]
            
            # Enhanced loss plotting with better scaling
            self._plot_enhanced_loss_curves(ax, epochs, train_losses, val_losses, phase)
            
            chart_data[f'phase_{phase}'] = {
                'epochs': epochs,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_val_loss': min(val_losses),
                'best_val_epoch': epochs[np.argmin(val_losses)]
            }
        
        plt.tight_layout()
        
        # Save chart
        if save:
            chart_path = self.output_dir / 'loss_progression.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Enhanced loss chart saved to: {chart_path}")
            chart_data['file_path'] = str(chart_path)
        
        plt.show()
        return chart_data
    
    def create_map_charts(self, save: bool = True) -> Dict[str, Any]:
        """
        Create mAP progression charts per phase.
        
        Args:
            save: Whether to save charts to disk
            
        Returns:
            Dictionary with chart data and file paths
        """
        if not self.metrics_data:
            logger.warning("No metrics data available")
            return {}
        
        # Check if mAP data exists
        has_map_data = any(record.get('val_map50') is not None for record in self.metrics_data)
        if not has_map_data:
            logger.warning("No mAP data found in metrics")
            return {}
        
        # Create figure
        num_phases = len(self.phase_data)
        fig, axes = plt.subplots(1, num_phases, figsize=(6*num_phases, 5))
        if num_phases == 1:
            axes = [axes]
        
        chart_data = {}
        
        for i, (phase, data) in enumerate(self.phase_data.items()):
            ax = axes[i]
            
            # Extract mAP data (only non-None values)
            map_data = [(record['epoch'], record['val_map50']) 
                       for record in data if record.get('val_map50') is not None]
            
            if not map_data:
                ax.text(0.5, 0.5, f'No mAP data\nfor Phase {phase}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax.set_title(f'Phase {phase} - mAP@0.5', fontsize=14, fontweight='bold')
                continue
            
            epochs, map_values = zip(*map_data)
            
            # Plot mAP
            ax.plot(epochs, map_values, 'g-', label='mAP@0.5', linewidth=2, marker='D', markersize=4)
            
            # Add precision/recall if available
            precision_data = [(record['epoch'], record['val_precision']) 
                            for record in data if record.get('val_precision') is not None]
            recall_data = [(record['epoch'], record['val_recall']) 
                          for record in data if record.get('val_recall') is not None]
            
            if precision_data:
                p_epochs, p_values = zip(*precision_data)
                ax.plot(p_epochs, p_values, 'b--', label='Precision', linewidth=1.5, alpha=0.7)
            
            if recall_data:
                r_epochs, r_values = zip(*recall_data)
                ax.plot(r_epochs, r_values, 'r--', label='Recall', linewidth=1.5, alpha=0.7)
            
            # Styling
            ax.set_title(f'Phase {phase} - mAP@0.5 & Metrics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            
            # Add annotations for best mAP
            best_map_epoch = epochs[np.argmax(map_values)]
            best_map_value = max(map_values)
            ax.annotate(f'Best mAP: {best_map_value:.4f}', 
                       xy=(best_map_epoch, best_map_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            chart_data[f'phase_{phase}'] = {
                'epochs': list(epochs),
                'map50': list(map_values),
                'best_map50': best_map_value,
                'best_map_epoch': best_map_epoch
            }
        
        plt.tight_layout()
        
        # Save chart
        if save:
            chart_path = self.output_dir / 'map_progression.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"mAP chart saved to: {chart_path}")
            chart_data['file_path'] = str(chart_path)
        
        plt.show()
        return chart_data
    
    def create_learning_rate_chart(self, save: bool = True) -> Dict[str, Any]:
        """
        Create learning rate progression chart across all phases.
        
        Args:
            save: Whether to save chart to disk
            
        Returns:
            Dictionary with chart data and file path
        """
        if not self.metrics_data:
            logger.warning("No metrics data available")
            return {}
        
        # Extract learning rate data
        epochs = [record['epoch'] for record in self.metrics_data]
        lr_values = [record.get('learning_rate', 0) for record in self.metrics_data]
        phases = [record['phase'] for record in self.metrics_data]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # Plot learning rate with phase colors
        phase_colors = ['blue', 'red', 'green']
        for phase in set(phases):
            phase_epochs = [e for e, p in zip(epochs, phases) if p == phase]
            phase_lrs = [lr for lr, p in zip(lr_values, phases) if p == phase]
            
            ax.plot(phase_epochs, phase_lrs, 
                   color=phase_colors[phase-1], label=f'Phase {phase}',
                   linewidth=2, marker='o', markersize=3)
        
        # Styling
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        chart_data = {
            'epochs': epochs,
            'learning_rates': lr_values,
            'phases': phases
        }
        
        plt.tight_layout()
        
        # Save chart
        if save:
            chart_path = self.output_dir / 'learning_rate.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning rate chart saved to: {chart_path}")
            chart_data['file_path'] = str(chart_path)
        
        plt.show()
        return chart_data
    
    def create_phase_comparison(self, save: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive phase comparison dashboard.
        
        Args:
            save: Whether to save chart to disk
            
        Returns:
            Dictionary with comparison data and file path
        """
        if len(self.phase_data) < 2:
            logger.warning("Need at least 2 phases for comparison")
            return {}
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        comparison_data = {}
        
        # 1. Loss comparison
        for phase, data in self.phase_data.items():
            epochs = [record['epoch'] for record in data]
            train_losses = [record['train_loss'] for record in data]
            val_losses = [record['val_loss'] for record in data]
            
            ax1.plot(epochs, train_losses, label=f'Phase {phase} Train', linewidth=2)
            ax1.plot(epochs, val_losses, '--', label=f'Phase {phase} Val', linewidth=2)
        
        ax1.set_title('Loss Comparison Across Phases', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. mAP comparison
        for phase, data in self.phase_data.items():
            map_data = [(record['epoch'], record['val_map50']) 
                       for record in data if record.get('val_map50') is not None]
            if map_data:
                epochs, map_values = zip(*map_data)
                ax2.plot(epochs, map_values, label=f'Phase {phase} mAP@0.5', 
                        linewidth=2, marker='o', markersize=3)
        
        ax2.set_title('mAP@0.5 Comparison', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP@0.5')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # 3. Learning rate progression
        all_epochs = [record['epoch'] for record in self.metrics_data]
        all_lrs = [record.get('learning_rate', 0) for record in self.metrics_data]
        all_phases = [record['phase'] for record in self.metrics_data]
        
        phase_colors = ['blue', 'red', 'green']
        for phase in set(all_phases):
            phase_epochs = [e for e, p in zip(all_epochs, all_phases) if p == phase]
            phase_lrs = [lr for lr, p in zip(all_lrs, all_phases) if p == phase]
            
            ax3.plot(phase_epochs, phase_lrs, 
                    color=phase_colors[phase-1], label=f'Phase {phase}',
                    linewidth=2)
        
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Final metrics comparison (bar chart)
        final_metrics = {}
        for phase, data in self.phase_data.items():
            final_record = data[-1]  # Last epoch of phase
            final_metrics[f'Phase {phase}'] = {
                'Final Val Loss': final_record['val_loss'],
                'Best mAP@0.5': max([r.get('val_map50', 0) for r in data if r.get('val_map50') is not None], default=0),
                'Final Precision': final_record.get('val_precision', 0),
                'Final Recall': final_record.get('val_recall', 0)
            }
        
        # Bar chart for final metrics
        phases = list(final_metrics.keys())
        metrics = ['Final Val Loss', 'Best mAP@0.5', 'Final Precision', 'Final Recall']
        x = np.arange(len(phases))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [final_metrics[phase].get(metric, 0) for phase in phases]
            ax4.bar(x + i*width, values, width, label=metric)
        
        ax4.set_title('Final Metrics Comparison', fontweight='bold')
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x + width*1.5)
        ax4.set_xticklabels(phases)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        comparison_data['final_metrics'] = final_metrics
        comparison_data['phases'] = list(self.phase_data.keys())
        
        plt.tight_layout()
        
        # Save chart
        if save:
            chart_path = self.output_dir / 'phase_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"Phase comparison chart saved to: {chart_path}")
            comparison_data['file_path'] = str(chart_path)
        
        plt.show()
        return comparison_data
    
    def generate_all_charts(self, save: bool = True) -> Dict[str, Any]:
        """
        Generate all available charts and return summary.
        
        Args:
            save: Whether to save charts to disk
            
        Returns:
            Dictionary with all chart data and file paths
        """
        logger.info("Generating all training charts...")
        
        results = {}
        
        # Generate individual charts
        results['loss_charts'] = self.create_loss_charts(save)
        results['map_charts'] = self.create_map_charts(save)
        results['learning_rate_chart'] = self.create_learning_rate_chart(save)
        
        # Generate comparison if multiple phases
        if len(self.phase_data) > 1:
            results['phase_comparison'] = self.create_phase_comparison(save)
        
        # Create summary
        results['summary'] = {
            'total_epochs': len(self.metrics_data),
            'phases': list(self.phase_data.keys()),
            'charts_generated': len([r for r in results.values() if r and 'file_path' in r]),
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"Generated {results['summary']['charts_generated']} charts in {self.output_dir}")
        return results
    
    def _get_contextual_title(self) -> str:
        """Generate contextual chart title based on training data."""
        if not self.metrics_data:
            return "SmartCash Training Progress"
        
        # Extract training context
        first_record = self.metrics_data[0]
        last_record = self.metrics_data[-1]
        
        # Try to determine backbone and configuration
        backbone = "Unknown Backbone"
        total_epochs = len(self.metrics_data)
        
        # Determine phase information
        phases_used = set(record['phase'] for record in self.metrics_data)
        
        if len(phases_used) > 1:
            phase_info = f"Two-Phase Training ({min(phases_used)} → {max(phases_used)})"
        elif phases_used:
            phase_info = f"Single-Phase Training (Phase {list(phases_used)[0]})"
        else:
            phase_info = "Training Progress"
        
        return f"SmartCash {backbone} - {phase_info} ({total_epochs} epochs)"
    
    def _plot_enhanced_loss_curves(self, ax, epochs, train_losses, val_losses, phase):
        """Plot loss curves with enhanced scaling for small decreasing values."""
        # Plot losses with enhanced styling
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
        
        # Enhanced scaling for small decreasing values
        all_losses = train_losses + val_losses
        if all_losses:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            loss_range = max_loss - min_loss
            
            # If loss range is very small (< 0.1), use linear scale with padding
            if loss_range < 0.1:
                # Small decreasing losses - add padding to show curve clearly
                padding = loss_range * 0.15 if loss_range > 0 else 0.01
                ax.set_ylim(min_loss - padding, max_loss + padding)
            elif min_loss > 0 and max_loss / min_loss > 10:
                # Large range - use log scale
                ax.set_yscale('log')
        
        # Enhanced styling
        ax.set_title(f'Phase {phase} - Loss Progression', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add best validation loss annotation
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = min(val_losses)
        ax.annotate(f'Best Val: {best_val_loss:.4f}@{epochs[best_val_epoch]}', 
                   xy=(epochs[best_val_epoch], best_val_loss),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                   fontsize=10)
    
    def create_loss_breakdown_charts(self, save: bool = True) -> Dict[str, Any]:
        """
        Create detailed loss breakdown charts showing individual loss components.
        
        Args:
            save: Whether to save charts to disk
            
        Returns:
            Dictionary with chart data and file paths
        """
        if not self.metrics_data:
            logger.warning("No metrics data available")
            return {}
        
        # Extract loss breakdown components
        loss_components = self._extract_loss_breakdown_data()
        if not loss_components:
            logger.warning("No loss breakdown data found")
            return {}
        
        # Create figure for loss breakdown
        num_phases = len(self.phase_data)
        fig, axes = plt.subplots(num_phases, 1, figsize=(12, 6*num_phases))
        if num_phases == 1:
            axes = [axes]
        
        fig.suptitle('Loss Breakdown Analysis', fontsize=16, fontweight='bold')
        
        chart_data = {}
        
        for i, (phase, data) in enumerate(self.phase_data.items()):
            ax = axes[i]
            
            # Extract epochs for this phase
            epochs = [record['epoch'] for record in data]
            
            # Plot loss components for this phase
            colors = plt.cm.Set3(np.linspace(0, 1, len(loss_components)))
            
            for (component, values), color in zip(loss_components.items(), colors):
                if len(values) >= len(epochs):
                    phase_values = values[:len(epochs)]
                    ax.plot(epochs, phase_values, label=self._format_loss_component_name(component), 
                           linewidth=2, marker='o', markersize=3, color=color, alpha=0.8)
            
            ax.set_title(f'Phase {phase} - Loss Components', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            chart_data[f'phase_{phase}_breakdown'] = {
                'epochs': epochs,
                'components': {comp: values[:len(epochs)] for comp, values in loss_components.items()}
            }
        
        plt.tight_layout()
        
        # Save chart
        if save:
            chart_path = self.output_dir / 'loss_breakdown.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Loss breakdown chart saved to: {chart_path}")
            chart_data['file_path'] = str(chart_path)
        
        plt.show()
        return chart_data
    
    def _extract_loss_breakdown_data(self) -> Dict[str, List[float]]:
        """Extract loss breakdown data from metrics records."""
        loss_components = {}
        
        for record in self.metrics_data:
            for key, value in record.items():
                if ('loss' in key.lower() and key not in ['train_loss', 'val_loss'] and 
                    isinstance(value, (int, float))):
                    if key not in loss_components:
                        loss_components[key] = []
                    loss_components[key].append(value)
        
        return loss_components
    
    def _format_loss_component_name(self, component_name: str) -> str:
        """Format loss component names for display."""
        # Remove common prefixes and format nicely
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
    
    def create_all_charts(self, save: bool = True) -> Dict[str, Any]:
        """
        Create all available enhanced charts and return comprehensive results.
        
        Args:
            save: Whether to save charts to disk
            
        Returns:
            Dictionary with all chart data and file paths
        """
        results = {}
        
        # 1. Enhanced loss progression charts
        loss_results = self.create_loss_charts(save)
        if loss_results:
            results['loss_charts'] = loss_results
        
        # 2. New: Loss breakdown charts
        breakdown_results = self.create_loss_breakdown_charts(save)
        if breakdown_results:
            results['loss_breakdown_charts'] = breakdown_results
        
        # 3. mAP progression charts
        map_results = self.create_map_charts(save)
        if map_results:
            results['map_charts'] = map_results
        
        # 4. Learning rate schedule
        lr_results = self.create_learning_rate_chart(save)
        if lr_results:
            results['learning_rate_chart'] = lr_results
        
        # 5. Comprehensive training dashboard
        dashboard_results = self.create_training_dashboard(save)
        if dashboard_results:
            results['training_dashboard'] = dashboard_results
        
        # Summary
        results['summary'] = {
            'total_epochs': len(self.metrics_data),
            'phases': list(self.phase_data.keys()),
            'charts_generated': len([r for r in results.values() if r and 'file_path' in r]),
            'output_directory': str(self.output_dir),
            'enhancements': [
                'Enhanced loss scaling for small decreasing values',
                'Contextual chart titles based on training configuration', 
                'New loss breakdown component charts',
                'Merged research metrics visualization'
            ]
        }
        
        logger.info(f"Generated {results['summary']['charts_generated']} enhanced charts in {self.output_dir}")
        return results


def create_training_visualizer(metrics_file: str, output_dir: str = "charts") -> TrainingChartsVisualizer:
    """
    Factory function to create training charts visualizer.
    
    Args:
        metrics_file: Path to JSON metrics history file
        output_dir: Directory to save chart images
        
    Returns:
        TrainingChartsVisualizer instance
    """
    return TrainingChartsVisualizer(metrics_file, output_dir)


def visualize_latest_training(training_logs_dir: str = "logs/training", 
                            output_dir: str = "charts") -> Optional[Dict[str, Any]]:
    """
    Automatically visualize the latest training session.
    
    Args:
        training_logs_dir: Directory containing training logs
        output_dir: Directory to save charts
        
    Returns:
        Visualization results or None if no data found
    """
    try:
        logs_path = Path(training_logs_dir)
        if not logs_path.exists():
            logger.error(f"Training logs directory not found: {logs_path}")
            return None
        
        # Find latest metrics file
        latest_file = logs_path / "latest_metrics.json"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
                metrics_file = latest_data['file_paths']['metrics']
        else:
            # Fallback: find most recent metrics file
            metrics_files = list(logs_path.glob("metrics_history_*.json"))
            if not metrics_files:
                logger.error("No metrics files found")
                return None
            metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"Visualizing metrics from: {metrics_file}")
        
        # Create visualizer and generate charts
        visualizer = create_training_visualizer(str(metrics_file), output_dir)
        results = visualizer.generate_all_charts(save=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to visualize latest training: {e}")
        return None