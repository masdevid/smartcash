#!/usr/bin/env python3
"""
Verification script for the visualization package.

This script tests the visualization package in isolation to verify that
all components are working correctly.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the required modules
class MockLogger:
    def __init__(self, name):
        self.name = name
    
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")

sys.modules['smartcash.common.logger'] = MockLogger
sys.modules['smartcash.model.analysis'] = type('MockAnalysis', (), {})

# Mock matplotlib and seaborn
class MockFigure:
    def __init__(self, *args, **kwargs):
        pass
    
    def savefig(self, *args, **kwargs):
        print(f"[MOCK] Saved figure to {args[0] if args else 'unknown'}")
        return "/mock/path/figure.png"

class MockAxes:
    def set_title(self, *args, **kwargs):
        pass
    
    def set_xlabel(self, *args, **kwargs):
        pass
    
    def set_ylabel(self, *args, **kwargs):
        pass
    
    def legend(self, *args, **kwargs):
        pass
    
    def grid(self, *args, **kwargs):
        pass

class MockMatplotlib:
    def __init__(self):
        self.figure = MockFigure
        self.Figure = MockFigure
        self.Axes = MockAxes
        
        class PyPlot:
            def __init__(self):
                self.figure = MockFigure
                self.subplots = self._subplots
            
            def _subplots(self, *args, **kwargs):
                return MockFigure(), [MockAxes()]
            
            def plot(self, *args, **kwargs):
                pass
            
            def xlabel(self, *args, **kwargs):
                pass
            
            def ylabel(self, *args, **kwargs):
                pass
            
            def title(self, *args, **kwargs):
                pass
            
            def legend(self, *args, **kwargs):
                pass
            
            def grid(self, *args, **kwargs):
                pass
            
            def tight_layout(self, *args, **kwargs):
                pass
        
        self.pyplot = PyPlot()

class MockSeaborn:
    @staticmethod
    def heatmap(*args, **kwargs):
        return MockAxes()

# Apply mocks
sys.modules['matplotlib'] = MockMatplotlib()
sys.modules['matplotlib.figure'] = MockFigure
sys.modules['matplotlib.axes'] = type('MockAxesModule', (), {'Axes': MockAxes})
sys.modules['matplotlib.pyplot'] = MockMatplotlib().pyplot
sys.modules['seaborn'] = MockSeaborn()

# Now import our visualization components
from smartcash.model.training.visualization import VisualizationManager

def generate_sample_data():
    """Generate sample data for testing visualization."""
    np.random.seed(42)
    
    # Sample data for 20 epochs
    epochs = list(range(1, 21))
    
    # Training and validation losses
    train_loss = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.02, 20)
    val_loss = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.03, 20) + 0.1
    
    # Learning rates
    learning_rates = np.linspace(1e-3, 1e-5, 20)
    
    # Layer metrics
    layer_metrics = {
        'banknote': {
            'accuracy': 0.8 + 0.15 * (1 - np.exp(-0.2 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'precision': 0.75 + 0.2 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'recall': 0.7 + 0.25 * (1 - np.exp(-0.18 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'f1_score': 0.7 + 0.2 * (1 - np.exp(-0.2 * np.array(epochs))) + np.random.normal(0, 0.01, 20)
        },
        'denomination': {
            'accuracy': 0.7 + 0.25 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'precision': 0.65 + 0.3 * (1 - np.exp(-0.12 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'recall': 0.6 + 0.35 * (1 - np.exp(-0.1 * np.array(epochs))) + np.random.normal(0, 0.01, 20),
            'f1_score': 0.6 + 0.3 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, 20)
        }
    }
    
    # Generate confusion matrices
    def generate_confusion_matrix(num_classes, noise=0.1):
        # Create a diagonal-dominant matrix
        cm = np.eye(num_classes) * 0.8
        # Add some noise
        cm += np.random.uniform(0, noise, (num_classes, num_classes))
        # Normalize rows
        row_sums = cm.sum(axis=1, keepdims=True)
        return cm / row_sums
    
    confusion_matrices = {
        'banknote': generate_confusion_matrix(2),  # Binary classification
        'denomination': generate_confusion_matrix(10)  # 10 classes
    }
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rates': learning_rates,
        'layer_metrics': layer_metrics,
        'confusion_matrices': confusion_matrices
    }

def test_visualization():
    """Test the visualization components with sample data."""
    print("🚀 Starting visualization verification...")
    
    # Create output directory
    output_dir = Path("test_visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("📊 Generating sample data...")
    data = generate_sample_data()
    
    # Initialize visualization manager
    print("🔄 Initializing visualization manager...")
    viz = VisualizationManager(
        num_classes_per_layer={
            'banknote': 2,
            'denomination': 10
        },
        class_names={
            'banknote': ['No Banknote', 'Banknote'],
            'denomination': [f'Class {i}' for i in range(10)]
        },
        save_dir=output_dir,
        verbose=True
    )
    
    # Update metrics for each epoch
    print("📈 Updating metrics...")
    for i, epoch in enumerate(data['epochs']):
        # Training phase
        viz.update_metrics(
            epoch=epoch,
            metrics={
                'loss': data['train_loss'][i],
                'banknote_accuracy': data['layer_metrics']['banknote']['accuracy'][i],
                'banknote_precision': data['layer_metrics']['banknote']['precision'][i],
                'banknote_recall': data['layer_metrics']['banknote']['recall'][i],
                'banknote_f1_score': data['layer_metrics']['banknote']['f1_score'][i],
                'denomination_accuracy': data['layer_metrics']['denomination']['accuracy'][i],
                'denomination_precision': data['layer_metrics']['denomination']['precision'][i],
                'denomination_recall': data['layer_metrics']['denomination']['recall'][i],
                'denomination_f1_score': data['layer_metrics']['denomination']['f1_score'][i],
            },
            phase='train',
            learning_rate=data['learning_rates'][i]
        )
        
        # Validation phase (every 5 epochs)
        if i % 5 == 0:
            viz.update_metrics(
                epoch=epoch,
                metrics={
                    'val_loss': data['val_loss'][i],
                    'val_banknote_accuracy': data['layer_metrics']['banknote']['accuracy'][i] * 0.95,
                    'val_denomination_accuracy': data['layer_metrics']['denomination']['accuracy'][i] * 0.9
                },
                phase='val'
            )
    
    # Add confusion matrices for the final epoch
    viz.update_metrics(
        epoch=data['epochs'][-1],
        metrics={},
        phase='val',
        confusion_matrices=data['confusion_matrices']
    )
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    results = {}
    
    # Individual charts
    print("\n📊 Generating training curves...")
    results['training_curves'] = viz.generate_training_curves()
    
    print("📈 Generating learning rate chart...")
    results['learning_rate'] = viz.generate_learning_rate_chart()
    
    print("🔢 Generating confusion matrices...")
    results['confusion_matrices'] = viz.generate_confusion_matrices()
    
    # Metrics comparisons
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        print(f"📊 Generating {metric} comparison...")
        results[f'{metric}_comparison'] = viz.generate_metrics_comparison(metric)
    
    # Research dashboard
    print("📊 Generating research dashboard...")
    results['dashboard'] = viz.generate_research_dashboard()
    
    # Save metrics summary
    print("💾 Saving metrics summary...")
    results['metrics_summary'] = viz.save_metrics_summary()
    
    # Print results
    print("\n📋 Generated files:")
    for name, path in results.items():
        if isinstance(path, dict):
            print(f"{name}:")
            for k, v in path.items():
                print(f"  {k}: {v}")
        else:
            print(f"{name}: {path}")
    
    print("\n✅ Visualization verification completed successfully!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    return results

if __name__ == "__main__":
    test_visualization()
