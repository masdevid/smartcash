#!/usr/bin/env python3
"""
Test script for the visualization package.

This script generates sample data and uses the visualization components
to create various charts and verify their functionality.
"""

import os
import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the missing module
sys.modules['smartcash.model.analysis'] = MagicMock()
sys.modules['smartcash.common.logger'] = MagicMock()

# Now import our visualization components
from smartcash.model.training.visualization.manager import VisualizationManager
from smartcash.model.training.visualization import types

def generate_sample_data():
    """Generate sample training data for testing visualization."""
    np.random.seed(42)
    
    # Generate sample metrics
    num_epochs = 20
    epochs = list(range(1, num_epochs + 1))
    
    # Training and validation losses
    train_loss = np.exp(-np.linspace(0, 2, num_epochs)) + np.random.normal(0, 0.02, num_epochs)
    val_loss = np.exp(-np.linspace(0, 2, num_epochs)) + np.random.normal(0, 0.03, num_epochs) + 0.1
    
    # Learning rate schedule
    learning_rates = np.linspace(1e-3, 1e-5, num_epochs)
    
    # Layer metrics
    layer_metrics = {
        'banknote': {
            'accuracy': 0.8 + 0.15 * (1 - np.exp(-0.2 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'precision': 0.75 + 0.2 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'recall': 0.7 + 0.25 * (1 - np.exp(-0.18 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'f1_score': 0.7 + 0.2 * (1 - np.exp(-0.2 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs)
        },
        'denomination': {
            'accuracy': 0.7 + 0.25 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'precision': 0.65 + 0.3 * (1 - np.exp(-0.12 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'recall': 0.6 + 0.35 * (1 - np.exp(-0.1 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs),
            'f1_score': 0.6 + 0.3 * (1 - np.exp(-0.15 * np.array(epochs))) + np.random.normal(0, 0.01, num_epochs)
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
    print("🚀 Starting visualization tests...")
    
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
    results['training_curves'] = viz.generate_training_curves()
    results['learning_rate'] = viz.generate_learning_rate_chart()
    results['confusion_matrices'] = viz.generate_confusion_matrices()
    
    # Metrics comparisons
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        results[f'{metric}_comparison'] = viz.generate_metrics_comparison(metric)
    
    # Research dashboard
    results['dashboard'] = viz.generate_research_dashboard()
    
    # Save metrics summary
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
    
    print("\n✅ Visualization tests completed successfully!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    return results

if __name__ == "__main__":
    test_visualization()
