#!/usr/bin/env python3
"""
UI Integration Example

Demonstrates how to use the enhanced metrics callback with color information
for UI frameworks like Streamlit, Gradio, or custom web interfaces.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback
from smartcash.model.training.utils.metric_color_utils import ColorScheme


class MockUIInterface:
    """
    Mock UI interface to demonstrate how to handle color-coded metrics.
    In practice, this would be a Streamlit app, Gradio interface, etc.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
        
    def update_metrics_display(self, phase: str, epoch: int, metrics: Dict[str, Any], colored_metrics: Dict[str, Dict]):
        """Handle incoming metrics from the training callback."""
        print(f"\n🖥️  UI UPDATE - {phase} Epoch {epoch}")
        print("=" * 60)
        
        # Store for history
        self.metrics_history.append({
            'phase': phase,
            'epoch': epoch,
            'metrics': metrics,
            'colored_metrics': colored_metrics
        })
        self.current_metrics = colored_metrics
        
        # Display categorized metrics with color coding
        self._display_loss_metrics(colored_metrics)
        self._display_accuracy_metrics(colored_metrics)
        self._display_map_metrics(colored_metrics)
        self._display_layer_metrics(colored_metrics)
        
        print("=" * 60)
    
    def _display_loss_metrics(self, colored_metrics: Dict[str, Dict]):
        """Display loss metrics with color coding."""
        loss_metrics = {k: v for k, v in colored_metrics.items() if 'loss' in k.lower()}
        if loss_metrics:
            print("\n📉 LOSS METRICS:")
            for metric_name, color_data in loss_metrics.items():
                self._display_metric(metric_name, color_data)
    
    def _display_accuracy_metrics(self, colored_metrics: Dict[str, Dict]):
        """Display accuracy metrics with color coding."""
        acc_metrics = {k: v for k, v in colored_metrics.items() if 'accuracy' in k.lower()}
        if acc_metrics:
            print("\n🎯 ACCURACY METRICS:")
            for metric_name, color_data in acc_metrics.items():
                self._display_metric(metric_name, color_data)
    
    def _display_map_metrics(self, colored_metrics: Dict[str, Dict]):
        """Display mAP metrics with color coding."""
        map_metrics = {k: v for k, v in colored_metrics.items() if 'map' in k.lower()}
        if map_metrics:
            print("\n📊 mAP METRICS:")
            for metric_name, color_data in map_metrics.items():
                self._display_metric(metric_name, color_data)
    
    def _display_layer_metrics(self, colored_metrics: Dict[str, Dict]):
        """Display layer-specific metrics with color coding."""
        layer_metrics = {k: v for k, v in colored_metrics.items() 
                        if any(layer in k.lower() for layer in ['layer_1', 'layer_2', 'layer_3'])}
        if layer_metrics:
            print("\n🔧 LAYER METRICS:")
            for metric_name, color_data in layer_metrics.items():
                self._display_metric(metric_name, color_data)
    
    def _display_metric(self, metric_name: str, color_data: Dict):
        """Display a single metric with all available color information."""
        value = color_data['value']
        status = color_data['status']
        
        # Get colors for different display contexts (handle empty colors for loss metrics)
        colors = color_data.get('colors', {})
        emoji = colors.get('emoji', '📊')  # Default emoji if not available
        html_color = colors.get('html', '#666666')  # Default color if not available
        rgb_color = colors.get('rgb', '(100, 100, 100)')  # Default RGB if not available
        
        print(f"   {emoji} {metric_name}: {value:.4f} ({status})")
        if colors:  # Only show color info if colors are available
            print(f"      └─ HTML: {html_color} | RGB: {rgb_color}")
        else:
            print(f"      └─ No color coding for this metric type")
    
    def get_current_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current training status for dashboard display."""
        if not self.current_metrics:
            return {}
        
        # Count metrics by status
        status_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'critical': 0}
        critical_metrics = []
        excellent_metrics = []
        
        for metric_name, color_data in self.current_metrics.items():
            status = color_data['status']
            if status in status_counts:
                status_counts[status] += 1
                
                if status == 'critical':
                    critical_metrics.append(metric_name)
                elif status == 'excellent':
                    excellent_metrics.append(metric_name)
        
        return {
            'status_counts': status_counts,
            'critical_metrics': critical_metrics,
            'excellent_metrics': excellent_metrics,
            'total_metrics': len(self.current_metrics)
        }
    
    def export_metrics_for_plotting(self) -> Dict[str, Any]:
        """Export metrics in a format suitable for plotting libraries."""
        if not self.metrics_history:
            return {}
        
        # Organize data for time series plotting
        epochs = []
        metrics_over_time = {}
        colors_over_time = {}
        
        for entry in self.metrics_history:
            epoch = entry['epoch']
            epochs.append(epoch)
            
            for metric_name, color_data in entry['colored_metrics'].items():
                if metric_name not in metrics_over_time:
                    metrics_over_time[metric_name] = []
                    colors_over_time[metric_name] = []
                
                metrics_over_time[metric_name].append(color_data['value'])
                colors_over_time[metric_name].append(color_data['colors']['html'])
        
        return {
            'epochs': epochs,
            'metrics': metrics_over_time,
            'colors': colors_over_time
        }


def demonstrate_ui_integration():
    """Demonstrate UI integration with the enhanced metrics callback."""
    print("🎨 UI Integration Demonstration")
    print("=" * 60)
    
    # Create mock UI interface
    ui_interface = MockUIInterface()
    
    # Create enhanced metrics callback with UI integration
    metrics_callback = create_ui_metrics_callback(
        verbose=False,  # Disable console output for UI focus
        console_scheme=ColorScheme.EMOJI,
        ui_callback=ui_interface.update_metrics_display
    )
    
    # Simulate training progression with different metric scenarios
    training_scenarios = [
        {
            'phase': 'training_phase_1',
            'epoch': 1,
            'metrics': {
                'train_loss': 0.7245,
                'val_loss': 2.3381,
                'val_accuracy': 0.25,
                'val_map50': 0.0,
                'layer_1_accuracy': 0.83,
                'layer_1_precision': 0.75
            }
        },
        {
            'phase': 'training_phase_1',
            'epoch': 10,
            'metrics': {
                'train_loss': 0.4521,
                'val_loss': 1.2341,
                'val_accuracy': 0.67,
                'val_map50': 0.23,
                'layer_1_accuracy': 0.89,
                'layer_1_precision': 0.84
            }
        },
        {
            'phase': 'training_phase_2',
            'epoch': 25,
            'metrics': {
                'train_loss': 0.2156,
                'val_loss': 0.5678,
                'val_accuracy': 0.89,
                'val_map50': 0.76,
                'layer_1_accuracy': 0.94,
                'layer_1_precision': 0.91,
                'layer_2_accuracy': 0.87,
                'layer_2_precision': 0.83,
                'layer_3_accuracy': 0.81,
                'layer_3_precision': 0.78
            }
        }
    ]
    
    # Process each scenario
    for scenario in training_scenarios:
        result = metrics_callback(
            scenario['phase'],
            scenario['epoch'],
            scenario['metrics'],
            max_epochs=50
        )
        
        # Show status summary after each update
        status_summary = ui_interface.get_current_status_summary()
        print(f"\n📈 STATUS SUMMARY:")
        print(f"   Total metrics: {status_summary['total_metrics']}")
        for status, count in status_summary['status_counts'].items():
            if count > 0:
                print(f"   {status.title()}: {count}")
        
        if status_summary['critical_metrics']:
            print(f"   ⚠️  Critical: {', '.join(status_summary['critical_metrics'])}")
        if status_summary['excellent_metrics']:
            print(f"   ✨ Excellent: {', '.join(status_summary['excellent_metrics'])}")
    
    # Demonstrate data export for plotting
    print(f"\n📊 EXPORT FOR PLOTTING:")
    plot_data = ui_interface.export_metrics_for_plotting()
    print(f"   Epochs tracked: {plot_data['epochs']}")
    print(f"   Metrics available: {list(plot_data['metrics'].keys())}")
    
    # Show how to access specific metric progression
    if 'val_loss' in plot_data['metrics']:
        val_losses = plot_data['metrics']['val_loss']
        val_loss_colors = plot_data['colors']['val_loss']
        print(f"   val_loss progression: {val_losses}")
        print(f"   val_loss colors: {val_loss_colors}")


def streamlit_example_code():
    """Example code for Streamlit integration."""
    example_code = '''
# Example Streamlit integration
import streamlit as st
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback

class StreamlitUI:
    def __init__(self):
        self.metrics_container = st.container()
    
    def update_metrics(self, phase, epoch, metrics, colored_metrics):
        with self.metrics_container:
            st.subheader(f"Training Metrics - {phase} Epoch {epoch}")
            
            # Create columns for different metric types
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("📉 **Loss Metrics**")
                for name, data in colored_metrics.items():
                    if 'loss' in name.lower():
                        color = data['colors']['html']
                        value = data['value']
                        st.markdown(f'<span style="color: {color}">{name}: {value:.4f}</span>', 
                                  unsafe_allow_html=True)
            
            with col2:
                st.write("🎯 **Accuracy Metrics**")
                for name, data in colored_metrics.items():
                    if 'accuracy' in name.lower():
                        color = data['colors']['html']
                        value = data['value']
                        st.markdown(f'<span style="color: {color}">{name}: {value:.4f}</span>', 
                                  unsafe_allow_html=True)
            
            with col3:
                st.write("📊 **mAP Metrics**")
                for name, data in colored_metrics.items():
                    if 'map' in name.lower():
                        color = data['colors']['html']
                        value = data['value']
                        st.markdown(f'<span style="color: {color}">{name}: {value:.4f}</span>', 
                                  unsafe_allow_html=True)

# Usage in Streamlit app
ui = StreamlitUI()
metrics_callback = create_ui_metrics_callback(
    verbose=False,
    ui_callback=ui.update_metrics
)
'''
    
    print("\n💻 STREAMLIT INTEGRATION EXAMPLE:")
    print("=" * 60)
    print(example_code)


if __name__ == "__main__":
    demonstrate_ui_integration()
    streamlit_example_code()
    
    print("\n✅ UI Integration Demo Complete!")
    print("\n🎯 Key Benefits:")
    print("   • Automatic color coding based on metric performance")
    print("   • Multiple color schemes for different UI frameworks")
    print("   • Real-time status summaries and alerts")
    print("   • Export-ready data for plotting and visualization")
    print("   • Seamless integration with existing training pipelines")