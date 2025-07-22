"""
file_path: smartcash/ui/dataset/visualization/components/sample_comparison_viewer.py

Interactive sample comparison viewer for raw vs preprocessed/augmented images.
Shows side-by-side comparison when samples are clicked in summary container.
"""
from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
import numpy as np
from pathlib import Path
from IPython.display import display, HTML, clear_output
import base64
from io import BytesIO

class SampleComparisonViewer:
    """Interactive viewer for comparing raw vs processed/augmented samples."""
    
    def __init__(self):
        """Initialize the sample comparison viewer."""
        self.current_samples = {}
        self.selected_split = 'train'
        self.comparison_mode = 'preprocessed'  # 'preprocessed' or 'augmented'
        
        # Create UI components
        self._create_widgets()
        self._setup_event_handlers()
    
    def _create_widgets(self):
        """Create widgets for the sample comparison viewer."""
        # Add CSS styles
        comparison_style = """
        .sample-comparison-container {
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            background: #ffffff;
        }
        .comparison-header {
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 16px;
            text-align: center;
        }
        .sample-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 16px 0;
        }
        .sample-panel {
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            background: #f8f9fa;
        }
        .sample-title {
            font-weight: 600;
            color: white;
            margin-bottom: 12px;
            padding: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        .sample-info {
            font-size: 12px;
            color: #6c757d;
            margin-top: 8px;
        }
        .no-sample-message {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 40px;
            border: 2px dashed #e1e4e8;
            border-radius: 8px;
            background: #f8f9fa;
        }
        """
        
        # Add styles to notebook
        display(HTML(f'<style>{comparison_style}</style>'))
        
        # Control widgets
        self.split_selector = widgets.Dropdown(
            options=['train', 'valid', 'test'],
            value='train',
            description='Split:',
            style={'description_width': 'initial'}
        )
        
        self.mode_selector = widgets.ToggleButtons(
            options=[('Preprocessed', 'preprocessed'), ('Augmented', 'augmented')],
            value='preprocessed',
            description='Comparison Mode:',
            style={'description_width': 'initial'}
        )
        
        # Sample selector
        self.sample_list = widgets.Select(
            options=[],
            description='Samples:',
            disabled=True,
            layout=widgets.Layout(height='150px')
        )
        
        # Comparison display area
        self.comparison_display = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #e1e4e8',
                border_radius='8px',
                padding='16px',
                min_height='400px'
            )
        )
        
        # Main container
        controls_container = widgets.HBox([
            self.split_selector,
            self.mode_selector
        ], layout=widgets.Layout(justify_content='center', margin='10px 0'))
        
        self.main_container = widgets.VBox([
            widgets.HTML('<div class="comparison-header">📸 Sample Comparison Viewer</div>'),
            controls_container,
            self.sample_list,
            self.comparison_display
        ], layout=widgets.Layout(width='100%'))
        
        # Initialize with empty state
        self._show_no_samples()
    
    def _setup_event_handlers(self):
        """Set up event handlers for widgets."""
        self.split_selector.observe(self._on_split_change, names='value')
        self.mode_selector.observe(self._on_mode_change, names='value')
        self.sample_list.observe(self._on_sample_select, names='value')
    
    def _on_split_change(self, change):
        """Handle split selection change."""
        self.selected_split = change['new']
        self._update_sample_list()
        self._show_no_samples()
    
    def _on_mode_change(self, change):
        """Handle comparison mode change."""
        self.comparison_mode = change['new']
        if self.sample_list.value:
            self._display_comparison(self.sample_list.value)
    
    def _on_sample_select(self, change):
        """Handle sample selection change."""
        if change['new']:
            self._display_comparison(change['new'])
    
    def update_samples(self, comprehensive_stats: Dict[str, Any]):
        """Update available samples from comprehensive stats.
        
        Args:
            comprehensive_stats: Stats data from refresh operation
        """
        try:
            # Extract sample information from stats
            self.current_samples = self._extract_sample_info(comprehensive_stats)
            
            # Update sample list
            self._update_sample_list()
            
            # If no samples selected, show the first available comparison
            if self.current_samples and not self.sample_list.value:
                first_sample = list(self.current_samples.keys())[0]
                self.sample_list.value = first_sample
                
        except Exception as e:
            with self.comparison_display:
                clear_output(wait=True)
                print(f"❌ Error updating samples: {e}")
    
    def _extract_sample_info(self, stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract sample information from comprehensive stats.
        
        Args:
            stats: Comprehensive statistics
            
        Returns:
            Dictionary of sample information
        """
        samples = {}
        
        # Generate mock sample data for demonstration
        # In real implementation, this would extract actual file paths from stats
        dataset_stats = stats.get('dataset_stats', {})
        if dataset_stats.get('success', False):
            by_split = dataset_stats.get('by_split', {})
            
            for split, split_data in by_split.items():
                raw_count = split_data.get('raw', 0)
                preprocessed_count = split_data.get('preprocessed', 0)
                
                # Create sample entries (mock data)
                for i in range(min(5, raw_count)):  # Limit to 5 samples for demo
                    sample_key = f"{split}_sample_{i+1}"
                    samples[sample_key] = {
                        'split': split,
                        'index': i,
                        'raw_path': f"data/{split}/raw/sample_{i+1}.jpg",
                        'preprocessed_path': f"data/{split}/preprocessed/pre_sample_{i+1}.npy" if i < preprocessed_count else None,
                        'augmented_path': f"data/{split}/augmented/aug_sample_{i+1}.npy" if i < preprocessed_count else None,
                        'has_preprocessed': i < preprocessed_count,
                        'has_augmented': i < preprocessed_count  # Assume augmented follows preprocessed
                    }
        
        return samples
    
    def _update_sample_list(self):
        """Update the sample list dropdown."""
        # Filter samples by selected split
        split_samples = {
            k: v for k, v in self.current_samples.items() 
            if v.get('split') == self.selected_split
        }
        
        if split_samples:
            options = [(f"Sample {v['index']+1}", k) for k, v in split_samples.items()]
            self.sample_list.options = options
            self.sample_list.disabled = False
        else:
            self.sample_list.options = []
            self.sample_list.disabled = True
            self._show_no_samples()
    
    def _display_comparison(self, sample_key: str):
        """Display side-by-side comparison for selected sample.
        
        Args:
            sample_key: Key of the sample to display
        """
        with self.comparison_display:
            clear_output(wait=True)
            
            try:
                sample_info = self.current_samples.get(sample_key)
                if not sample_info:
                    print("❌ Sample not found")
                    return
                
                # Check if comparison data is available
                has_comparison = (
                    self.comparison_mode == 'preprocessed' and sample_info.get('has_preprocessed') or
                    self.comparison_mode == 'augmented' and sample_info.get('has_augmented')
                )
                
                if not has_comparison:
                    self._show_no_comparison(sample_info)
                    return
                
                # Display comparison layout
                self._render_comparison(sample_info)
                
            except Exception as e:
                print(f"❌ Error displaying comparison: {e}")
    
    def _show_no_samples(self):
        """Show message when no samples are available."""
        with self.comparison_display:
            clear_output(wait=True)
            display(HTML(
                '<div class="no-sample-message">'
                f'📂 No samples available for {self.selected_split} split<br>'
                '<small>Run the refresh operation to load sample data</small>'
                '</div>'
            ))
    
    def _show_no_comparison(self, sample_info: Dict[str, Any]):
        """Show message when comparison data is not available."""
        with self.comparison_display:
            clear_output(wait=True)
            display(HTML(
                '<div class="no-sample-message">'
                f'⚠️ {self.comparison_mode.title()} data not available for this sample<br>'
                f'<small>Sample: {sample_info.get("raw_path", "Unknown")}</small>'
                '</div>'
            ))
    
    def _render_comparison(self, sample_info: Dict[str, Any]):
        """Render the actual comparison display.
        
        Args:
            sample_info: Information about the sample to display
        """
        try:
            # Generate comparison HTML
            comparison_html = f"""
            <div class="sample-comparison-container">
                <div class="comparison-header">
                    📸 Sample {sample_info['index']+1} - {sample_info['split'].title()} Split
                </div>
                
                <div class="sample-grid">
                    <div class="sample-panel">
                        <div class="sample-title">🖼️ Raw Image</div>
                        <div style="background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 4px; border: 2px dashed #ccc;">
                            <div style="text-align: center; color: #666;">
                                📷<br>
                                <small>Original Image</small><br>
                                <code>{sample_info.get('raw_path', 'No path')}</code>
                            </div>
                        </div>
                        <div class="sample-info">
                            Format: JPG/PNG<br>
                            Type: Raw Image
                        </div>
                    </div>
                    
                    <div class="sample-panel">
                        <div class="sample-title">🔄 {self.comparison_mode.title()} Data</div>
                        <div style="background: #e8f5e8; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 4px; border: 2px dashed #4caf50;">
                            <div style="text-align: center; color: #2e7d32;">
                                🗂️<br>
                                <small>{self.comparison_mode.title()} Array</small><br>
                                <code>{sample_info.get(f'{self.comparison_mode}_path', 'No path')}</code>
                            </div>
                        </div>
                        <div class="sample-info">
                            Format: NumPy Array (.npy)<br>
                            Type: {self.comparison_mode.title()} Data<br>
                            <small>Shape: (H, W, C) - processed tensor</small>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 16px; padding: 12px; background: #f8f9fa; border-radius: 4px;">
                    <strong>💡 Processing Information:</strong><br>
                    <small>
                        Raw → {self.comparison_mode.title()}: 
                        {'Normalized, resized, and converted to tensor format' if self.comparison_mode == 'preprocessed' else 'Applied augmentation transforms (rotation, scaling, color adjustments)'}
                    </small>
                </div>
            </div>
            """
            
            display(HTML(comparison_html))
            
        except Exception as e:
            print(f"❌ Error rendering comparison: {e}")
    
    def get_widget(self) -> widgets.Widget:
        """Get the main comparison viewer widget."""
        return self.main_container


def create_sample_comparison_viewer() -> SampleComparisonViewer:
    """Create a sample comparison viewer component.
    
    Returns:
        Configured sample comparison viewer ready for use
        
    Usage:
        >>> viewer = create_sample_comparison_viewer()
        >>> display(viewer.get_widget())
        >>> # Later update with stats data:
        >>> viewer.update_samples(comprehensive_stats)
    """
    return SampleComparisonViewer()