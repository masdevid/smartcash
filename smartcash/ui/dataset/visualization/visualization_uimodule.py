"""
File: smartcash/ui/dataset/visualization/visualization_uimodule.py
Description: Visualization Module implementation using BaseUIModule mixin pattern.
"""

from typing import Dict, Any, Optional, List, Type

# Third-party imports
import ipywidgets as widgets

# Core imports
from smartcash.ui.core.base_ui_module import BaseUIModule
# Enhanced UI Module Factory removed - use direct instantiation
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Local imports
from .components.visualization_ui import create_visualization_ui
from .configs.visualization_config_handler import VisualizationConfigHandler
from .configs.visualization_defaults import get_default_visualization_config


class VisualizationUIModule(BaseUIModule):
    """
    Modul Visualisasi untuk analisis dan visualisasi dataset.
    
    Fitur:
    - 📊 Visualisasi dan eksplorasi data
    - 🔍 Analisis data interaktif
    - 📈 Berbagai tipe chart dan opsi visualisasi
    - 🔄 Pembaruan data waktu-nyata
    - 🇮🇩 Antarmuka dalam Bahasa Indonesia
    """
    
    # Define required UI components at class level
    _required_components = [
        'main_container',
        'header_container', 
        'form_container',
        'dashboard_container',
        'visualization_container',
        'controls_container',
        'progress_container'
    ]
    
    def __init__(self, enable_environment: bool = False):
        """
        Inisialisasi modul UI Visualisasi.
        
        Args:
            enable_environment: Apakah mengaktifkan fitur manajemen environment
        """
        # Initialize base module first
        super().__init__(
            module_name='visualization',
            parent_module='dataset',
            enable_environment=enable_environment
        )
        
        # Initialize instance attributes
        self._current_visualization = None
        self._datasets = {}
        self.components = {}
        self._progress_bar = None
        self._status_label = None
        self._backend_apis = {}
        self._operations = {}
        self._dashboard_cards = None
        self._config = None
        self._config_handler = None
        self._latest_stats = None
        self._sample_viewer = None
        self._chart_container = None
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize operations
        self._initialize_operations()
        
        # Initialize UI components
        self.components = self.create_ui_components(self._config)
        self._ui_components = self.components  # For backward compatibility
        
        # Initialize dashboard cards
        self._initialize_dashboard()
        
        # Initialize sample comparison viewer
        self._initialize_sample_viewer()
        
        # Initialize chart container
        self._initialize_chart_container()
    
    def _initialize_config(self) -> None:
        """Initialize configuration and config handler."""
        self._config = self.get_default_config()
        self._config_handler = self.create_config_handler(self._config)
        
    
    def _initialize_operations(self):
        """Initialize all visualization operations."""
        from .operations import (
            RefreshVisualizationOperation,
            LoadPreprocessedOperation,
            LoadAugmentedOperation
        )
        
        self._operations = {
            'refresh': RefreshVisualizationOperation(self),
            'load_preprocessed': LoadPreprocessedOperation(self),
            'load_augmented': LoadAugmentedOperation(self)
        }
    
    def _update_dashboard_stats(self) -> None:
        """Update statistics on the dashboard cards with empty stats."""
        if not hasattr(self, '_dashboard_cards') or not self._dashboard_cards:
            return
        
        # Initialize with empty stats (will be updated by refresh operation)
        empty_stats = {
            'dataset_stats': {
                'by_split': {
                    'train': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                    'valid': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                    'test': {'raw': 0, 'preprocessed': 0, 'augmented': 0}
                },
                'overview': {'total_files': 0}
            },
            'augmentation_stats': {
                'by_split': {
                    'train': {'file_count': 0},
                    'valid': {'file_count': 0},
                    'test': {'file_count': 0}
                }
            }
        }
        
        # Update all cards with empty stats
        if hasattr(self._dashboard_cards, 'update_all_cards'):
            self._dashboard_cards.update_all_cards(empty_stats)
    
    def _initialize_dashboard(self):
        """Initialize the dashboard with enhanced visualization stats cards."""
        from .components.visualization_stats_cards import create_visualization_stats_dashboard
        
        # Create enhanced visualization stats dashboard
        self._dashboard_cards = create_visualization_stats_dashboard()
        
        # Add container cards to layout
        if 'containers' in self.components and 'dashboard_container' in self.components['containers']:
            self.components['containers']['dashboard_container'].children = [self._dashboard_cards.get_container()]
        
        # Initialize with empty stats (will be updated by refresh operation)
        self._update_dashboard_stats()
        
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the visualization module.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_visualization_config()
        
    def create_config_handler(self, config: Dict[str, Any]) -> VisualizationConfigHandler:
        """
        Create and return a config handler instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            VisualizationConfigHandler instance
        """
        return VisualizationConfigHandler(config)
        
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and return UI components for the visualization module.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of UI components
        """
        return create_visualization_ui(config=config)
    
    def _initialize_components(self):
        """Initialize all visualization components."""
        # Initialize UI components
        self._ui_components = create_visualization_ui()
        
        # Set up event handlers
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Visualization module-specific button handlers.
        
        Returns:
            Dictionary mapping button names to their handler methods
        """
        # Call parent method to get base handlers (save, reset, etc.)
        handlers = super()._get_module_button_handlers()
        
        # Add Visualization-specific handlers
        visualization_handlers = {
            # Action buttons
            'refresh': self._on_refresh_click,
            'preprocessed': self._on_preprocessed_click,
            'augmented': self._on_augmented_click,
            'save': self._handle_save_config,
            'reset': self._handle_reset_config
        }
        
        handlers.update(visualization_handlers)
        return handlers
        
    def _on_refresh_click(self, button=None):
        """Handle refresh button click."""
        from .operations import RefreshVisualizationOperation
        
        operation = RefreshVisualizationOperation(self)
        result = operation.execute()
        
        # All visualization operations now return dictionaries
        if not result.get('success', False):
            self.log_error(result.get('message', 'Unknown error'))
        else:
            self.log_info(f"✅ {result.get('message', 'Visualization refreshed successfully')}")
    
    def _on_preprocessed_click(self, button=None):
        """Handle preprocessed sample button click."""
        from .operations import LoadPreprocessedOperation
        
        operation = LoadPreprocessedOperation(self)
        result = operation.execute()
        
        # All visualization operations now return dictionaries
        if not result.get('success', False):
            self.log_error(result.get('message', 'Unknown error'))
        else:
            self.log_info(f"✅ {result.get('message', 'Preprocessed samples loaded successfully')}")
            # Update dashboard stats after loading preprocessed data
            self._update_dashboard_stats()
    
    def _on_augmented_click(self, button=None):
        """Handle augmented sample button click."""
        from .operations import LoadAugmentedOperation
        
        operation = LoadAugmentedOperation(self)
        result = operation.execute()
        
        # All visualization operations now return dictionaries
        if not result.get('success', False):
            self.log_error(result.get('message', 'Unknown error'))
        else:
            self.log_info(f"✅ {result.get('message', 'Augmented samples loaded successfully')}")
            # Update dashboard stats after loading augmented data
            self._update_dashboard_stats()
        
    def _on_action_button_click(self, button_name: str):
        """Legacy method to handle action button clicks.
        
        Args:
            button_name: Name of the button that was clicked
        """
        self.log_warning(f"Using legacy button handler for: {button_name}")
        # Map legacy button names to new handlers if needed
        if button_name == 'refresh':
            self._on_refresh_click()
        elif button_name == 'preprocessed':
            self._on_preprocessed_click()
        elif button_name == 'augmented':
            self._on_augmented_click()
    
    def _on_visualization_type_change(self, change):
        """Handle visualization type change event."""
        if change['name'] == 'value':
            self.update_visualization(change['new'])
    
    def get_backend_api(self, api_name: str) -> Optional[Any]:
        """Get a backend API by name.
        
        Args:
            api_name: Name of the API to retrieve
            
        Returns:
            The API implementation or None if not available
        """
        return self._backend_apis.get(api_name)
    
    def update_visualization(self, viz_type: str) -> None:
        """Update the current visualization.
        
        Args:
            viz_type: Type of visualization to display
            
        Supported visualization types:
            - 'bar': Bar chart
            - 'line': Line chart
            - 'scatter': Scatter plot
            - 'preprocessed_samples': Show preprocessed samples
            - 'augmented_samples': Show augmented samples
        """
        self.log_info(f"Memperbarui visualisasi ke tipe: {viz_type}")
        
        try:
            # Clear previous visualization
            if 'visualization_container' in self.components.get('containers', {}):
                self.components['containers']['visualization'].clear_output()
            
            # Handle different visualization types
            if viz_type in ['bar', 'line', 'scatter']:
                self._render_chart(viz_type)
            elif viz_type == 'preprocessed_samples':
                self._render_preprocessed_samples()
            elif viz_type == 'augmented_samples':
                self._render_augmented_samples()
            else:
                self.log_warning(f"Tipe visualisasi tidak didukung: {viz_type}")
                
            self._current_visualization = viz_type
            
        except Exception as e:
            error_msg = f"Gagal memperbarui visualisasi: {str(e)}"
            self.log_error(error_msg)
            self.update_operation_status(error_msg, "error")
    
    def _render_chart(self, chart_type: str) -> None:
        """Render a chart visualization.
        
        Args:
            chart_type: Type of chart to render (bar, line, scatter)
        """
        # Get the visualization container
        viz_container = self.components.get('containers', {}).get('visualization')
        if not viz_container:
            raise ValueError("Visualization container not found")
        
        # Get the dataset to visualize
        if not self._datasets:
            viz_container.append_display_data({"text/plain": "Tidak ada data untuk divisualisasikan"})
            return
        
        # TODO: Implement actual chart rendering logic
        # This is a placeholder for the actual implementation
        with viz_container:
            import matplotlib.pyplot as plt
            
            # Sample data - replace with actual data from self._datasets
            data = {
                'categories': ['A', 'B', 'C', 'D'],
                'values': [10, 20, 15, 30]
            }
            
            # Create the appropriate chart type
            fig, ax = plt.subplots(figsize=(8, 4))
            if chart_type == 'bar':
                ax.bar(data['categories'], data['values'])
                ax.set_title('Bar Chart')
            elif chart_type == 'line':
                ax.plot(data['categories'], data['values'], marker='o')
                ax.set_title('Line Chart')
            elif chart_type == 'scatter':
                ax.scatter(range(len(data['values'])), data['values'])
                ax.set_title('Scatter Plot')
            
            plt.tight_layout()
            plt.show()
    
    def _render_preprocessed_samples(self) -> None:
        """Render preprocessed samples in the visualization container."""
        viz_container = self.components.get('containers', {}).get('visualization')
        if not viz_container:
            raise ValueError("Visualization container not found")
        
        samples = self._datasets.get('preprocessed', [])
        if not samples:
            viz_container.append_display_data({"text/plain": "Tidak ada sampel preprocessed yang tersedia"})
            return
        
        # Display sample information
        sample_info = [
            f"Total sampel preprocessed: {len(samples)}",
            f"Kolom yang tersedia: {', '.join(samples[0].keys()) if samples else 'Tidak ada'}"
        ]
        
        # Display first few samples as a table
        import pandas as pd
        df = pd.DataFrame(samples[:5])  # Show first 5 samples
        
        with viz_container:
            from IPython.display import display, HTML
            
            # Display sample info
            display(HTML("<h4>Informasi Sampel Preprocessed</h4>"))
            display(HTML("<br>".join(f"<div>{line}</div>" for line in sample_info)))
            
            # Display samples table
            display(HTML("<h4>Contoh Sampel (5 pertama)</h4>"))
            display(df)
    
    def _render_augmented_samples(self) -> None:
        """Render augmented samples in the visualization container."""
        viz_container = self.components.get('containers', {}).get('visualization')
        if not viz_container:
            raise ValueError("Visualization container not found")
        
        samples = self._datasets.get('augmented', [])
        if not samples:
            viz_container.append_display_data({"text/plain": "Tidak ada sampel augmented yang tersedia"})
            return
        
        # Display sample information
        sample_info = [
            f"Total sampel augmented: {len(samples)}",
            f"Kolom yang tersedia: {', '.join(samples[0].keys()) if samples else 'Tidak ada'}"
        ]
        
        # Display first few samples as a table
        import pandas as pd
        df = pd.DataFrame(samples[:5])  # Show first 5 samples
        
        with viz_container:
            from IPython.display import display, HTML
            
            # Display sample info
            display(HTML("<h4>Informasi Sampel Augmented</h4>"))
            display(HTML("<br>".join(f"<div>{line}</div>" for line in sample_info)))
            
            # Display samples table
            display(HTML("<h4>Contoh Sampel (5 pertama)</h4>"))
            display(df)
    
    def add_dataset(self, dataset_id: str, dataset_data: Any):
        """Add a dataset for visualization.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_data: Dataset to visualize
        """
        self._datasets[dataset_id] = dataset_data
        self._update_ui()
    
    def _update_ui(self):
        """Update UI components based on current state."""
        # Update UI components based on current state
        pass
    
    def update_stats_cards(self, comprehensive_stats: Dict[str, Any]) -> None:
        """Update stats cards with comprehensive statistics from backend.
        
        Args:
            comprehensive_stats: Complete statistics from refresh operation
        """
        try:
            if hasattr(self, '_dashboard_cards') and self._dashboard_cards:
                # Store latest stats
                self._latest_stats = comprehensive_stats
                
                # Update all cards with the comprehensive stats
                if hasattr(self._dashboard_cards, 'update_all_cards'):
                    self._dashboard_cards.update_all_cards(comprehensive_stats)
                
                # Log successful update
                if hasattr(self, 'log_info'):
                    self.log_info("✅ Stats cards updated with backend data")
                    # Log summary of data
                    dataset_stats = comprehensive_stats.get('dataset_stats', {})
                    if dataset_stats.get('success', False):
                        overview = dataset_stats.get('overview', {})
                        total_files = overview.get('total_files', 0)
                        self.log_info(f"📊 Updated with {total_files:,} total files")
                        
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"❌ Error updating stats cards: {e}")
    
    def update_charts(self, comprehensive_stats: Dict[str, Any]) -> None:
        """Update charts with comprehensive statistics from backend.
        
        Args:
            comprehensive_stats: Complete statistics from refresh operation
        """
        try:
            # Store latest stats for chart updates
            self._latest_stats = comprehensive_stats
            
            # Update sample comparison viewer
            if hasattr(self, '_sample_viewer') and self._sample_viewer:
                self._sample_viewer.update_samples(comprehensive_stats)
            
            # Update tabbed chart container
            if hasattr(self, '_chart_container') and self._chart_container:
                self._chart_container.update_charts(comprehensive_stats)
            
            # Log the update
            if hasattr(self, 'log_info'):
                class_dist = comprehensive_stats.get('class_distribution', {})
                if class_dist.get('success', False):
                    total_classes = class_dist.get('total_classes', 0)
                    total_objects = class_dist.get('total_objects', 0)
                    self.log_info(f"📈 Charts updated: {total_classes} classes, {total_objects:,} objects")
                    self.log_info(f"📸 Sample viewer updated with comparison data")
                else:
                    self.log_info("📈 Charts updated (no data available)")
                    
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"❌ Error updating charts: {e}")
    
    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """Get the latest comprehensive statistics.
        
        Returns:
            Latest statistics data or None if not available
        """
        return getattr(self, '_latest_stats', None)
    
    def _initialize_sample_viewer(self):
        """Initialize the sample comparison viewer."""
        try:
            from .components.sample_comparison_viewer import create_sample_comparison_viewer
            
            # Create sample viewer
            self._sample_viewer = create_sample_comparison_viewer()
            
            # Add to summary container if available
            if ('containers' in self.components and 
                'summary_container' in self.components['containers']):
                
                summary_container = self.components['containers']['summary_container']
                
                # Check if summary container has set_content method
                if hasattr(summary_container, 'set_content'):
                    # Create a tabbed interface in summary
                    from IPython.display import display, HTML
                    
                    # Add sample viewer to summary with a header
                    viewer_widget = self._sample_viewer.get_widget()
                    summary_content = widgets.VBox([
                        widgets.HTML('<h3>📸 Sample Analysis & Comparison</h3>'),
                        viewer_widget
                    ])
                    
                    summary_container.set_content(summary_content)
                else:
                    # Fallback: add directly if possible
                    if hasattr(summary_container, 'children'):
                        viewer_widget = self._sample_viewer.get_widget()
                        summary_container.children = list(summary_container.children) + [viewer_widget]
            
            if hasattr(self, 'log_info'):
                self.log_info("✅ Sample comparison viewer initialized")
                
        except Exception as e:
            if hasattr(self, 'log_warning'):
                self.log_warning(f"⚠️ Sample viewer initialization warning: {e}")
            # Don't fail completely if sample viewer fails to initialize
    
    def get_sample_viewer(self) -> Optional[Any]:
        """Get the sample comparison viewer instance.
        
        Returns:
            Sample viewer instance or None if not initialized
        """
        return getattr(self, '_sample_viewer', None)
    
    def _initialize_chart_container(self):
        """Initialize the tabbed chart container."""
        try:
            from .components.charts import create_tabbed_chart_container
            
            # Create tabbed chart container
            self._chart_container = create_tabbed_chart_container()
            
            # Add to visualization container if available
            if ('containers' in self.components and 
                'visualization_container' in self.components['containers']):
                
                viz_container = self.components['containers']['visualization_container']
                
                # Add chart container to visualization area
                if hasattr(viz_container, 'children'):
                    chart_widget = self._chart_container.get_widget()
                    viz_container.children = list(viz_container.children) + [chart_widget]
                elif hasattr(viz_container, 'set_content'):
                    chart_widget = self._chart_container.get_widget()
                    viz_container.set_content(chart_widget)
            
            if hasattr(self, 'log_info'):
                self.log_info("✅ Tabbed chart container initialized")
                
        except Exception as e:
            if hasattr(self, 'log_warning'):
                self.log_warning(f"⚠️ Chart container initialization warning: {e}")
            # Don't fail completely if chart container fails to initialize
    
    def get_chart_container(self) -> Optional[Any]:
        """Get the tabbed chart container instance.
        
        Returns:
            Chart container instance or None if not initialized
        """
        return getattr(self, '_chart_container', None)

