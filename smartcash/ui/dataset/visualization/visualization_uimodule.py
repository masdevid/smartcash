"""
File: smartcash/ui/dataset/visualization/visualization_uimodule.py
Description: Visualization Module implementation using BaseUIModule mixin pattern.
"""

from typing import Dict, Any, Optional, List, Type

# Third-party imports
import ipywidgets as widgets

# Core imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory
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
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize operations
        self._initialize_operations()
        
        # Initialize UI components
        self.components = self.create_ui_components(self._config)
        self._ui_components = self.components  # For backward compatibility
        
        # Initialize dashboard cards
        self._initialize_dashboard()
    
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
        """Update statistics on the dashboard cards."""
        if not hasattr(self, '_dashboard_cards') or not self._dashboard_cards:
            return
        
        # Get actual stats from backend or use default values
        stats = {
            'train': {
                'preprocessed': 0,
                'augmented': 0,
                'total': 0
            },
            'validation': {
                'preprocessed': 0,
                'augmented': 0,
                'total': 0
            },
            'test': {
                'preprocessed': 0,
                'augmented': 0,
                'total': 0
            },
            'total': {
                'preprocessed': 0,
                'augmented': 0,
                'total': 0
            }
        }
        
        # Update statistik dari data yang ada
        for split in ['train', 'validation', 'test']:
            if split in self._datasets:
                samples = self._datasets[split]
                stats[split]['preprocessed'] = len(samples.get('preprocessed', []))
                stats[split]['augmented'] = len(samples.get('augmented', []))
                stats[split]['total'] = max(1, len(samples.get('raw', [])))
                
                # Update total
                stats['total']['preprocessed'] += stats[split]['preprocessed']
                stats['total']['augmented'] += stats[split]['augmented']
                stats['total']['total'] += stats[split]['total']
        
        # Update UI cards if they exist
        if hasattr(self, '_dashboard_cards') and self._dashboard_cards:
            for card_name, card in self._dashboard_cards.items():
                if card_name in stats['train']:
                    value = stats['train'][card_name]
                    card.update(value=value)
    
    def _initialize_dashboard(self):
        """Initialize the dashboard with stats cards."""
        from smartcash.ui.components import create_dashboard_cards
        
        # Buat container untuk dashboard cards
        dashboard = create_dashboard_cards()
        self._dashboard_cards = dashboard["cards"]
        
        # Tambahkan container cards ke dalam layout
        if 'containers' in self.components and 'dashboard_container' in self.components['containers']:
            self.components['containers']['dashboard_container'].children = [dashboard["container"]]
        
        # Initialize stats dictionary
        stats = {
            'train': {'preprocessed': 0, 'augmented': 0, 'total': 0},
            'validation': {'preprocessed': 0, 'augmented': 0, 'total': 0},
            'test': {'preprocessed': 0, 'augmented': 0, 'total': 0},
            'total': {'preprocessed': 0, 'augmented': 0, 'total': 0}
        }
        
        # Update statistik awal
        self._update_dashboard_stats()
        
        # Update statistik dari data yang ada
        for split in ['train', 'validation', 'test']:
            if split in self._datasets:
                samples = self._datasets[split]
                stats[split]['preprocessed'] = len(samples.get('preprocessed', []))
                stats[split]['augmented'] = len(samples.get('augmented', []))
                stats[split]['total'] = max(1, len(samples.get('raw', [])))
                
                # Update total
                stats['total']['preprocessed'] += stats[split]['preprocessed']
                stats['total']['augmented'] += stats[split]['augmented']
                stats['total']['total'] += stats[split]['total']
        
        # Update UI
        for split, card in self._dashboard_cards.items():
            if split in stats:
                # Format data untuk card
                data = stats[split]
                if split != 'total':
                    # Untuk split train/val/test
                    preprocessed = data['preprocessed']
                    augmented = data['augmented']
                    total = data['total']
                    
                    preprocessed_pct = (preprocessed / total * 100) if total > 0 else 0
                    augmented_pct = (augmented / total * 100) if total > 0 else 0
                    
                    subtitle = (
                        f"{preprocessed:,} preprocessed ({preprocessed_pct:.1f}%)\n"
                        f"{augmented:,} augmented ({augmented_pct:.1f}%)"
                    )
                    progress = min(1.0, (preprocessed + augmented) / (total * 2) if total > 0 else 0)
                    
                    card.update(
                        value=total,
                        subtitle=subtitle.replace(",", "."),
                        progress=progress
                    )
                else:
                    # Untuk total card
                    total = data['total']
                    preprocessed = data['preprocessed']
                    augmented = data['augmented']
                    
                    preprocessed_pct = (preprocessed / total * 100) if total > 0 else 0
                    augmented_pct = (augmented / total * 100) if total > 0 else 0
                    
                    subtitle = (
                        f"{preprocessed:,} preprocessed ({preprocessed_pct:.1f}% of raw)\n"
                        f"{augmented:,} augmented ({augmented_pct:.1f}% of raw)"
                    )
                    
                    card.update(
                        value=total,
                        subtitle=subtitle.replace(",", "."),
                        progress=1.0
                    )
        
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
        handlers = {}
        
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
        
        if not result['success']:
            self.logger.error(result['message'])
            self.log(f"❌ {result['message']}", "error")
        else:
            self.log(f"✅ {result['message']}", "info")
    
    def _on_preprocessed_click(self, button=None):
        """Handle preprocessed sample button click."""
        from .operations import LoadPreprocessedOperation
        
        operation = LoadPreprocessedOperation(self)
        result = operation.execute()
        
        if not result['success']:
            self.logger.error(result['message'])
            self.log(f"❌ {result['message']}", "error")
        else:
            self.log(f"✅ {result['message']}", "info")
            # Update dashboard stats after loading preprocessed data
            self._update_dashboard_stats()
    
    def _on_augmented_click(self, button=None):
        """Handle augmented sample button click."""
        from .operations import LoadAugmentedOperation
        
        operation = LoadAugmentedOperation(self)
        result = operation.execute()
        
        if not result['success']:
            self.logger.error(result['message'])
            self.log(f"❌ {result['message']}", "error")
        else:
            self.log(f"✅ {result['message']}", "info")
            # Update dashboard stats after loading augmented data
            self._update_dashboard_stats()
        
    def _on_action_button_click(self, button_name: str):
        """Legacy method to handle action button clicks.
        
        Args:
            button_name: Name of the button that was clicked
        """
        self.logger.warning(f"Using legacy button handler for: {button_name}")
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
        self.logger.info(f"Memperbarui visualisasi ke tipe: {viz_type}")
        
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
                self.logger.warning(f"Tipe visualisasi tidak didukung: {viz_type}")
                
            self._current_visualization = viz_type
            
        except Exception as e:
            error_msg = f"Gagal memperbarui visualisasi: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
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


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
initialize_visualization_ui = EnhancedUIModuleFactory.create_display_function(
    VisualizationUIModule,
    function_name='initialize_visualization_ui'
)

# Create component function for programmatic use
get_visualization_component = EnhancedUIModuleFactory.create_component_function(
    VisualizationUIModule,
    function_name='get_visualization_component'
)