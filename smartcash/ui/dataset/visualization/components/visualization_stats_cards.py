"""
file_path: smartcash/ui/dataset/visualization/components/visualization_stats_cards.py

Specialized stats cards for visualization module with preprocessed/augmented counts.
Implements 4 data stats cards (Train, Valid, Test, Overall) as required in development_logs.txt
"""
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

class VisualizationStatsCard:
    """Enhanced stats card for visualization module showing preprocessed/augmented data."""
    
    def __init__(self, title: str, split_name: str, card_color: str = "#3498db"):
        """Initialize visualization stats card.
        
        Args:
            title: Card title (Train, Valid, Test, Overall)
            split_name: Data split name for backend integration
            card_color: Border color for the card
        """
        self.title = title
        self.split_name = split_name
        self.card_color = card_color
        
        # Data fields
        self.raw_count = 0
        self.preprocessed_count = 0
        self.augmented_count = 0
        self.total_raw = 0  # For percentage calculation
        
        # Create UI components
        self._create_widgets()
    
    def _create_widgets(self):
        """Create widgets for the stats card."""
        # Enhanced CSS for visualization cards
        card_style = f"""
        .viz-stats-card {{
            border-left: 4px solid {self.card_color};
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            min-height: 140px;
        }}
        .viz-stats-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        .viz-card-title {{
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .viz-stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            padding: 4px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .viz-stat-row:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}
        .viz-stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            font-weight: 500;
        }}
        .viz-stat-value {{
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .viz-stat-percentage {{
            font-size: 11px;
            color: #95a5a6;
            margin-left: 8px;
        }}
        .viz-progress-mini {{
            height: 3px;
            background: #ecf0f1;
            border-radius: 2px;
            margin: 4px 0;
            overflow: hidden;
        }}
        .viz-progress-bar-preprocessed {{
            height: 100%;
            background: #27ae60;
            width: 0%;
            transition: width 0.3s ease;
        }}
        .viz-progress-bar-augmented {{
            height: 100%;
            background: #f39c12;
            width: 0%;
            transition: width 0.3s ease;
        }}
        .viz-placeholder-card {{
            border-left-color: #95a5a6 !important;
            background: linear-gradient(135deg, #f8f9fa 0%, #ecf0f1 100%);
            opacity: 0.85;
        }}
        .viz-placeholder-card .viz-card-title {{
            color: #7f8c8d;
        }}
        .viz-placeholder-card .viz-stat-value {{
            color: #95a5a6;
        }}
        """
        
        # Add styles to notebook
        display(HTML(f'<style>{card_style}</style>'))
        
        # Create HTML widget
        self.card = widgets.HTML()
        self.update()
    
    def update(self, stats_data: Optional[Dict[str, Any]] = None):
        """Update card with new statistics data with robust placeholder support.
        
        Args:
            stats_data: Dictionary containing raw, preprocessed, and augmented counts
        """
        # Initialize with default values
        self.raw_count = 0
        self.preprocessed_count = 0
        self.augmented_count = 0
        self.total_raw = 0
        
        # Flag to track if we have real data or placeholder
        self.is_placeholder = True
        self.status_message = "Click refresh to load data"
        
        if stats_data and stats_data.get('success', False):
            # We have valid data - extract actual statistics
            dataset_success = stats_data.get('dataset_stats', {}).get('success', True)
            
            if dataset_success:
                split_data = stats_data.get('dataset_stats', {}).get('by_split', {}).get(self.split_name, {})
                aug_data = stats_data.get('augmentation_stats', {}).get('by_split', {}).get(self.split_name, {})
                
                self.raw_count = split_data.get('raw', 0)
                self.preprocessed_count = split_data.get('preprocessed', 0) 
                self.augmented_count = aug_data.get('file_count', 0)
                
                # For overall card, sum all splits
                if self.split_name == 'overall':
                    all_splits = stats_data.get('dataset_stats', {}).get('by_split', {})
                    aug_splits = stats_data.get('augmentation_stats', {}).get('by_split', {})
                    
                    self.raw_count = sum(split.get('raw', 0) for split in all_splits.values())
                    self.preprocessed_count = sum(split.get('preprocessed', 0) for split in all_splits.values())
                    self.augmented_count = sum(aug.get('file_count', 0) for aug in aug_splits.values())
                
                # Total raw for percentage calculation
                overview = stats_data.get('dataset_stats', {}).get('overview', {})
                self.total_raw = overview.get('total_files', 0) or max(self.raw_count, 1)
                
                # Check if we actually have data or if it's just zeros
                if self.raw_count > 0 or self.preprocessed_count > 0 or self.augmented_count > 0:
                    self.is_placeholder = False
                    self.status_message = f"Updated: {stats_data.get('last_updated', 'Unknown time')}"
                else:
                    self.status_message = "No data found - check data directory"
            else:
                # Backend failed to load data
                self.status_message = "Backend unavailable - showing placeholder"
        
        # Calculate percentages
        preprocessed_pct = self._calculate_percentage(self.preprocessed_count, self.raw_count)
        augmented_pct = self._calculate_percentage(self.augmented_count, self.raw_count)
        
        # Calculate progress bars
        preprocessed_progress = min((self.preprocessed_count / max(self.raw_count, 1)) * 100, 100)
        augmented_progress = min((self.augmented_count / max(self.raw_count, 1)) * 100, 100)
        
        # Add placeholder styling if needed
        card_class = "viz-stats-card"
        if self.is_placeholder:
            card_class += " viz-placeholder-card"
        
        # Status indicator emoji
        status_emoji = "📊" if not self.is_placeholder else "📋"
        
        # Generate HTML with enhanced placeholder support
        html = f"""
        <div class="{card_class}">
            <div class="viz-card-title">{status_emoji} {self.title} Dataset</div>
            
            <div class="viz-stat-row">
                <span class="viz-stat-label">Preprocessed:</span>
                <span class="viz-stat-value">
                    {self.preprocessed_count:,}
                    <span class="viz-stat-percentage">({preprocessed_pct}%)</span>
                </span>
            </div>
            <div class="viz-progress-mini">
                <div class="viz-progress-bar-preprocessed" style="width: {preprocessed_progress:.1f}%;"></div>
            </div>
            
            <div class="viz-stat-row">
                <span class="viz-stat-label">Augmented:</span>
                <span class="viz-stat-value">
                    {self.augmented_count:,}
                    <span class="viz-stat-percentage">({augmented_pct}%)</span>
                </span>
            </div>
            <div class="viz-progress-mini">
                <div class="viz-progress-bar-augmented" style="width: {augmented_progress:.1f}%;"></div>
            </div>
            
            <div class="viz-stat-row">
                <span class="viz-stat-label">Raw Images:</span>
                <span class="viz-stat-value">{self.raw_count:,}</span>
            </div>
            
            <div class="viz-stat-row" style="border-top: 1px solid #ecf0f1; padding-top: 8px; margin-top: 8px;">
                <span class="viz-stat-label" style="font-size: 10px; color: #95a5a6;">
                    {self.status_message}
                </span>
            </div>
        </div>
        """
        
        self.card.value = html
    
    def _calculate_percentage(self, count: int, total: int) -> int:
        """Calculate percentage with safe division."""
        if total == 0:
            return 0
        return round((count / total) * 100)
    
    def get_widget(self) -> widgets.Widget:
        """Get the card widget."""
        return self.card


class VisualizationStatsCardContainer:
    """Container for all 4 visualization stats cards (Train, Valid, Test, Overall)."""
    
    def __init__(self):
        """Initialize the stats cards container."""
        # Card configurations
        self.card_configs = [
            {"title": "Train", "split": "train", "color": "#27ae60"},
            {"title": "Valid", "split": "valid", "color": "#3498db"},
            {"title": "Test", "split": "test", "color": "#e74c3c"},
            {"title": "Overall", "split": "overall", "color": "#9b59b6"}
        ]
        
        # Create cards
        self.cards = {}
        for config in self.card_configs:
            card = VisualizationStatsCard(
                title=config["title"],
                split_name=config["split"],
                card_color=config["color"]
            )
            self.cards[config["split"]] = card
        
        # Create container layout
        self._create_container()
    
    def _create_container(self):
        """Create the container layout for all cards."""
        # Create card widgets
        card_widgets = [card.get_widget() for card in self.cards.values()]
        
        # Create responsive grid layout
        self.container = widgets.HBox(
            children=card_widgets,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='space-between',
                align_items='stretch',
                width='100%',
                gap='10px'
            )
        )
        
        # Individual card layouts for equal sizing
        for widget in card_widgets:
            widget.layout = widgets.Layout(
                width='24%',
                min_width='200px',
                max_width='300px'
            )
    
    def update_all_cards(self, comprehensive_stats: Dict[str, Any]):
        """Update all cards with comprehensive statistics.
        
        Args:
            comprehensive_stats: Complete stats from refresh operation
        """
        for card in self.cards.values():
            card.update(comprehensive_stats)
    
    def get_container(self) -> widgets.Widget:
        """Get the main container widget."""
        return self.container
    
    def get_card(self, split_name: str) -> Optional[VisualizationStatsCard]:
        """Get a specific card by split name.
        
        Args:
            split_name: Split name ('train', 'valid', 'test', 'overall')
            
        Returns:
            The requested card or None if not found
        """
        return self.cards.get(split_name)


def create_visualization_stats_dashboard() -> VisualizationStatsCardContainer:
    """Create complete visualization stats dashboard with 4 cards.
    
    Returns:
        Configured stats card container ready for display
        
    Usage:
        >>> dashboard = create_visualization_stats_dashboard()
        >>> display(dashboard.get_container())
        >>> # Later update with real data:
        >>> dashboard.update_all_cards(stats_data)
    """
    return VisualizationStatsCardContainer()