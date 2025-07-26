"""
file_path: smartcash/ui/components/visualization_container.py

Custom container untuk menata layout visualisasi dengan dashboard cards.
"""
from typing import Dict, Any, Optional, List
import ipywidgets as widgets

class VisualizationContainer:
    """Container khusus untuk tata letak visualisasi dengan dashboard cards."""
    
    def __init__(self, **kwargs):
        """Inisialisasi container visualisasi.
        
        Args:
            **kwargs: Argumen layout tambahan
        """
        self._dashboard_cards = None
        self._container = None
        self._initialize_ui()
    
    def _initialize_ui(self):
        """Inisialisasi komponen UI."""
        # Create a simple VBox container for visualization content
        self._container = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                margin='10px 0',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )
        )
    
    @property
    def container(self):
        """Get the main container widget."""
        return self._container
    
    def update_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """Perbarui statistik pada dashboard cards.
        
        Args:
            stats: Dictionary berisi statistik untuk setiap split
                Contoh:
                {
                    'train': {
                        'preprocessed': 1000,
                        'augmented': 5000,
                        'total': 1200  # total raw images
                    },
                    'validation': {...},
                    'test': {...},
                    'total': {...}
                }
        """
        if not self._dashboard_cards:
            return
        
        # Hitung total untuk persentase
        total_raw = stats.get('total', {}).get('total', 1)
        
        for split, card in self._dashboard_cards.items():
            if split not in stats:
                continue
                
            split_stats = stats[split]
            
            # Format subtitle dengan statistik
            if split != 'total':
                # Untuk split train/val/test
                preprocessed = split_stats.get('preprocessed', 0)
                augmented = split_stats.get('augmented', 0)
                total = split_stats.get('total', 1)
                
                preprocessed_pct = (preprocessed / total) * 100 if total > 0 else 0
                augmented_pct = (augmented / total) * 100 if total > 0 else 0
                
                subtitle = (
                    f"{preprocessed:,} preprocessed ({preprocessed_pct:.1f}%)\n"
                    f"{augmented:,} augmented ({augmented_pct:.1f}%)"
                )
                progress = min(1.0, (preprocessed + augmented) / (total * 2) if total > 0 else 0)
                
                card.update(
                    value=split_stats.get('total', 0),
                    subtitle=subtitle.replace(",", "."),
                    progress=progress
                )
            else:
                # Untuk total card
                preprocessed = split_stats.get('preprocessed', 0)
                augmented = split_stats.get('augmented', 0)
                total = split_stats.get('total', 1)
                
                preprocessed_pct = (preprocessed / total) * 100 if total > 0 else 0
                augmented_pct = (augmented / total) * 100 if total > 0 else 0
                
                subtitle = (
                    f"{preprocessed:,} preprocessed ({preprocessed_pct:.1f}% of raw)\n"
                    f"{augmented:,} augmented ({augmented_pct:.1f}% of raw)"
                )
                
                card.update(
                    value=total,
                    subtitle=subtitle.replace(",", "."),
                    progress=1.0
                )
