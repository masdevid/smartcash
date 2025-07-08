"""
File: smartcash/ui/dataset/visualization/handlers/visualization_ui_handler.py
Description: UI handler for the visualization module
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
from IPython.display import display
import os

from smartcash.ui.core.handlers.ui_handler import UIHandler
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager

class VisualizationUIHandler(UIHandler):
    """UI handler for the visualization module."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None, scanner=None):
        """Initialize the visualization UI handler.
        
        Args:
            ui_components: UI components to handle
            logger: Logger instance
            scanner: Optional dataset scanner instance
        """
        super().__init__(ui_components=ui_components, logger=logger or get_logger(__name__))
        self.env_manager = get_environment_manager()
        self.dataset_path = self.env_manager.get_dataset_path()
        self.scanner = scanner or self._init_scanner()
    
    def _init_scanner(self):
        """Initialize dataset scanner."""
        try:
            from smartcash.dataset.downloader import create_dataset_scanner
            scanner = create_dataset_scanner(logger=self.logger)
            self.logger.info("✅ Dataset scanner initialized successfully")
            return scanner
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize dataset scanner: {str(e)}")
            return None
        
    def refresh_data(self, button):
        """Refresh data statistics.
        
        Args:
            button: Button that triggered the event
        """
        try:
            # Show loading indicator
            self._set_loading(True)
                
            # Get latest data
            stats = self._get_dataset_stats()
            
            # Update UI with latest data
            self._update_ui(stats)
            
            # Log success
            self.logger.info("Data statistics updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update data statistics: {str(e)}", exc_info=True)
            self._show_error(f"Failed to update data: {str(e)}")
            
        finally:
            # Hide loading indicator
            self._set_loading(False)
    
    def _set_loading(self, loading: bool):
        """Set loading state.
        
        Args:
            loading: Whether to show loading state
        """
        if 'loading_indicator' in self.ui_components:
            self.ui_components['loading_indicator'].value = loading
    
    def _get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics from DatasetScanner.
        
        Returns:
            Dictionary containing dataset statistics for each split
        """
        stats = {
            'train': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
            'valid': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
            'test': {'raw': 0, 'preprocessed': 0, 'augmented': 0}
        }
        
        try:
            # Get statistics from DatasetScanner if available
            if self.scanner:
                scan_result = self.scanner.scan_existing_dataset_parallel()
                
                if 'splits' in scan_result:
                    for split in ['train', 'valid', 'test']:
                        if split in scan_result['splits']:
                            stats[split]['raw'] = scan_result['splits'][split].get('images', 0)
            
            # Update with preprocessed and augmented data
            self._update_preprocessed_stats(stats)
            self._update_augmented_stats(stats)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get dataset statistics: {str(e)}", exc_info=True)
        
        return stats
    
    def _update_preprocessed_stats(self, stats: Dict[str, Any]) -> None:
        """Update stats with preprocessed data count.
        
        Args:
            stats: Dictionary to store statistics
        """
        try:
            preprocessed_dir = self.dataset_path / 'preprocessed'
            if preprocessed_dir.exists():
                for split in ['train', 'valid', 'test']:
                    split_dir = preprocessed_dir / split
                    if split_dir.exists():
                        # Count image files in preprocessed directory
                        img_count = len([f for f in split_dir.glob('**/*') 
                                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                        stats[split]['preprocessed'] = img_count
        except Exception as e:
            self.logger.warning(f"⚠️ Could not count preprocessed data: {str(e)}")
    
    def _update_augmented_stats(self, stats: Dict[str, Any]) -> None:
        """Update stats with augmented data count.
        
        Args:
            stats: Dictionary to store statistics
        """
        try:
            augmented_dir = self.dataset_path / 'augmented'
            if augmented_dir.exists():
                for split in ['train', 'valid', 'test']:
                    split_dir = augmented_dir / split
                    if split_dir.exists():
                        # Count image files in augmented directory
                        img_count = len([f for f in split_dir.glob('**/*') 
                                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                        stats[split]['augmented'] = img_count
        except Exception as e:
            self.logger.warning(f"⚠️ Could not count augmented data: {str(e)}")
    
    def _update_ui(self, stats: Dict[str, Any]) -> None:
        """Update UI with latest statistics.
        
        Args:
            stats: Latest statistics data
        """
        for split in ['train', 'valid', 'test']:
            if split in stats and f'{split}_card' in self.ui_components:
                data = stats[split]
                raw = data.get('raw', 0)
                preprocessed = data.get('preprocessed', 0)
                augmented = data.get('augmented', 0)
                
                # Calculate percentages
                preprocessed_pct = (preprocessed / raw * 100) if raw > 0 else 0
                augmented_pct = (augmented / raw * 100) if raw > 0 else 0
                
                # Update preprocessed label
                preprocessed_label = self.ui_components[f'{split}_preprocessed_label']
                preprocessed_label.value = (
                    f'<div style="padding: 5px 10px;">'
                    f'Preprocessed: <span style="font-weight: bold;">'
                    f'{preprocessed}/{raw} ({preprocessed_pct:.1f}%)</span>'
                    f'</div>'
                )
                
                # Update augmented label
                augmented_label = self.ui_components[f'{split}_augmented_label']
                augmented_label.value = (
                    f'<div style="padding: 5px 10px 15px 10px;">'
                    f'Augmented: <span style="font-weight: bold;">'
                    f'{augmented}/{raw} ({augmented_pct:.1f}%)</span>'
                    f'</div>'
                )
    
    def _show_error(self, message: str) -> None:
        """Show error message in the log accordion.
        
        Args:
            message: Error message to display
        """
        if 'log_output' in self.ui_components:
            with self.ui_components['log_output']:
                print(f"[ERROR] {message}")
            
            # Open log accordion if closed
            if 'log_accordion' in self.ui_components:
                self.ui_components['log_accordion'].selected_index = 0
