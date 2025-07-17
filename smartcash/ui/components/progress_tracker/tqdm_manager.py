"""
File: smartcash/ui/components/progress_tracker/tqdm_manager.py
Deskripsi: Manager dengan layout [message][bar][percentage] dan optimasi Colab
"""

import sys
import re
from typing import Dict
from tqdm.auto import tqdm
from IPython.display import clear_output
import ipywidgets as widgets
from smartcash.ui.components.progress_tracker.progress_config import ProgressBarConfig

class TqdmManager:
    """Manager untuk tqdm progress bars dengan separate outputs per level"""
    
    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.tqdm_bars: Dict[str, tqdm] = {}
        self.progress_values: Dict[str, int] = {}
        self.progress_messages: Dict[str, str] = {}
        self.output_mapping = {
            'overall': 'overall_output',
            'primary': 'overall_output',  # Single level uses overall
            'step': 'step_output',
            'current': 'current_output'
        }
    
    def initialize_bars(self, bar_configs: list[ProgressBarConfig]):
        """Initialize tqdm progress bars in separate output widgets"""
        if not hasattr(self, 'ui_manager') or not hasattr(self.ui_manager, '_ui_components'):
            return
            
        self.close_all_bars()
        
        for bar_config in bar_configs:
            if not bar_config.visible:
                continue
                
            output_key = f"{bar_config.name}_output"
            if output_key not in self.ui_manager._ui_components:
                continue
                
            output_widget = self.ui_manager._ui_components[output_key]
            
            with output_widget:
                try:
                    clear_output(wait=True)
                    tqdm_bar = tqdm(
                        total=100,
                        desc=bar_config.description,
                        bar_format='{desc}: {bar}| {percentage:3.0f}%',
                        colour=bar_config.get_tqdm_color(),
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True
                    )
                    self.tqdm_bars[bar_config.name] = tqdm_bar
                except Exception as e:
                    print(f"Error initializing progress bar {bar_config.name}: {e}")
    
    def update_bar(self, level_name: str, progress: int, message: str = "", 
                   bar_configs: list[ProgressBarConfig] = None):
        """Update progress bar with proper message handling"""
        try:
            # Ensure progress is within bounds
            progress = max(0, min(100, int(progress)))
            
            # Initialize bar if it doesn't exist
            if level_name not in self.tqdm_bars:
                if not hasattr(self, 'ui_manager') or not hasattr(self.ui_manager, '_ui_components'):
                    return
                    
                output_key = f"{level_name}_output"
                if output_key not in self.ui_manager._ui_components:
                    return
                    
                output_widget = self.ui_manager._ui_components[output_key]
                with output_widget:
                    bar = tqdm(
                        total=100,
                        desc="",
                        bar_format='{desc}{bar}| {percentage:3.0f}%',
                        colour='#0078D7',
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True
                    )
                    self.tqdm_bars[level_name] = bar
            
            # Get the progress bar
            bar = self.tqdm_bars[level_name]
            
            # Update progress
            bar.n = progress
            
            # Update message if provided
            if message:
                clean_message = self._clean_message(message)
                formatted_desc = self._truncate_message(clean_message, 45)
                bar.set_description_str(formatted_desc)
            
            # Force refresh
            bar.refresh()
            
            # Store values
            self.progress_values[level_name] = progress
            if message:
                self.progress_messages[level_name] = message
                
        except Exception as e:
            print(f"Error updating progress bar: {e}")
    
    def set_all_complete(self, message: str, bar_configs: list[ProgressBarConfig] = None):
        """Set all bars ke complete state tanpa emoji duplikat"""
        for level_name, bar in self.tqdm_bars.items():
            bar.n = 100
            bar.refresh()
            clean_message = self._truncate_message(message, 40)
            bar.set_description(clean_message)
    
    def set_all_error(self, message: str):
        """Set all bars ke error state tanpa emoji duplikat"""
        for level_name, bar in self.tqdm_bars.items():
            current_value = self.progress_values.get(level_name, 0)
            bar.n = current_value
            bar.refresh()
            clean_message = self._truncate_message(message, 40)
            bar.set_description(clean_message)
    
    def close_all_bars(self):
        """Close semua tqdm bars dengan cleanup per output"""
        for bar_name, bar in self.tqdm_bars.items():
            try:
                bar.close()
                # Clear specific output widget
                output_attr = self.output_mapping.get(bar_name)
                if (output_attr and hasattr(self.ui_manager, '_ui_components') and 
                    output_attr in self.ui_manager._ui_components):
                    output_widget = self.ui_manager._ui_components[output_attr]
                    with output_widget:
                        clear_output(wait=True)
            except Exception:
                pass
        self.tqdm_bars.clear()
    
    def reset(self):
        """Reset manager state"""
        self.progress_values.clear()
        self.progress_messages.clear()
        self.close_all_bars()
    
    def get_progress_value(self, level_name: str) -> int:
        """Get current progress value"""
        return self.progress_values.get(level_name, 0)
    
    def get_progress_message(self, level_name: str) -> str:
        """Get current progress message"""
        return self.progress_messages.get(level_name, "")
    
    @staticmethod
    def _clean_message(message: str) -> str:
        """Clean message dari emoji duplikat dan format tidak perlu"""
        if not message:
            return message
            
        # Remove leading emojis yang akan ditambahkan ulang
        cleaned = re.sub(r'^[📊🔄⚡🔍📥☁️✅❌🚀💾🧹📁🔤🔄⚠️ℹ️]+\s*', '', message)
        # Remove progress indicators seperti (1/10), [50%], dll
        cleaned = re.sub(r'\[[\d%/]+\]|\(\d+/\d+\)', '', cleaned)
        # Clean multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip() or message
    
    @staticmethod
    def _has_emoji(text: str) -> bool:
        """Check apakah text sudah mengandung emoji untuk avoid double icon"""
        emoji_pattern = re.compile(r'[📊🔄⚡🔍📥☁️✅❌🚀💾🧹📁🔤⚠️ℹ️]')
        return bool(emoji_pattern.search(text))
    
    @staticmethod
    def _truncate_message(message: str, max_length: int) -> str:
        """Truncate message dengan ellipsis untuk fit layout"""
        if len(message) <= max_length:
            return message
        return f"{message[:max_length-3]}..."