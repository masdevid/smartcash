"""
File: smartcash/ui/components/progress_tracker/tqdm_manager.py
Deskripsi: Manager untuk tqdm progress bars dengan Colab optimization
"""

import sys
import re
from typing import Dict
from tqdm.auto import tqdm
from IPython.display import clear_output
import ipywidgets as widgets
from .progress_config import ProgressBarConfig

class TqdmManager:
    """Manager untuk tqdm progress bars dengan Colab optimization"""
    
    def __init__(self, output_widget: widgets.Output):
        self.output_widget = output_widget
        self.tqdm_bars: Dict[str, tqdm] = {}
        self.progress_values: Dict[str, int] = {}
        self.progress_messages: Dict[str, str] = {}
    
    def initialize_bars(self, bar_configs: list[ProgressBarConfig]):
        """Initialize tqdm progress bars untuk setiap level"""
        self.close_all_bars()
        
        with self.output_widget:
            clear_output(wait=True)
            
            for bar_config in bar_configs:
                if bar_config.visible:
                    tqdm_bar = tqdm(
                        total=100,
                        desc=f"{bar_config.emoji} {bar_config.description}",
                        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
                        colour=bar_config.get_tqdm_color(),
                        position=bar_config.position,
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True
                    )
                    self.tqdm_bars[bar_config.name] = tqdm_bar
    
    def update_bar(self, level_name: str, progress: int, message: str = "", 
                   bar_configs: list[ProgressBarConfig] = None):
        """Update specific progress bar dengan message cleaning"""
        if level_name not in self.tqdm_bars:
            return
        
        progress = max(0, min(100, progress))
        bar = self.tqdm_bars[level_name]
        bar.n = progress
        bar.refresh()
        
        if message:
            clean_message = self._clean_message(message)
            config = next((c for c in (bar_configs or []) if c.name == level_name), None)
            emoji = config.emoji if config else "📊"
            bar.set_description(f"{emoji} {self._truncate_message(clean_message, 40)}")
        
        self.progress_values[level_name] = progress
        if message:
            self.progress_messages[level_name] = message
    
    def set_all_complete(self, message: str, bar_configs: list[ProgressBarConfig] = None):
        """Set all bars ke complete state"""
        for level_name, bar in self.tqdm_bars.items():
            bar.n = 100
            bar.refresh()
            clean_message = self._truncate_message(message, 35)
            bar.set_description(f"✅ {clean_message}")
    
    def set_all_error(self, message: str):
        """Set all bars ke error state"""
        for level_name, bar in self.tqdm_bars.items():
            current_value = self.progress_values.get(level_name, 0)
            bar.n = current_value
            bar.refresh()
            clean_message = self._truncate_message(message, 35)
            bar.set_description(f"❌ {clean_message}")
    
    def close_all_bars(self):
        """Close semua tqdm bars dengan cleanup"""
        for bar in self.tqdm_bars.values():
            bar.close()
        self.tqdm_bars.clear()
        
        with self.output_widget:
            clear_output(wait=True)
    
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
        """Clean message dari emoji duplikat"""
        cleaned = re.sub(r'^[📊🔄⚡🔍📥☁️✅❌🚀]+\s*', '', message)
        return cleaned.strip() or message
    
    @staticmethod
    def _truncate_message(message: str, max_length: int) -> str:
        """Truncate message dengan ellipsis"""
        return message if len(message) <= max_length else f"{message[:max_length-3]}..."