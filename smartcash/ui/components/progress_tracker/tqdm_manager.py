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
        """Initialize tqdm progress bars di separate output widgets"""
        self.close_all_bars()
        
        for bar_config in bar_configs:
            if bar_config.visible:
                output_attr = self.output_mapping.get(bar_config.name)
                if not output_attr or not hasattr(self.ui_manager, output_attr):
                    continue
                
                output_widget = getattr(self.ui_manager, output_attr)
                
                with output_widget:
                    clear_output(wait=True)
                    tqdm_bar = tqdm(
                        total=100,
                        desc=f"{bar_config.emoji} {bar_config.description}",
                        bar_format='{desc}: {bar}| {percentage:3.0f}%',
                        colour=bar_config.get_tqdm_color(),
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True
                    )
                    self.tqdm_bars[bar_config.name] = tqdm_bar
    
    def update_bar(self, level_name: str, progress: int, message: str = "", 
                   bar_configs: list[ProgressBarConfig] = None):
        """Update progress bar dengan format [message][bar][percentage] tanpa double icon"""
        if level_name not in self.tqdm_bars:
            return
        
        progress = max(0, min(100, progress))
        bar = self.tqdm_bars[level_name]
        bar.n = progress
        bar.refresh()
        
        if message:
            clean_message = self._clean_message(message)
            config = next((c for c in (bar_configs or []) if c.name == level_name), None)
            
            # Cek apakah message sudah ada emoji, jika belum tambahkan
            if not self._has_emoji(clean_message) and config:
                formatted_desc = f"{config.emoji} {self._truncate_message(clean_message, 40)}"
            else:
                formatted_desc = self._truncate_message(clean_message, 45)
            
            bar.set_description(formatted_desc)
        
        self.progress_values[level_name] = progress
        if message:
            self.progress_messages[level_name] = message
    
    def set_all_complete(self, message: str, bar_configs: list[ProgressBarConfig] = None):
        """Set all bars ke complete state dengan format konsisten"""
        for level_name, bar in self.tqdm_bars.items():
            bar.n = 100
            bar.refresh()
            clean_message = self._truncate_message(message, 30)
            bar.set_description(f"✅ {clean_message}")
    
    def set_all_error(self, message: str):
        """Set all bars ke error state dengan format konsisten"""
        for level_name, bar in self.tqdm_bars.items():
            current_value = self.progress_values.get(level_name, 0)
            bar.n = current_value
            bar.refresh()
            clean_message = self._truncate_message(message, 30)
            bar.set_description(f"❌ {clean_message}")
    
    def close_all_bars(self):
        """Close semua tqdm bars dengan cleanup per output"""
        for bar_name, bar in self.tqdm_bars.items():
            try:
                bar.close()
                # Clear specific output widget
                output_attr = self.output_mapping.get(bar_name)
                if output_attr and hasattr(self.ui_manager, output_attr):
                    output_widget = getattr(self.ui_manager, output_attr)
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