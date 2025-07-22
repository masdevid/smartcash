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
    """Manager untuk tqdm progress bars dengan separate outputs per level dan duplicate prevention"""
    
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
        self._active_displays = set()  # Track active displays to prevent duplicates
        self._last_operation_id = None  # Track operation instances
    
    def initialize_bars(self, bar_configs: list[ProgressBarConfig]):
        """Initialize tqdm progress bars with enhanced duplicate prevention"""
        if not hasattr(self, 'ui_manager') or not hasattr(self.ui_manager, '_ui_components'):
            return
        
        # Generate operation ID to prevent conflicts
        operation_id = f"{id(self.ui_manager)}_{hash(str(bar_configs))}"
        
        # Skip if same operation is already active
        if operation_id == self._last_operation_id and self.tqdm_bars:
            return
        
        # Clean up previous instances
        self.close_all_bars()
        self._last_operation_id = operation_id
        
        # Sort bars by their intended display order
        bar_configs = sorted(bar_configs, key=lambda x: getattr(x, 'position', 0))
        
        for idx, bar_config in enumerate(bar_configs):
            if not bar_config.visible:
                continue
                
            output_key = f"{bar_config.name}_output"
            if output_key not in self.ui_manager._ui_components:
                continue
                
            output_widget = self.ui_manager._ui_components[output_key]
            
            # Clear and initialize with logging level adaptation
            with output_widget:
                try:
                    clear_output(wait=True)
                    
                    # Adaptive styling based on logging level
                    tqdm_color = self._get_adaptive_color(bar_config, idx)
                    
                    tqdm_bar = tqdm(
                        total=100,
                        desc=bar_config.description,
                        bar_format='{desc}: {bar}| {percentage:3.0f}%',
                        colour=tqdm_color,
                        leave=True,
                        file=sys.stdout,
                        dynamic_ncols=True,
                        position=idx,
                        unit_scale=True
                    )
                    self.tqdm_bars[bar_config.name] = tqdm_bar
                    self._active_displays.add(output_key)
                    
                except Exception as e:
                    print(f"Error initializing progress bar {bar_config.name}: {e}")
    
    def update_bar(self, level_name: str, progress: int, message: str = "", 
                   bar_configs: list[ProgressBarConfig] = None):
        """Update progress bar with enhanced duplicate prevention and logging level adaptation"""
        try:
            # Ensure progress is within bounds
            progress = max(0, min(100, int(progress)))
            
            # Check if we're in a notebook environment
            try:
                from IPython.display import display
                is_notebook = True
            except ImportError:
                is_notebook = False
            
            # Prevent duplicate bar creation for same operation
            output_key = f"{level_name}_output"
            if (level_name not in self.tqdm_bars and 
                output_key in self._active_displays):
                # Bar should exist, but might have been cleared - reinitialize
                self._reinitialize_single_bar(level_name, output_key)
            
            # Initialize bar if it doesn't exist
            if level_name not in self.tqdm_bars:
                if not hasattr(self, 'ui_manager') or not hasattr(self.ui_manager, '_ui_components'):
                    return
                    
                if output_key not in self.ui_manager._ui_components:
                    return
                
                # Prevent multiple bars in same output
                if output_key in self._active_displays:
                    return
                
                # In notebook, use the output widget
                if is_notebook:
                    output_widget = self.ui_manager._ui_components[output_key]
                    with output_widget:
                        clear_output(wait=True)
                        
                        # Create a new progress bar with adaptive styling
                        adaptive_color = self._get_adaptive_color_by_level(level_name)
                        
                        bar = tqdm(
                            total=100,
                            desc="",
                            bar_format='{desc}: {bar}| {percentage:3.0f}%',
                            colour=adaptive_color,
                            leave=True,
                            file=sys.stdout,
                            dynamic_ncols=True,
                            position=len(self.tqdm_bars)
                        )
                        self.tqdm_bars[level_name] = bar
                        self._active_displays.add(output_key)
                else:
                    # In script mode, create a simple progress bar
                    bar = tqdm(
                        total=100,
                        desc=level_name.capitalize(),
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {desc}',
                        colour='green',
                        leave=False,
                        file=sys.stdout
                    )
                    self.tqdm_bars[level_name] = bar
                    self._active_displays.add(output_key)
            
            # Get the progress bar
            bar = self.tqdm_bars.get(level_name)
            if not bar:
                return
            
            # Update progress and description with logging level adaptation
            if message:
                clean_message = self._clean_message(message)
                adapted_message = self._adapt_message_to_level(clean_message, level_name, progress)
                formatted_desc = self._truncate_message(adapted_message, 45)
                bar.set_description(formatted_desc, refresh=False)
            
            # Update progress
            bar.n = progress
            bar.last_print_n = progress - 1  # Force refresh
            bar.refresh()
            
            # Store values
            self.progress_values[level_name] = progress
            if message:
                self.progress_messages[level_name] = message
                
        except Exception as e:
            print(f"Error updating progress bar {level_name}: {e}")
    
    def set_all_complete(self, message: str, bar_configs: list[ProgressBarConfig] = None):
        """Set all bars ke complete state tanpa emoji duplikat"""
        for level_name, bar in self.tqdm_bars.items():
            bar.n = 100
            bar.refresh()
            clean_message = self._truncate_message(message, 40)
            bar.set_description(clean_message)
    
    def set_all_error(self, message: str):
        """Set all bars ke error state dengan indikator visual yang jelas"""
        for level_name, bar in self.tqdm_bars.items():
            current_value = self.progress_values.get(level_name, 0)
            bar.n = current_value
            
            # Update bar color to red for error state
            bar.colour = 'red'
            
            # Update bar format to show error indicator
            bar.bar_format = '{desc}: {bar}| {percentage:3.0f}% [ERROR]'
            
            # Update description with error indicator
            clean_message = f"❌ {self._truncate_message(message, 36)}"
            bar.set_description(clean_message)
            bar.refresh()
    
    def close_all_bars(self):
        """Close semua tqdm bars dengan enhanced cleanup untuk prevent duplicates"""
        for bar_name, bar in self.tqdm_bars.items():
            try:
                bar.close()
                # Clear specific output widget
                output_attr = self.output_mapping.get(bar_name, f"{bar_name}_output")
                if (output_attr and hasattr(self.ui_manager, '_ui_components') and 
                    output_attr in self.ui_manager._ui_components):
                    output_widget = self.ui_manager._ui_components[output_attr]
                    with output_widget:
                        clear_output(wait=True)
            except Exception:
                pass
        
        # Clear tracking sets
        self.tqdm_bars.clear()
        self._active_displays.clear()
        self._last_operation_id = None
    
    def reset(self):
        """Reset manager state dengan complete cleanup"""
        self.progress_values.clear()
        self.progress_messages.clear()
        self._active_displays.clear()
        self._last_operation_id = None
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
    
    def _get_adaptive_color(self, bar_config, index: int) -> str:
        """Get adaptive color based on logging level and progress bar type"""
        # Use bar_config color if available, otherwise use sequence
        if hasattr(bar_config, 'get_tqdm_color'):
            return bar_config.get_tqdm_color()
        
        # Modern color palette that adapts to logging state
        color_sequence = ['green', 'blue', 'cyan', 'yellow', 'magenta']
        return color_sequence[index % len(color_sequence)]
    
    def _get_adaptive_color_by_level(self, level_name: str) -> str:
        """Get color based on progress bar level"""
        color_map = {
            'overall': 'green',
            'primary': 'green', 
            'step': 'blue',
            'current': 'cyan'
        }
        return color_map.get(level_name, 'green')
    
    def _adapt_message_to_level(self, message: str, level_name: str, progress: int) -> str:
        """Adapt message formatting based on logging level and progress"""
        # Add level-specific prefixes for better context
        level_prefix = {
            'overall': '📊',
            'primary': '📊', 
            'step': '🔄',
            'current': '⚡'
        }.get(level_name, '📋')
        
        if progress == 100:
            return f"✅ {message}"
        elif progress > 75:
            return f"🟢 {message}"
        elif progress > 50:
            return f"{level_prefix} {message}"
        elif progress > 25:
            return f"🟡 {message}"
        else:
            return f"🔴 {message}"
    
    def _reinitialize_single_bar(self, level_name: str, output_key: str):
        """Reinitialize a single progress bar if needed"""
        if (hasattr(self.ui_manager, '_ui_components') and 
            output_key in self.ui_manager._ui_components):
            
            output_widget = self.ui_manager._ui_components[output_key]
            with output_widget:
                clear_output(wait=True)
                
            # Remove from active displays to allow recreation
            self._active_displays.discard(output_key)
            if level_name in self.tqdm_bars:
                del self.tqdm_bars[level_name]