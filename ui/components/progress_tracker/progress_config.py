"""
File: smartcash/ui/components/progress_tracker/progress_config.py
Deskripsi: Updated configuration dengan auto hide 1 jam dan tanpa step info
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List

class ProgressLevel(Enum):
    """Progress level enum untuk type safety"""
    SINGLE = 1
    DUAL = 2
    TRIPLE = 3

@dataclass
class ProgressBarConfig:
    """Configuration untuk individual progress bar"""
    name: str
    description: str
    emoji: str
    color: str
    position: int
    visible: bool = True
    
    def get_tqdm_color(self) -> str:
        """Convert hex color ke tqdm color name"""
        color_mapping = {
            '#28a745': 'green',
            '#007bff': 'blue', 
            '#ffc107': 'yellow',
            '#dc3545': 'red',
            '#6c757d': 'white'
        }
        return color_mapping.get(self.color, 'blue')

@dataclass
class ProgressConfig:
    """Configuration untuk progress tracking dengan 1 jam auto hide"""
    level: ProgressLevel = ProgressLevel.SINGLE
    operation: str = "Process"
    steps: List[str] = field(default_factory=list)
    step_weights: Dict[str, int] = field(default_factory=dict)
    auto_advance: bool = True
    auto_hide: bool = False  # Default false
    auto_hide_delay: float = 3600.0  # 1 jam dalam detik
    animation_speed: float = 0.1
    width_adjustment: int = 0
    show_step_info: bool = False  # Hapus step info
    
    def get_level_configs(self) -> List[ProgressBarConfig]:
        """Get bar configurations dengan warna hijau seragam"""
        level_configs = {
            ProgressLevel.SINGLE: [
                ProgressBarConfig("primary", "Progress", "📊", "#28a745", 0)
            ],
            ProgressLevel.DUAL: [
                ProgressBarConfig("overall", "Overall Progress", "📊", "#28a745", 0),
                ProgressBarConfig("current", "Current Operation", "⚡", "#28a745", 1)
            ],
            ProgressLevel.TRIPLE: [
                ProgressBarConfig("overall", "Overall Progress", "📊", "#28a745", 0),
                ProgressBarConfig("step", "Step Progress", "🔄", "#28a745", 1),
                ProgressBarConfig("current", "Current Operation", "⚡", "#28a745", 2)
            ]
        }
        return level_configs[self.level]
    
    def get_default_weights(self) -> Dict[str, int]:
        """Generate equal weights untuk semua steps"""
        if not self.steps:
            return {}
        
        num_steps = len(self.steps)
        base_weight = 100 // num_steps
        remainder = 100 % num_steps
        
        weights = {}
        for i, step in enumerate(self.steps):
            weights[step] = base_weight + (1 if i < remainder else 0)
        
        return weights
    
    def get_container_height(self) -> str:
        """Get container height berdasarkan level dengan separate bars"""
        heights = {
            ProgressLevel.SINGLE: '120px',   # Header + Status + 1 bar
            ProgressLevel.DUAL: '160px',     # Header + Status + 2 bars  
            ProgressLevel.TRIPLE: '200px'    # Header + Status + 3 bars
        }
        return heights[self.level]