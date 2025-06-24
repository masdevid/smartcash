"""
File: smartcash/ui/components/progress_tracker/progress_config.py
Deskripsi: Configuration untuk progress tracking dengan auto hide
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .types import ProgressConfig, ProgressLevel

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

def get_level_configs(level: ProgressLevel) -> List[ProgressBarConfig]:
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
    return level_configs[level]

def get_default_weights(steps: List[str]) -> Dict[str, int]:
    """Generate equal weights untuk semua steps"""
    if not steps:
        return {}
    
    num_steps = len(steps)
    base_weight = 100 // num_steps
    remainder = 100 % num_steps
    
    weights = {}
    for i, step in enumerate(steps):
        weights[step] = base_weight + (1 if i < remainder else 0)
    
    return weights

def get_container_height(level: ProgressLevel) -> str:
    """Get container height berdasarkan level dengan separate bars"""
    heights = {
        ProgressLevel.SINGLE: '120px',   # Header + Status + 1 bar
        ProgressLevel.DUAL: '160px',     # Header + Status + 2 bars  
        ProgressLevel.TRIPLE: '200px'    # Header + Status + 3 bars
    }
    return heights[level]