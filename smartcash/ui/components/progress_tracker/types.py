"""
File: smartcash/ui/components/progress_tracker/types.py
Deskripsi: Tipe data yang digunakan oleh progress tracker
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Union, Callable, Any

class ProgressLevel(Enum):
    """Level detail progress"""
    SINGLE = auto()  # Satu progress bar
    DUAL = auto()    # Dua progress bars
    TRIPLE = auto()  # Tiga progress bars

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
    
    def __post_init__(self):
        """Validasi dan inisialisasi dengan default steps"""
        # Default steps untuk setiap level
        default_steps = {
            ProgressLevel.SINGLE: ["Progress"],
            ProgressLevel.DUAL: ["Overall", "Current"],
            ProgressLevel.TRIPLE: ["Overall", "Current", "Details"]
        }
        
        # Set default steps jika tidak disediakan
        if not self.steps and self.level in default_steps:
            self.steps = default_steps[self.level]
            
        # Validasi steps untuk level DUAL/TRIPLE
        if not self.steps and self.level != ProgressLevel.SINGLE:
            raise ValueError("Steps harus diisi untuk level DUAL atau TRIPLE")
            
        # Inisialisasi weights jika tidak disediakan
        if not self.step_weights and self.steps:
            self.step_weights = {step: 1 for step in self.steps}
    
    def get_step_weights(self) -> Dict[str, int]:
        """Get weights untuk setiap step"""
        if not self.steps:
            return {}
            
        if not self.step_weights:
            return {step: 1 for step in self.steps}
            
        # Pastikan semua step ada di weights
        weights = {step: self.step_weights.get(step, 1) for step in self.steps}
        
        return weights
    
    def get_container_height(self) -> str:
        """Get container height berdasarkan level dengan separate bars"""
        heights = {
            ProgressLevel.SINGLE: '120px',   # Header + Status + 1 bar
            ProgressLevel.DUAL: '160px',     # Header + Status + 2 bars  
            ProgressLevel.TRIPLE: '200px'    # Header + Status + 3 bars
        }
        return heights[self.level]
        
    def get_level_configs(self) -> List[Dict[str, Any]]:
        """Get configurations untuk progress bars berdasarkan level"""
        from .progress_config import get_level_configs
        return get_level_configs(self.level)
