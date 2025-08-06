"""
Base phase manager class.

Provides the common interface and shared functionality for all phase managers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger


class BasePhaseManager(ABC):
    """Abstract base class for all phase managers."""
    
    def __init__(self, model, model_api, config, progress_tracker):
        """
        Initialize base phase manager.
        
        Args:
            model: PyTorch model
            model_api: Model API instance
            config: Training configuration
            progress_tracker: Progress tracking instance
        """
        self.model = model
        self.model_api = model_api
        self.config = config
        self.progress_tracker = progress_tracker
        self.logger = get_logger(self.__class__.__name__)
        
        # Common state
        self._is_single_phase = False
        self._current_phase = None
        
    @abstractmethod
    def setup_phase(self, phase_num: int, **kwargs) -> Dict[str, Any]:
        """
        Set up a training phase with required components.
        
        Args:
            phase_num: Phase number to set up
            **kwargs: Additional phase setup parameters
            
        Returns:
            Dictionary containing phase setup components
        """
        pass
    
    @abstractmethod
    def execute_phase(self, phase_num: int, **kwargs) -> Dict[str, Any]:
        """
        Execute a training phase.
        
        Args:
            phase_num: Phase number to execute
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing phase execution results
        """
        pass
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set whether training is in single phase mode."""
        self._is_single_phase = is_single_phase
        if self.logger:
            mode = "single" if is_single_phase else "multi"
            self.logger.info(f"Phase manager set to {mode}-phase mode")
    
    def is_single_phase(self) -> bool:
        """Check if training is in single phase mode."""
        return self._is_single_phase
    
    def get_current_phase(self) -> Optional[int]:
        """Get the current phase number."""
        return self._current_phase
    
    def _set_current_phase(self, phase_num: int):
        """Set the current phase number."""
        self._current_phase = phase_num