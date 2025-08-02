#!/usr/bin/env python3
"""
YOLOv5 utilities manager for SmartCash training pipeline.

Handles lazy loading and management of YOLOv5 utilities with proper path resolution
and error handling. Provides a clean interface for YOLOv5 functionality without 
blocking module imports.

Complexity: O(1) for all operations after initial lazy loading.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Initialize logger
from smartcash.common.logger import get_logger

logger = get_logger(__name__, level="DEBUG")


class YOLOv5UtilitiesManager:
    """
    Manages YOLOv5 utilities with lazy loading and proper error handling.
    
    This class provides a single point of access for all YOLOv5 utilities,
    ensuring they're loaded only when needed and providing consistent
    error handling across the application.
    
    Time Complexity: O(1) for all operations after initialization
    Space Complexity: O(1) - stores only references to imported functions
    """
    
    def __init__(self):
        """Initialize the YOLOv5 utilities manager."""
        self._utils_cache: Optional[Dict[str, Any]] = None
        self._availability_checked = False
        self._is_available = False
        self._yolov5_root = self._resolve_yolov5_path()
        
    def _resolve_yolov5_path(self) -> Path:
        """
        Resolve YOLOv5 path relative to current file location.
        
        Returns:
            Path: Absolute path to YOLOv5 directory
            
        Time Complexity: O(1) - path operations are constant time
        """
        # Current file: smartcash/model/training/core/yolo_utils_manager.py
        # YOLOv5 is at: yolov5/ (project root)
        # Go up 5 levels: core -> training -> model -> smartcash -> project_root
        yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
        resolved_path = yolov5_path.resolve()
        
        logger.debug(f"Resolved YOLOv5 path: {resolved_path}")
        return resolved_path
    
    def _ensure_yolov5_in_path(self) -> bool:
        """
        Ensure YOLOv5 directory is in Python path.
        
        Returns:
            bool: True if path was added successfully
            
        Time Complexity: O(1) - list operations are constant time
        """
        if not self._yolov5_root.exists():
            logger.warning(f"YOLOv5 directory not found: {self._yolov5_root}")
            return False
            
        yolov5_str = str(self._yolov5_root)
        if yolov5_str not in sys.path:
            sys.path.insert(0, yolov5_str)
            logger.info(f"Added YOLOv5 path to sys.path: {yolov5_str}")
            
        return True
    
    def _import_with_cwd_change(self) -> Any:
        """
        Import YOLOv5 metrics with temporary CWD change.
        
        Returns:
            Module: YOLOv5 metrics module
            
        Raises:
            ImportError: If YOLOv5 directory not found or import fails
            
        Time Complexity: O(1) - import operations are constant time
        """
        if not self._yolov5_root.exists():
            raise ImportError(f"YOLOv5 directory not found: {self._yolov5_root}")
        
        original_cwd = os.getcwd()
        try:
            os.chdir(str(self._yolov5_root))
            return __import__('utils.metrics', fromlist=['ap_per_class', 'box_iou'])
        finally:
            os.chdir(original_cwd)
    
    def _load_yolov5_utilities(self) -> Dict[str, Any]:
        """
        Load YOLOv5 utilities with multiple import strategies.
        
        Returns:
            Dict[str, Any]: Dictionary of YOLOv5 utility functions
            
        Raises:
            ImportError: If all import attempts fail
            
        Time Complexity: O(1) - constant number of import attempts
        """
        logger.debug("Loading YOLOv5 utilities...")
        
        # Ensure YOLOv5 is in path
        self._ensure_yolov5_in_path()
        
        # Define import strategies in order of preference
        import_strategies = [
            lambda: __import__('utils.metrics', fromlist=['ap_per_class', 'box_iou']),
            lambda: __import__('yolov5.utils.metrics', fromlist=['ap_per_class', 'box_iou']),
            self._import_with_cwd_change
        ]
        
        for strategy in import_strategies:
            try:
                metrics_module = strategy()
                
                # Import corresponding general module
                general_module_name = metrics_module.__name__.replace('metrics', 'general')
                general_module = __import__(
                    general_module_name, 
                    fromlist=['xywh2xyxy', 'non_max_suppression']
                )
                
                # Create utilities dictionary
                utilities = {
                    'ap_per_class': metrics_module.ap_per_class,
                    'box_iou': metrics_module.box_iou,
                    'xywh2xyxy': general_module.xywh2xyxy,
                    'non_max_suppression': general_module.non_max_suppression
                }
                
                logger.debug("YOLOv5 utilities loaded successfully")
                return utilities
                
            except (ImportError, AttributeError) as e:
                logger.debug(f"Import strategy failed: {e}")
                continue
        
        raise ImportError(
            "YOLOv5 utilities not found - YOLOv5 is required for hierarchical validation"
        )
    
    def is_available(self) -> bool:
        """
        Check if YOLOv5 utilities are available.
        
        Returns:
            bool: True if YOLOv5 utilities can be loaded
            
        Time Complexity: O(1) after first call (cached result)
        """
        if self._availability_checked:
            return self._is_available
        
        try:
            self._load_yolov5_utilities()
            self._is_available = True
        except ImportError:
            self._is_available = False
        
        self._availability_checked = True
        return self._is_available
    
    def get_utilities(self) -> Dict[str, Any]:
        """
        Get YOLOv5 utilities with lazy loading.
        
        Returns:
            Dict[str, Any]: Dictionary containing YOLOv5 utility functions
            
        Raises:
            ImportError: If YOLOv5 utilities are not available
            
        Time Complexity: O(1) after first call (cached result)
        """
        if self._utils_cache is not None:
            return self._utils_cache
        
        if not self.is_available():
            raise ImportError("YOLOv5 utilities are not available")
        
        self._utils_cache = self._load_yolov5_utilities()
        return self._utils_cache
    
    def get_function(self, function_name: str) -> Callable:
        """
        Get a specific YOLOv5 utility function.
        
        Args:
            function_name: Name of the function to retrieve
            
        Returns:
            Callable: The requested YOLOv5 function
            
        Raises:
            KeyError: If function name is not found
            ImportError: If YOLOv5 utilities are not available
            
        Time Complexity: O(1) - dictionary lookup
        """
        utilities = self.get_utilities()
        
        if function_name not in utilities:
            available_functions = list(utilities.keys())
            raise KeyError(
                f"Function '{function_name}' not found. "
                f"Available functions: {available_functions}"
            )
        
        return utilities[function_name]


# Global instance for efficient reuse
_global_manager: Optional[YOLOv5UtilitiesManager] = None


def get_yolo_utils_manager() -> YOLOv5UtilitiesManager:
    """
    Get the global YOLOv5 utilities manager instance.
    
    Returns:
        YOLOv5UtilitiesManager: Global manager instance
        
    Time Complexity: O(1) - returns cached instance after first call
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = YOLOv5UtilitiesManager()
    
    return _global_manager


# Convenience functions for direct access to YOLOv5 utilities
def get_ap_per_class() -> Callable:
    """Get YOLOv5 ap_per_class function."""
    return get_yolo_utils_manager().get_function('ap_per_class')


def get_box_iou() -> Callable:
    """Get YOLOv5 box_iou function."""
    return get_yolo_utils_manager().get_function('box_iou')


def get_xywh2xyxy() -> Callable:
    """Get YOLOv5 xywh2xyxy function."""
    return get_yolo_utils_manager().get_function('xywh2xyxy')


def get_non_max_suppression() -> Callable:
    """Get YOLOv5 non_max_suppression function."""
    return get_yolo_utils_manager().get_function('non_max_suppression')


def is_yolov5_available() -> bool:
    """
    Check if YOLOv5 utilities are available.
    
    Returns:
        bool: True if YOLOv5 is available
        
    Time Complexity: O(1) after first call
    """
    return get_yolo_utils_manager().is_available()