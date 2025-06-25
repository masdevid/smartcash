"""
Common type definitions for the dependency management system.
"""
from typing import Dict, Any, Optional, Union, List, Callable

# Type alias for UI components dictionary
UIComponents = Dict[str, Any]

# Type alias for handler functions
HandlerFunction = Callable[[UIComponents], None]

# Type alias for handler map
HandlerMap = Dict[str, HandlerFunction]

__all__ = [
    'UIComponents',
    'HandlerFunction',
    'HandlerMap'
]
