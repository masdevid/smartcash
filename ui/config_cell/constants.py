"""
File: smartcash/ui/config_cell/constants.py
Deskripsi: Shared constants for config cell components
"""
from enum import Enum

class StatusType(str, Enum):
    """Status types for UI feedback."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

# Status color mapping for status bars
STATUS_COLORS = {
    StatusType.INFO: 'blue',
    StatusType.SUCCESS: 'green',
    StatusType.WARNING: 'orange',
    StatusType.ERROR: 'red',
}
