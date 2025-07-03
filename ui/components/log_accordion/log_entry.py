"""
Log entry model for the LogAccordion component.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from .log_level import LogLevel


@dataclass
class LogEntry:
    """Represents a single log entry with metadata."""
    
    message: str
    level: LogLevel = LogLevel.INFO
    namespace: Optional[str] = None
    module: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    count: int = 1
    show_duplicate_indicator: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the log entry to a dictionary."""
        return {
            'message': self.message,
            'level': self.level,
            'namespace': self.namespace,
            'module': self.module,
            'timestamp': self.timestamp,
            'count': self.count,
            'show_duplicate_indicator': self.show_duplicate_indicator
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create a LogEntry from a dictionary."""
        return cls(
            message=data['message'],
            level=LogLevel(data['level']) if isinstance(data['level'], str) else data['level'],
            namespace=data.get('namespace'),
            module=data.get('module'),
            timestamp=data['timestamp'],
            count=data.get('count', 1),
            show_duplicate_indicator=data.get('show_duplicate_indicator', False)
        )
    
    def is_duplicate_of(self, other: 'LogEntry', time_window_ms: int = 1000) -> bool:
        """Check if this log entry is a duplicate of another within the given time window."""
        if not isinstance(other, LogEntry):
            return False
            
        time_diff = abs((self.timestamp - other.timestamp).total_seconds() * 1000)
        
        return (
            self.message == other.message and
            self.level == other.level and
            self.namespace == other.namespace and
            self.module == other.module and
            time_diff < time_window_ms
        )
    
    def increment_duplicate_count(self, max_count: int = 2) -> None:
        """Increment the duplicate count, up to the specified maximum."""
        if self.count < max_count:
            self.count += 1
        self.show_duplicate_indicator = True
