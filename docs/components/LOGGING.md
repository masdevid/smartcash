# Logging Components

Components for displaying and managing application logs in a user-friendly way.

## Components

### LogAccordion (`log_accordion.py`)
Collapsible log viewer that can display multiple log entries.

**Props:**
- `logs` (List[LogEntry]): List of log entries to display
- `max_entries` (int, optional): Maximum number of entries to show
- `filter_level` (str, optional): Minimum log level to display
- `on_clear` (callable, optional): Callback when logs are cleared

### LogEntry (`log_entry.py`)
Individual log entry component.

**Props:**
- `message` (str): Log message
- `level` (str): Log level ('debug', 'info', 'warning', 'error', 'critical')
- `timestamp` (datetime): When the log was created
- `source` (str, optional): Source of the log
- `details` (Any, optional): Additional log data

### LogLevel (`log_level.py`)
Log level definitions and utilities.

**Constants:**
- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

**Methods:**
- `get_level_name(level: int) -> str`
- `get_level_value(level: str) -> int`
- `get_level_color(level: str) -> str`

## Usage

### Basic Logging
```python
from smartcash.ui.components.log_accordion import LogAccordion, LogEntry
from datetime import datetime

# Create log entries
logs = [
    LogEntry(
        message="Application started",
        level="info",
        timestamp=datetime.now(),
        source="app"
    )
]

# Display logs
log_viewer = LogAccordion(logs=logs)
```

### Logging with Context Manager
```python
from contextlib import contextmanager
from smartcash.ui.components.log_accordion import LogAccordion

@contextmanager
def logging_context(log_accordion: LogAccordion, message: str):
    try:
        log_accordion.add_log("info", f"Starting: {message}")
        yield
        log_accordion.add_log("info", f"Completed: {message}")
    except Exception as e:
        log_accordion.add_log("error", f"Failed: {message} - {str(e)}")
        raise

# Usage
with logging_context(log_viewer, "Processing data"):
    # Your code here
    pass
```

## Best Practices

- Use appropriate log levels
- Include relevant context in log messages
- Be mindful of log volume
- Consider privacy and security when logging sensitive data
- Use structured logging when possible

## Integration

### With Python's logging Module
```python
import logging
from smartcash.ui.components.log_accordion import LogAccordion, LogEntry

class UIHandler(logging.Handler):
    def __init__(self, log_accordion: LogAccordion):
        super().__init__()
        self.log_accordion = log_accordion
    
    def emit(self, record):
        log_entry = LogEntry(
            message=self.format(record),
            level=record.levelname.lower(),
            timestamp=datetime.fromtimestamp(record.created)
        )
        self.log_accordion.add_log_entry(log_entry)

# Setup logging
log_accordion = LogAccordion()
handler = UIHandler(log_accordion)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.basicConfig(handlers=[handler], level=logging.INFO)
```

## Features

- Real-time log updates
- Filter logs by level
- Search within logs
- Copy log entries to clipboard
- Expand/collapse log details
- Auto-scrolling to latest logs

## Performance

- Virtualized rendering for large log volumes
- Configurable maximum log entries
- Efficient updates using batched rendering
- Memory-efficient log storage
