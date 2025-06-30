# Logging Migration Guide

This guide provides instructions for migrating from the deprecated logging utilities to the new `UILogger` implementation.

## Table of Contents
1. [Overview](#overview)
2. [Migrating from `buffered_logger`](#migrating-from-buffered_logger)
3. [Migrating from `simple_logger`](#migrating-from-simple_logger)
4. [New Features in `UILogger`](#new-features-in-uilogger)
5. [Examples](#examples)

## Overview

The logging system has been consolidated into a single, more powerful `UILogger` class that combines the functionality of the previous logging utilities. The following modules are now deprecated:

- `buffered_logger` - Replaced by `UILogger` with buffering enabled
- `simple_logger` - Replaced by `UILogger`

## Migrating from `buffered_logger`

### Old Code
```python
from smartcash.ui.utils.buffered_logger import BufferedLogger, create_buffered_logger

# Create a buffered logger
logger = create_buffered_logger()
logger.info("This is a buffered log message")

# Later, when UI is ready
ui_logger = UILogger(ui_components={})
logger.flush_to_ui_logger(ui_logger)
```

### New Code
```python
from smartcash.ui.utils.ui_logger import UILogger

# Create a UILogger with buffering enabled
logger = UILogger(ui_components={}, enable_buffering=True)
logger.info("This is a buffered log message")

# Later, when UI is ready
logger.flush_buffered_logs()
```

### Key Changes
- Replace `create_buffered_logger()` with `UILogger(..., enable_buffering=True)`
- Replace `flush_to_ui_logger(ui_logger)` with `flush_buffered_logs()`
- All buffered logs are automatically flushed when `flush_buffered_logs()` is called

## Migrating from `simple_logger`

### Old Code
```python
from smartcash.ui.utils.simple_logger import create_simple_logger

logger = create_simple_logger()
logger.info("This is a simple log message")
logger.success("Operation completed successfully")
```

### New Code
```python
from smartcash.ui.utils.ui_logger import UILogger

logger = UILogger(ui_components={})
logger.info("This is a simple log message")
logger.success("Operation completed successfully")
```

### Key Changes
- Replace `create_simple_logger()` with `UILogger(ui_components={})`
- All logging methods (`debug`, `info`, `warning`, `error`, `critical`, `success`) work the same way
- No need to pass UI components if you're only logging to console

## New Features in `UILogger`

### Namespace Support
```python
# Automatically detects module name for namespacing
logger = UILogger(ui_components={}, name=__name__)
```

### File Logging
```python
# Enable file logging with rotation
logger = UILogger(
    ui_components={},
    log_to_file=True,
    log_dir="/path/to/logs",
    log_level=logging.INFO
)
```

### Stdout Suppression
```python
# Suppress stdout from other libraries
logger = UILogger(ui_components={}, suppress_stdout=True)
```

## Examples

### Basic Usage
```python
from smartcash.ui.utils.ui_logger import UILogger

# Create a logger
logger = UILogger(ui_components={})


# Log messages at different levels
logger.debug("Debug message")
logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
logger.success("Success message")

# Log an exception
try:
    1 / 0
except Exception as e:
    logger.exception("An error occurred")
```

### Advanced Usage with Buffering
```python
from smartcash.ui.utils.ui_logger import UILogger

# Create a logger with buffering enabled
logger = UILogger(ui_components={}, enable_buffering=True)

# These messages will be buffered
logger.info("This message will be buffered")
logger.warning("This warning will be buffered too")

# Later, when UI is ready
logger.flush_buffered_logs()  # All buffered messages are now displayed

# Future messages are displayed immediately
logger.info("This message appears right away")
```
