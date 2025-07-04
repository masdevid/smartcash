# Environment and Configuration Management

## Overview
This document covers the environment management and configuration systems used in SmartCash. These modules handle environment detection, path resolution, and configuration management across different execution environments (local, Colab, etc.).

## Table of Contents
- [Environment Manager](#environment-manager)
  - [Key Features](#environment-key-features)
  - [Usage Examples](#environment-usage)
  - [API Reference](#environment-api)
- [Configuration Manager](#configuration-manager)
  - [Key Features](#config-key-features)
  - [Usage Examples](#config-usage)
  - [API Reference](#config-api)
- [Best Practices](#best-practices)
- [Integration Guide](#integration-guide)

## Environment Manager

The `EnvironmentManager` class provides environment detection and path management across different execution contexts (local, Colab, etc.).

### Environment Key Features
- Automatic detection of Colab environment
- Google Drive mounting and detection
- Consistent path resolution across environments
- System information collection
- Singleton pattern for global access

### Environment Usage

#### Basic Initialization
```python
from smartcash.common.environment import get_environment_manager

# Get environment manager (singleton)
env = get_environment_manager()

# Check environment type
if env.is_colab:
    print("Running in Google Colab")
    
    # Mount Google Drive if needed
    if not env.is_drive_mounted:
        success, message = env.mount_drive()
        print(message)
```

#### Path Management
```python
# Get important paths
base_dir = env.base_dir
data_path = env.get_dataset_path()

print(f"Base directory: {base_dir}")
print(f"Data directory: {data_path}")
```

#### System Information
```python
# Get system information
system_info = env.get_system_info()
print(f"Environment: {system_info['environment']}")
print(f"Python version: {system_info['python_version']}")
print(f"CUDA available: {system_info.get('cuda_available', False)}")
```

### Environment API

#### Class: `EnvironmentManager`

##### Properties
- `is_colab`: `bool` - Whether running in Google Colab
- `base_dir`: `Path` - Base directory for the application
- `drive_path`: `Optional[Path]` - Path to Google Drive (if mounted)
- `is_drive_mounted`: `bool` - Whether Google Drive is mounted

##### Methods
- `get_environment_manager(base_dir=None, logger=None) -> EnvironmentManager`
  Get the singleton instance of EnvironmentManager.

- `mount_drive() -> Tuple[bool, str]`
  Mount Google Drive in Colab environment.
  Returns: (success, message)

- `get_dataset_path() -> Path`
  Get the path to the dataset directory.

- `get_system_info() -> Dict[str, Any]`
  Get system information including environment, paths, and hardware details.

- `refresh_drive_status() -> bool`
  Refresh the Google Drive mount status.

## Configuration Manager

The `SimpleConfigManager` class handles configuration management with support for multiple environments and config synchronization.

### Config Key Features
- Config file management (YAML, JSON, TOML)
- Automatic config synchronization
- Environment-specific configurations
- Config file discovery and validation
- Singleton pattern for global access

### Config Usage

#### Basic Initialization
```python
from smartcash.common.config.manager import get_config_manager

# Get config manager (singleton)
config_manager = get_config_manager()

# Load a config
config = config_manager.load_config("training.yaml")

# Access config values
learning_rate = config.get("learning_rate", 0.001)
batch_size = config.get("batch_size", 32)
```

#### Config Synchronization
```python
# Sync all configs from repo to Drive
result = config_manager.sync_configs_to_drive()
print(result['message'])

# Sync specific config with force overwrite
result = config_manager.sync_configs_to_drive(
    force_overwrite=True,
    target_configs=["model_config.yaml", "training.yaml"]
)
```

### Config API

#### Class: `SimpleConfigManager`

##### Initialization
```python
SimpleConfigManager(
    base_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    auto_sync: bool = False
)
```

##### Methods
- `get_config_manager(base_dir=None, config_file=None, auto_sync=False) -> SimpleConfigManager`
  Get the singleton instance of SimpleConfigManager.

- `load_config(config_name: str = None) -> Dict[str, Any]`
  Load configuration from file.

- `save_config(config: Dict[str, Any], config_name: str = None) -> bool`
  Save configuration to file.

- `sync_configs_to_drive(force_overwrite: bool = False, target_configs: Optional[List[str]] = None) -> Dict[str, Any]`
  Sync configuration files to Google Drive.
  
  Parameters:
  - `force_overwrite`: Whether to overwrite existing files
  - `target_configs`: List of specific config files to sync (None for all)
  
  Returns:
  ```python
  {
      'success': bool,
      'message': str,
      'synced_count': int,
      'skipped_count': int,
      'error_count': int,
      'synced_files': List[str],
      'skipped_files': List[str],
      'error_files': List[Tuple[str, str]],
      'discovered_configs': List[str],
      'success_rate': float
  }
  ```

- `discover_repo_configs() -> List[str]`
  Discover all available config files in the repository.

## Best Practices

### Environment Management
1. **Always use the singleton pattern** to access the environment manager:
   ```python
   env = get_environment_manager()
   ```

2. **Check environment before operations**:
   ```python
   if env.is_colab:
       # Colab-specific code
   ```

3. **Use the provided path methods** instead of hardcoding paths:
   ```python
   # Good
   data_path = env.get_dataset_path()
   
   # Avoid
   data_path = "/content/data"  # Hardcoded path
   ```

### Configuration Management
1. **Use environment-specific configs**:
   ```yaml
   # configs/development.yaml
   debug: true
   log_level: DEBUG
   
   # configs/production.yaml
   debug: false
   log_level: WARNING
   ```

2. **Sync configs on startup** when auto_sync is enabled:
   ```python
   config_manager = get_config_manager(auto_sync=True)
   ```

3. **Handle missing config values** gracefully:
   ```python
   # Good
   batch_size = config.get("batch_size", 32)  # Default value 32
   
   # Avoid
   batch_size = config["batch_size"]  # Raises KeyError if not found
   ```

## Integration Guide

### Initial Setup
1. **Initialize environment and config managers** early in your application:
   ```python
   from smartcash.common.environment import get_environment_manager
   from smartcash.common.config.manager import get_config_manager
   
   # Initialize environment
   env = get_environment_manager()
   
   # Initialize config with auto-sync
   config_manager = get_config_manager(auto_sync=True)
   ```

2. **Load configuration** based on environment:
   ```python
   if env.is_colab:
       config = config_manager.load_config("colab_config.yaml")
   else:
       config = config_manager.load_config("local_config.yaml")
   ```

3. **Use environment paths** for file operations:
   ```python
   import pandas as pd
   
   # Good - uses environment-specific paths
   data_path = env.get_dataset_path() / "dataset.csv"
   df = pd.read_csv(data_path)
   ```

### Advanced Usage

#### Custom Config Locations
```python
# Use a custom base directory for configs
custom_config_manager = get_config_manager(
    base_dir="/path/to/custom/configs",
    config_file="my_config.yaml"
)
```

#### Error Handling
```python
try:
    config = config_manager.load_config("critical_config.yaml")
except FileNotFoundError:
    # Handle missing config
    logger.error("Critical config not found")
    raise
```

#### Environment-Specific Code Paths
```python
if env.is_colab and env.is_drive_mounted:
    # Colab with Drive mounted
    model_path = env.drive_path / "models" / "latest.pt"
else:
    # Local or no Drive
    model_path = env.base_dir / "models" / "latest.pt"
```

This documentation provides a comprehensive guide to using the environment and configuration management systems in SmartCash. For more details, refer to the source code in `smartcash/common/environment.py` and `smartcash/common/config/manager.py`.
