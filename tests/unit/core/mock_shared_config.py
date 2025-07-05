"""
Mock implementation of shared configuration manager for testing.
"""
from typing import Any, Dict, Optional


class ConfigManager:
    """Mock ConfigManager for testing."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the mock ConfigManager."""
        self._store = {}
        self._defaults = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._store.get(key, self._defaults.get(key, default))
    
    def set(self, key: str, value: Any) -> bool:
        """Set a configuration value."""
        self._store[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """Delete a configuration value."""
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    def set_defaults(self, defaults: Dict[str, Any]) -> None:
        """Set default configuration values."""
        self._defaults.update(defaults)
    
    def save(self) -> bool:
        """Save the configuration."""
        return True
    
    def load(self) -> bool:
        """Load the configuration."""
        return True


# Create a singleton instance for testing
config_manager = ConfigManager()
