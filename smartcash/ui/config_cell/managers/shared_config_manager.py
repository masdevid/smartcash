"""
File: smartcash/ui/config_cell/managers/shared_config_manager.py
Deskripsi: Manager untuk berbagi konfigurasi antar child components dalam parent module yang sama

ARCHITECTURE:
============
SharedConfigManager menyediakan mekanisme untuk:
1. Menyimpan konfigurasi dalam memory yang bisa diakses antar cells
2. Broadcast perubahan konfigurasi ke semua subscribers
3. Auto-sync dengan file YAML
4. Event-driven updates untuk reactive UI

DESIGN PATTERN:
==============
Observer Pattern dimana:
- SharedConfigManager adalah Subject
- Child components adalah Observers
- Config changes trigger notifications ke semua observers

USAGE:
======
```python
# Di child component A
manager = SharedConfigManager.get_instance('dataset')
manager.update_config('split', new_config)  # Broadcast ke semua subscribers

# Di child component B (akan auto-update jika subscribe)
manager.subscribe('split', callback_function)
```
"""

from typing import Dict, Any, Optional, Callable, List, Set
from collections import defaultdict
import threading
import weakref
from datetime import datetime
import logging

from smartcash.common.config.manager import get_config_manager

logger = logging.getLogger(__name__)

class SharedConfigManager:
    """📡 Manager untuk berbagi konfigurasi antar child components.
    
    Singleton per parent module untuk ensure semua child share same instance.
    
    Features:
    - In-memory config storage untuk fast access
    - Event-driven updates (Observer pattern)
    - Auto-sync dengan persistent storage (YAML)
    - Thread-safe operations
    - Weak references untuk prevent memory leaks
    """
    
    # Singleton instances per parent module
    _instances: Dict[str, 'SharedConfigManager'] = {}
    _lock = threading.Lock()
    
    def __init__(self, parent_module: str):
        """Initialize shared config manager untuk parent module.
        
        Args:
            parent_module: Nama parent module (e.g., 'dataset', 'model')
        """
        self.parent_module = parent_module
        self.config_manager = get_config_manager()
        
        # In-memory config storage
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Subscribers: module_name -> list of callbacks
        # Use weakref untuk prevent memory leaks
        self._subscribers: Dict[str, List[weakref.ref]] = defaultdict(list)
        
        # Track last update time untuk each module
        self._last_updated: Dict[str, datetime] = {}
        
        # Lock untuk thread safety
        self._config_lock = threading.Lock()
        
        logger.info(f"📡 SharedConfigManager initialized untuk {parent_module}")
    
    @classmethod
    def get_instance(cls, parent_module: str) -> 'SharedConfigManager':
        """Get singleton instance untuk parent module.
        
        Args:
            parent_module: Nama parent module
            
        Returns:
            SharedConfigManager instance
        """
        with cls._lock:
            if parent_module not in cls._instances:
                cls._instances[parent_module] = cls(parent_module)
            return cls._instances[parent_module]
    
    def update_config(self, module_name: str, config: Dict[str, Any], 
                     persist: bool = True) -> None:
        """Update configuration dan broadcast ke subscribers.
        
        Args:
            module_name: Nama child module (e.g., 'split', 'preprocessing')
            config: New configuration
            persist: Whether to save to YAML file
        """
        with self._config_lock:
            # Update in-memory config
            self._configs[module_name] = config.copy()
            self._last_updated[module_name] = datetime.now()
            
            # Persist to YAML if requested
            if persist:
                try:
                    config_file = f"{module_name}_config.yaml"
                    self.config_manager.save_config(config, config_file)
                    logger.debug(f"💾 Config saved to {config_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to persist config: {e}")
            
            # Broadcast to subscribers
            self._notify_subscribers(module_name, config)
            
            logger.info(f"📡 Config updated for {module_name}")
    
    def get_config(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get current configuration untuk module.
        
        Args:
            module_name: Nama child module
            
        Returns:
            Current configuration atau None
        """
        with self._config_lock:
            # Check in-memory first
            if module_name in self._configs:
                return self._configs[module_name].copy()
            
            # Try load from YAML
            try:
                config_file = f"{module_name}_config.yaml"
                config = self.config_manager.load_config(config_file)
                if config:
                    self._configs[module_name] = config
                    return config.copy()
            except Exception as e:
                logger.debug(f"No saved config for {module_name}: {e}")
            
            return None
    
    def subscribe(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """Subscribe to configuration changes.
        
        Args:
            module_name: Module to subscribe to
            callback: Function called with new config when updated
            
        Returns:
            Unsubscribe function
        """
        # Create weak reference to callback
        weak_callback = weakref.ref(callback)
        
        with self._config_lock:
            self._subscribers[module_name].append(weak_callback)
            logger.debug(f"📡 New subscriber for {module_name}")
        
        # Return unsubscribe function
        def unsubscribe():
            with self._config_lock:
                self._subscribers[module_name] = [
                    ref for ref in self._subscribers[module_name]
                    if ref() is not None and ref() != weak_callback
                ]
                logger.debug(f"📡 Unsubscribed from {module_name}")
        
        return unsubscribe
    
    def _notify_subscribers(self, module_name: str, config: Dict[str, Any]) -> None:
        """Notify all subscribers about config change.
        
        Args:
            module_name: Module that changed
            config: New configuration
        """
        # Get live callbacks and clean dead references
        live_callbacks = []
        dead_refs = []
        
        for ref in self._subscribers[module_name]:
            callback = ref()
            if callback is not None:
                live_callbacks.append(callback)
            else:
                dead_refs.append(ref)
        
        # Clean dead references
        if dead_refs:
            self._subscribers[module_name] = [
                ref for ref in self._subscribers[module_name]
                if ref not in dead_refs
            ]
        
        # Notify live callbacks (outside lock to prevent deadlock)
        for callback in live_callbacks:
            try:
                callback(config.copy())
            except Exception as e:
                logger.error(f"❌ Error in subscriber callback: {e}")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations untuk parent module.
        
        Returns:
            Dictionary of module_name -> config
        """
        with self._config_lock:
            return {
                module: config.copy()
                for module, config in self._configs.items()
            }
    
    def clear_module_config(self, module_name: str) -> None:
        """Clear configuration untuk specific module.
        
        Args:
            module_name: Module to clear
        """
        with self._config_lock:
            if module_name in self._configs:
                del self._configs[module_name]
                del self._last_updated[module_name]
                logger.info(f"🧹 Cleared config for {module_name}")
    
    def get_last_updated(self, module_name: str) -> Optional[datetime]:
        """Get last update time untuk module config.
        
        Args:
            module_name: Module name
            
        Returns:
            Last update datetime atau None
        """
        return self._last_updated.get(module_name)


# Convenience functions
def get_shared_config_manager(parent_module: str) -> SharedConfigManager:
    """Get shared config manager untuk parent module.
    
    Args:
        parent_module: Parent module name (e.g., 'dataset')
        
    Returns:
        SharedConfigManager instance
    """
    return SharedConfigManager.get_instance(parent_module)


def broadcast_config_update(parent_module: str, module_name: str, 
                          config: Dict[str, Any], persist: bool = True) -> None:
    """Broadcast config update ke semua subscribers.
    
    Args:
        parent_module: Parent module name
        module_name: Child module name
        config: New configuration
        persist: Whether to save to YAML
    """
    manager = get_shared_config_manager(parent_module)
    manager.update_config(module_name, config, persist)


def subscribe_to_config(parent_module: str, module_name: str,
                       callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    """Subscribe to config changes untuk module.
    
    Args:
        parent_module: Parent module name
        module_name: Module to subscribe to
        callback: Callback function
        
    Returns:
        Unsubscribe function
    """
    manager = get_shared_config_manager(parent_module)
    return manager.subscribe(module_name, callback)