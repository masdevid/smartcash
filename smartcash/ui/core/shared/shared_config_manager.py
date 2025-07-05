"""
File: smartcash/ui/core/shared/shared_config_manager.py
Deskripsi: Manager untuk berbagi konfigurasi antar child components dalam parent module yang sama

ARCHITECTURE:
============
SharedConfigManager menyediakan mekanisme untuk:
1. Menyimpan konfigurasi dalam memory yang bisa diakses antar cells
2. Broadcast perubahan konfigurasi ke semua subscribers
3. Auto-sync dengan file YAML
4. Event-driven updates untuk reactive UI
5. Versioning dan history tracking
6. Config validation
7. Config templates

DESIGN PATTERN:
==============
Observer Pattern dimana:
- SharedConfigManager adalah Subject
- Child components adalah Observers
- Config changes trigger notifications ke semua observers

ADDITIONAL FEATURES:
===================
- 🔄 Config versioning
- 📊 Config diff tracking
- 🔐 Config validation rules
- 📦 Config templates

USAGE:
======
```python
# Basic usage
manager = SharedConfigManager.get_instance('dataset')
manager.update_config('split', new_config)  # Broadcast ke semua subscribers

# Subscribe to changes
def on_config_updated(config):
    print(f"Config updated: {config}")
unsubscribe = manager.subscribe('split', on_config_updated)

# Advanced features
manager.set_validation_rule('split', lambda x: 'required_field' in x)  # Add validation
manager.register_template('default_split', {'train_ratio': 0.7, 'test_ratio': 0.3})  # Register template
manager.apply_template('split', 'default_split')  # Apply template

# Versioning and diff
versions = manager.get_version_count('split')
diff = manager.get_config_diff('split', -2, -1)  # Get diff between versions
manager.rollback_config('split', 1)  # Rollback 1 version
```
"""

import os
import re
import yaml
import copy
import time
import json
import uuid
import logging
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Callable, Tuple, TypeVar, Union, Set, Generic, Type
)
from pathlib import Path
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from uuid import UUID

from typing import (
    Any, Dict, List, Optional, Callable, TypeVar, Union, Tuple, Type, cast, NamedTuple
)
from collections import defaultdict
import threading
import weakref
from datetime import datetime
import logging
from enum import Enum

from smartcash.common.config.manager import get_config_manager

T = TypeVar('T')

class ConfigDiff(NamedTuple):
    """Represents the difference between two config versions."""
    added: Dict[str, Any]
    removed: Dict[str, Any]
    changed: Dict[str, Dict[str, Any]]
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes in the diff."""
        return bool(self.added or self.removed or self.changed)

class ConfigError(Exception):
    """Raised when there's an error with config operations."""
    pass

class ConfigVersioningError(ConfigError):
    """Raised when there's an error with config versioning."""
    pass

logger = logging.getLogger(__name__)

class SharedConfigManager:
    """📡 Manager untuk berbagi konfigurasi antar child components.
    
    Singleton per parent module untuk ensure semua child share same instance.
    
    Features Utama:
    - In-memory config storage untuk fast access
    - Event-driven updates (Observer pattern)
    - Auto-sync dengan persistent storage (YAML)
    - Thread-safe operations
    - Weak references untuk prevent memory leaks
    
    🔄 Versioning & History:
    - Setiap perubahan config disimpan dalam version history
    - Rollback ke versi sebelumnya dengan mudah
    - Dapatkan perbedaan antara dua versi config
    
    🔐 Validation:
    - Validasi config dengan custom rules
    - Validasi otomatis saat update config
    - Pesan error yang jelas saat validasi gagal
    
    📦 Templates:
    - Daftarkan template config yang sering digunakan
    - Terapkan template ke config yang ada
    - Gabungkan template dengan config yang sudah ada
    
    📊 Monitoring & Debugging:
    - Dapatkan statistik penggunaan
    - Lacak perubahan config dari waktu ke waktu
    - Logging yang informatif untuk debugging
    
    Contoh Penggunaan:
    ```python
    # Initialize
    manager = SharedConfigManager.get_instance('my_module')
    
    # Set validation rule
    manager.set_validation_rule('config1', lambda x: 'required_field' in x)
    
    # Register template
    manager.register_template('default', {'setting1': 'default_value'})
    
    # Apply template
    manager.apply_template('config1', 'default')
    
    # Get config diff
    diff = manager.get_config_diff('config1', -2, -1)
    
    # Rollback if needed
    if diff.changed:
        manager.rollback_config('config1', 1)
    ```
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
        
        # Versioning
        self._config_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._validation_rules: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self._config_templates: Dict[str, Dict[str, Any]] = {}
        
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
                     persist: bool = False, skip_versioning: bool = False) -> None:
        """Update configuration for a module.
        
        Args:
            module_name: Name of the module
            config: New configuration
            persist: If True, save to disk
            skip_versioning: If True, don't create a new version entry
            
        Raises:
            ValueError: If config validation fails
            RuntimeError: If unable to acquire lock
        """
        logger.debug(f"🔍 [update_config] Entering update_config for module: {module_name}")
        logger.debug(f"   - persist: {persist}, skip_versioning: {skip_versioning}")
        logger.debug(f"   - config keys: {list(config.keys()) if config else 'None'}")
        logger.debug(f"   - Thread: {threading.current_thread().name} (ID: {threading.get_ident()})")
        logger.debug(f"   - Lock state: {'🔒 LOCKED' if self._config_lock.locked() else '🔓 UNLOCKED'}")
        logger.debug(f"   - Lock owner: {getattr(self._config_lock, '_owner', 'None')}")
        
        # Log before acquiring lock to check for deadlocks
        logger.debug(f"🔄 [update_config] Attempting to acquire _config_lock for {module_name}")
        
        # Use a flag to track if we need to release the lock
        lock_acquired = False
        try:
            # Try to acquire the lock with a timeout
            lock_acquired = self._config_lock.acquire(timeout=5)
            if not lock_acquired:
                error_msg = f"❌ [update_config] Failed to acquire _config_lock for {module_name} after 5 seconds"
                logger.error(error_msg)
                logger.error(f"   - Current lock owner: {getattr(self._config_lock, '_owner', 'Unknown')}")
                raise RuntimeError(error_msg)
                
            logger.debug(f"✅ [update_config] Successfully acquired _config_lock for {module_name}")
            logger.debug(f"   - Lock owner: {threading.get_ident()}")
            
            # Validate if there are rules
            logger.debug(f"🔍 [update_config] Checking validation rules for {module_name}")
            logger.debug(f"   - Available validation rules: {list(self._validation_rules.keys())}")
            
            if module_name in self._validation_rules:
                logger.debug(f"🔍 Found validation rule for {module_name}")
                validator = self._validation_rules[module_name]
                logger.debug(f"🔍 Validator type: {type(validator).__name__}")
                logger.debug(f"🔍 Running validator: {validator}")
                
                try:
                    # Log before calling validator
                    logger.debug("🔍 Calling validator function...")
                    validation_result = validator(config)
                    logger.debug(f"✅ Validator completed with result: {validation_result}")
                    
                    if not validation_result:
                        error_msg = f"❌ Config validation failed for {module_name}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                except Exception as e:
                    logger.error(f"❌ Error during validation: {str(e)}")
                    logger.exception("Validation error details:")
                    raise
            
            # Save version history if enabled
            logger.debug("📝 Processing version history...")
            if not skip_versioning:
                logger.debug(f"   - Versioning enabled for {module_name}")
                if module_name not in self._config_versions:
                    logger.debug(f"   - Initializing version history for {module_name}")
                    self._config_versions[module_name] = []
                
                # Get current config for history
                current = self.get_config(module_name)
                if current:
                    logger.debug(f"   - Adding current config to version history")
                    self._config_versions[module_name].append(current.copy())
                    logger.debug(f"   - Version history length: {len(self._config_versions[module_name])}")
            
            # Update in-memory config
            logger.debug(f"💾 Updating in-memory config for {module_name}")
            self._configs[module_name] = config.copy()
            self._last_updated[module_name] = datetime.now()
            
            # Persist to YAML if requested
            if persist:
                try:
                    config_file = f"{module_name}_config.yaml"
                    logger.debug(f"💾 Attempting to save config to {config_file}")
                    self.config_manager.save_config(config, config_file)
                    logger.debug(f"✅ Config saved to {config_file}")
                except Exception as e:
                    error_msg = f"❌ Failed to persist config: {e}"
                    logger.error(error_msg)
                    logger.exception("Persist error details:")
                    raise ConfigError(f"Failed to persist config: {e}") from e
            
            # Broadcast to subscribers
            logger.debug(f"📢 Preparing to notify subscribers for {module_name}")
            logger.debug(f"   - Subscriber count: {len(self._subscribers.get(module_name, []))}")
            
            # Make a copy of the config before notifying subscribers
            config_copy = config.copy()
            
            # Release the lock before notifying subscribers to prevent deadlocks
            logger.debug("🔓 Releasing lock before notifying subscribers")
            if lock_acquired:
                self._config_lock.release()
                lock_acquired = False
            
            # Notify subscribers outside the lock to prevent deadlocks
            try:
                self._notify_subscribers(module_name, config_copy)
                logger.info(f"✅ Successfully updated config for {module_name}")
            except Exception as e:
                logger.error(f"❌ Error notifying subscribers: {e}")
                logger.exception("Subscriber notification error:")
                raise
        finally:
            # Ensure the lock is always released
            if lock_acquired and hasattr(self, '_config_lock') and self._config_lock.locked():
                logger.debug("🔓 Releasing _config_lock in finally block")
                try:
                    self._config_lock.release()
                except Exception as release_error:
                    logger.error(f"❌ Error releasing _config_lock: {release_error}")
                    logger.exception("Lock release error:")
    
    def set_validation_rule(self, module_name: str, 
                          validator: Callable[[Dict[str, Any]], bool]) -> None:
        """Set validation rule untuk module config.
        
        Args:
            module_name: Nama module
            validator: Function yang menerima config dict dan return boolean
        """
        logger.debug(f"🔍 [set_validation_rule] Setting validation rule for {module_name}")
        logger.debug(f"   - Validator: {validator}")
        
        with self._config_lock:
            logger.debug("🔒 Acquired _config_lock in set_validation_rule")
            self._validation_rules[module_name] = validator
            logger.debug(f"🔐 Set validation rule for {module_name}")
            logger.debug(f"   - Current validation rules: {list(self._validation_rules.keys())}")
            
            # Debug: Check if the validator is callable
            if not callable(validator):
                logger.error(f"❌ Validator for {module_name} is not callable: {validator}")
            else:
                logger.debug("✅ Validator is callable")
    
    def register_template(self, template_name: str,
                        template_config: Dict[str, Any]) -> None:
        """Register config template.
        
        Args:
            template_name: Nama template
            template_config: Konfigurasi template
        """
        with self._config_lock:
            self._config_templates[template_name] = template_config.copy()
            logger.debug(f"📦 Registered template: {template_name}")
    
    def apply_template(self, module_name: str,
                     template_name: str,
                     merge: bool = True) -> None:
        """Apply template ke module config.
        
        Args:
            module_name: Nama module target
            template_name: Nama template yang akan diaplikasikan
            merge: Jika True, gabungkan dengan config yang sudah ada
            
        Raises:
            ValueError: Jika template tidak ditemukan
        """
        with self._config_lock:
            if template_name not in self._config_templates:
                raise ValueError(f"Template not found: {template_name}")
            
            template = self._config_templates[template_name].copy()
            
            if merge and (current := self.get_config(module_name)):
                # Merge dengan existing
                template.update(current)
            
            self.update_config(module_name, template, skip_versioning=True)
            logger.info(f"🔄 Applied template '{template_name}' to {module_name}")
    
    def get_config_diff(self, 
                       module_name: str,
                       version_a: int = -2,
                       version_b: int = -1) -> ConfigDiff:
        """Get diff antara dua config versions.
        
        Args:
            module_name: Module name
            version_a: Version index (negative = dari belakang)
            version_b: Version index
            
        Returns:
            ConfigDiff dengan added, removed, dan changed items
            
        Raises:
            ConfigVersioningError: Jika versioning tidak diaktifkan atau versi tidak valid
        """
        with self._config_lock:
            if module_name not in self._config_versions:
                raise ConfigVersioningError(f"No version history for {module_name}")
                
            versions = self._config_versions[module_name]
            if not versions or len(versions) < 2:
                return ConfigDiff(added={}, removed={}, changed={})
            
            try:
                # Handle negative indices
                if version_a < 0:
                    version_a = len(versions) + version_a
                if version_b < 0:
                    version_b = len(versions) + version_b
                
                # Get configs
                config_a = versions[version_a] if 0 <= version_a < len(versions) else {}
                config_b = versions[version_b] if 0 <= version_b < len(versions) else self.get_config(module_name) or {}
                
                # Calculate diff
                added = {k: v for k, v in config_b.items() if k not in config_a}
                removed = {k: v for k, v in config_a.items() if k not in config_b}
                changed = {
                    k: {'old': config_a[k], 'new': config_b[k]}
                    for k in set(config_a) & set(config_b)
                    if config_a[k] != config_b[k]
                }
                
                return ConfigDiff(added=added, removed=removed, changed=changed)
                
            except IndexError as e:
                raise ConfigVersioningError(f"Invalid version index: {e}") from e
    
    def rollback_config(self, module_name: str, steps: int = 1) -> bool:
        """Rollback config ke version sebelumnya.
        
        Args:
            module_name: Module name
            steps: Jumlah version untuk rollback
            
        Returns:
            True jika berhasil rollback, False jika tidak ada cukup history
        """
        with self._config_lock:
            if module_name not in self._config_versions:
                logger.warning(f"No version history for {module_name}")
                return False
                
            versions = self._config_versions[module_name]
            if len(versions) < steps:
                logger.warning(f"Not enough versions to rollback {steps} steps")
                return False
            
            try:
                # Get target version
                target_config = versions[-(steps + 1)]
                
                # Apply without adding to history
                self._configs[module_name] = target_config.copy()
                
                # Remove rolled back versions dari history
                self._config_versions[module_name] = versions[:-steps]
                
                # Update last modified time
                self._last_updated[module_name] = datetime.now()
                
                # Notify subscribers
                self._notify_subscribers(module_name, target_config)
                
                logger.info(f"⏪ Rolled back {module_name} config {steps} version(s)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to rollback config: {e}")
                return False
    
    def get_version_count(self, module_name: str) -> int:
        """Get jumlah version history untuk module.
        
        Args:
            module_name: Nama module
            
        Returns:
            Jumlah versi yang tersimpan, 0 jika tidak ada
        """
        with self._config_lock:
            return len(self._config_versions.get(module_name, []))
    
    def clear_version_history(self, module_name: Optional[str] = None) -> None:
        """Clear version history.
        
        Args:
            module_name: Jika disediakan, hapus history untuk module tersebut saja.
                       Jika None, hapus semua history.
        """
        with self._config_lock:
            if module_name is not None:
                if module_name in self._config_versions:
                    self._config_versions[module_name].clear()
                    logger.debug(f"🧹 Cleared version history for {module_name}")
            else:
                self._config_versions.clear()
                logger.debug("🧹 Cleared all version history")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.
        
        Returns:
            Dict berisi statistik penggunaan
        """
        with self._config_lock:
            base_configs = self.get_all_configs()
            
            return {
                'modules': len(base_configs),
                'total_versions': sum(
                    len(versions) 
                    for versions in self._config_versions.values()
                ),
                'validation_rules': len(self._validation_rules),
                'templates': len(self._config_templates),
                'subscribers': sum(len(callbacks) for callbacks in self._subscribers.values()),
                'last_updated': {
                    mod: ts.isoformat() 
                    for mod, ts in self._last_updated.items()
                } if self._last_updated else {}
            }
    
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
        
        This method MUST be called WITHOUT holding any locks to prevent deadlocks.
        
        Args:
            module_name: Module that changed
            config: New configuration (should be a copy of the config)
        """
        logger.debug(f"\n🔔 [notify_subscribers] Starting notification for {module_name}")
        logger.debug(f"   - Thread: {threading.current_thread().name} (ID: {threading.get_ident()})")
        logger.debug(f"   - Lock state: {'🔒 LOCKED' if hasattr(self, '_config_lock') and self._config_lock.locked() else '🔓 UNLOCKED'}")
        if hasattr(self, '_config_lock') and self._config_lock.locked():
            logger.warning("⚠️  WARNING: _notify_subscribers called while holding _config_lock! This could cause deadlocks!")
            logger.warning(f"   - Lock owner: {getattr(self._config_lock, '_owner', 'Unknown')}")
        
        start_time = time.time()
        
        # Get a snapshot of current subscribers
        subscribers = []
        try:
            logger.debug(f"🔍 [notify_subscribers] Getting subscribers for {module_name}")
            logger.debug(f"   - Subscribers dict keys: {list(self._subscribers.keys())}")
            
            # Make a thread-safe copy of the subscribers list
            logger.debug("   - Attempting to acquire _config_lock for subscribers copy")
            lock_acquired = False
            try:
                lock_acquired = self._config_lock.acquire(timeout=2)  # 2 second timeout
                if not lock_acquired:
                    logger.error("❌ [notify_subscribers] Failed to acquire _config_lock after 2 seconds!")
                    logger.error(f"   - Current lock owner: {getattr(self._config_lock, '_owner', 'Unknown')}")
                    return
                    
                logger.debug(f"   - Successfully acquired _config_lock (held by: {threading.get_ident()})")
                subscribers = list(self._subscribers.get(module_name, []))
                logger.debug(f"   - Found {len(subscribers)} subscribers for {module_name}")
                
                # Clean up any dead references while we have the lock
                if subscribers:
                    live_refs = []
                    dead_refs = []
                    
                    for ref in subscribers:
                        if ref() is not None:
                            live_refs.append(ref)
                        else:
                            dead_refs.append(ref)
                    
                    # Update the subscribers list if we found dead references
                    if dead_refs:
                        logger.debug(f"   - Cleaning up {len(dead_refs)} dead references")
                        self._subscribers[module_name] = live_refs
                    
                    subscribers = live_refs
            except Exception as e:
                logger.error(f"❌ Unexpected error in update_config: {e}")
                logger.exception("Update config error:")
                raise
            finally:
                # Always release the lock in finally block to prevent deadlocks
                if lock_acquired and self._config_lock.locked():
                    logger.debug("🔓 Releasing _config_lock in finally block")
                    try:
                        self._config_lock.release()
                    except Exception as release_error:
                        logger.error(f"❌ Error releasing _config_lock: {release_error}")
                        logger.exception("Lock release error:")
        
        except Exception as e:
            logger.error(f"❌ Error getting subscribers: {e}")
            logger.exception("Subscriber retrieval error:")
            return
            
        # Notify all live callbacks (outside the lock to prevent deadlocks)
        elapsed = time.time() - start_time
        logger.debug(f"⏱️  [notify_subscribers] Prepared subscribers in {elapsed:.4f}s")
        
        if not subscribers:
            logger.debug("ℹ️  No subscribers to notify")
            return
            
        logger.debug(f"📤 [notify_subscribers] Notifying {len(subscribers)} subscribers")
        logger.debug(f"   - First few subscribers: {[str(s())[:50] + '...' for s in subscribers[:3]]}")
            
        logger.debug(f"   - Notifying {len(subscribers)} live callbacks")
        
        # Make a deep copy of the config to prevent race conditions
        try:
            config_copy = copy.deepcopy(config)
        except Exception as e:
            logger.error(f"❌ Failed to copy config: {e}")
            config_copy = config.copy()  # Fallback to shallow copy
        
        # Notify each subscriber with timeout protection
        for i, ref in enumerate(subscribers, 1):
            try:
                callback = ref()
                if callback is None:
                    logger.debug(f"   - Subscriber {i}: Reference is dead, skipping")
                    continue
                    
                callback_name = getattr(callback, "__name__", str(callback))
                logger.debug(f"\n📤 [notify_subscribers] Notifying subscriber {i}/{len(subscribers)}: {callback_name}")
                
                # Add a timeout to prevent hanging on a single subscriber
                def call_with_timeout():
                    try:
                        callback(config_copy)
                        return True, None
                    except Exception as e:
                        return False, e
                
                # Use ThreadPoolExecutor with timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_with_timeout)
                    try:
                        success, error = future.result(timeout=5)  # 5 second timeout per callback
                        if success:
                            logger.debug(f"✅ Successfully notified callback: {callback_name}")
                        else:
                            logger.error(f"❌ Error in subscriber callback {callback_name}: {error}")
                            logger.debug(f"   - Error details: {error}", exc_info=True)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"⏱️  Callback {callback_name} timed out after 5 seconds")
                        # Attempt to cancel the future (though it may continue running in the background)
                        future.cancel()
                
            except Exception as e:
                logger.error(f"❌ Unexpected error notifying subscriber {i}: {e}")
                logger.debug(f"   - Error details: {str(e)}", exc_info=True)
                
            # Log progress periodically
            if i % 5 == 0 or i == len(subscribers):
                logger.debug(f"   - Progress: {i}/{len(subscribers)} subscribers notified")
        
        total_time = time.time() - start_time
        logger.debug(f"✅ [notify_subscribers] Completed notifications in {total_time:.4f}s")
    
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