"""
File: smartcash/ui/core/shared/shared_config_manager.py
Deskripsi: Wrapper untuk SharedConfigManager yang ada dengan additional utilities
specific untuk core module. Menghindari duplikasi dengan reuse existing implementation.
"""

from typing import Dict, Any, Optional, Callable
import threading

from smartcash.ui.core.shared.shared_config_manager import (
    SharedConfigManager,
    get_shared_config_manager as get_base_manager
)


class CoreSharedConfigManager:
    """Extended shared config manager untuk core module.
    
    Wrapper yang menambahkan functionality tanpa duplikasi:
    - 🔄 Config versioning
    - 📊 Config diff tracking
    - 🔐 Config validation rules
    - 📦 Config templates
    """
    
    def __init__(self, parent_module: str):
        """Initialize dengan base shared config manager."""
        self.parent_module = parent_module
        self._base_manager = get_base_manager(parent_module)
        
        # Additional features
        self._config_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._validation_rules: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self._config_templates: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    # === Delegate ke Base Manager ===
    
    def get_config(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get config untuk module."""
        return self._base_manager.get_config(module_name)
    
    def update_config(self, module_name: str, config: Dict[str, Any]) -> None:
        """Update config dengan versioning."""
        with self._lock:
            # Save version history
            if module_name not in self._config_versions:
                self._config_versions[module_name] = []
            
            # Get current config untuk history
            if current := self.get_config(module_name):
                self._config_versions[module_name].append(current.copy())
            
            # Validate jika ada rules
            if module_name in self._validation_rules:
                if not self._validation_rules[module_name](config):
                    raise ValueError(f"Config validation failed for {module_name}")
            
            # Update via base manager
            self._base_manager.update_config(module_name, config)
    
    def subscribe(self, 
                 module_name: str,
                 callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """Subscribe untuk config updates."""
        return self._base_manager.subscribe(module_name, callback)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get semua configs."""
        return self._base_manager.get_all_configs()
    
    # === Additional Features ===
    
    def set_validation_rule(self, 
                          module_name: str,
                          validator: Callable[[Dict[str, Any]], bool]) -> None:
        """Set validation rule untuk module config."""
        with self._lock:
            self._validation_rules[module_name] = validator
    
    def register_template(self, 
                        template_name: str,
                        template_config: Dict[str, Any]) -> None:
        """Register config template."""
        with self._lock:
            self._config_templates[template_name] = template_config.copy()
    
    def apply_template(self, 
                      module_name: str,
                      template_name: str,
                      merge: bool = True) -> None:
        """Apply template ke module config."""
        with self._lock:
            if template_name not in self._config_templates:
                raise ValueError(f"Template not found: {template_name}")
            
            template = self._config_templates[template_name].copy()
            
            if merge and (current := self.get_config(module_name)):
                # Merge dengan existing
                template.update(current)
            
            self.update_config(module_name, template)
    
    def get_config_diff(self, 
                       module_name: str,
                       version_a: int = -2,
                       version_b: int = -1) -> Dict[str, Any]:
        """Get diff antara dua config versions.
        
        Args:
            module_name: Module name
            version_a: Version index (negative = dari belakang)
            version_b: Version index
            
        Returns:
            Dict dengan keys: added, removed, changed
        """
        with self._lock:
            if module_name not in self._config_versions:
                return {'added': {}, 'removed': {}, 'changed': {}}
            
            versions = self._config_versions[module_name]
            if not versions or len(versions) < 2:
                return {'added': {}, 'removed': {}, 'changed': {}}
            
            # Get configs
            config_a = versions[version_a] if version_a < len(versions) else {}
            config_b = versions[version_b] if version_b < len(versions) else self.get_config(module_name) or {}
            
            # Calculate diff
            added = {k: v for k, v in config_b.items() if k not in config_a}
            removed = {k: v for k, v in config_a.items() if k not in config_b}
            changed = {
                k: {'old': config_a[k], 'new': config_b[k]}
                for k in set(config_a) & set(config_b)
                if config_a[k] != config_b[k]
            }
            
            return {
                'added': added,
                'removed': removed,
                'changed': changed
            }
    
    def rollback_config(self, module_name: str, steps: int = 1) -> bool:
        """Rollback config ke version sebelumnya.
        
        Args:
            module_name: Module name
            steps: Jumlah version untuk rollback
            
        Returns:
            True jika berhasil rollback
        """
        with self._lock:
            if module_name not in self._config_versions:
                return False
            
            versions = self._config_versions[module_name]
            if len(versions) < steps:
                return False
            
            # Get target version
            target_config = versions[-(steps)]
            
            # Apply without adding to history
            self._base_manager.update_config(module_name, target_config)
            
            # Remove rolled back versions dari history
            self._config_versions[module_name] = versions[:-steps]
            
            return True
    
    def get_version_count(self, module_name: str) -> int:
        """Get jumlah version history."""
        return len(self._config_versions.get(module_name, []))
    
    def clear_version_history(self, module_name: Optional[str] = None) -> None:
        """Clear version history."""
        with self._lock:
            if module_name:
                if module_name in self._config_versions:
                    self._config_versions[module_name].clear()
            else:
                self._config_versions.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            base_configs = self.get_all_configs()
            
            return {
                'modules': len(base_configs),
                'total_versions': sum(
                    len(versions) 
                    for versions in self._config_versions.values()
                ),
                'validation_rules': len(self._validation_rules),
                'templates': len(self._config_templates),
                'version_details': {
                    module: len(versions)
                    for module, versions in self._config_versions.items()
                }
            }


# === Global Registry ===

_core_managers: Dict[str, CoreSharedConfigManager] = {}
_core_lock = threading.Lock()

def get_core_shared_manager(parent_module: str) -> CoreSharedConfigManager:
    """Get atau create core shared config manager."""
    with _core_lock:
        if parent_module not in _core_managers:
            _core_managers[parent_module] = CoreSharedConfigManager(parent_module)
        
        return _core_managers[parent_module]