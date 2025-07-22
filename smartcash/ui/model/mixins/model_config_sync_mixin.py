"""
File: smartcash/ui/model/mixins/model_config_sync_mixin.py
Description: Mixin for cross-module configuration synchronization.

Centralizes configuration management patterns to eliminate duplication across
backbone, training, evaluation, and pretrained modules.
"""

import copy
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from smartcash.common.logger import get_logger


class ModelConfigSyncMixin:
    """
    Mixin for cross-module configuration synchronization.
    
    Provides standardized functionality for:
    - Cross-module configuration access and caching
    - Deep configuration merging with validation
    - UI synchronization with selective updates
    - Dependency validation and automatic updates
    - Configuration consistency enforcement
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_sync_logger = get_logger(f"{self.__class__.__name__}.config_sync")
        self._module_config_cache = {}
        self._config_dependencies = {}
    
    def get_module_config(
        self, 
        module_name: str, 
        auto_initialize: bool = False,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration from another model module.
        
        Args:
            module_name: Name of the module ('backbone', 'training', 'evaluation', 'pretrained')
            auto_initialize: Whether to auto-initialize the module if not available
            use_cache: Whether to use cached configuration
            
        Returns:
            Module configuration dictionary or None if not available
        """
        # Check cache first
        if use_cache and module_name in self._module_config_cache:
            self._config_sync_logger.debug(f"📋 Using cached config for {module_name}")
            return self._module_config_cache[module_name]
        
        try:
            # Dynamic module import and config retrieval
            module_config = None
            
            if module_name == 'backbone':
                from smartcash.ui.model.backbone.backbone_uimodule import get_backbone_uimodule
                backbone_module = get_backbone_uimodule(auto_initialize=auto_initialize)
                if backbone_module and hasattr(backbone_module, 'get_current_config'):
                    module_config = backbone_module.get_current_config()
                    
            elif module_name == 'training':
                from smartcash.ui.model.training.training_uimodule import get_training_uimodule
                training_module = get_training_uimodule(auto_initialize=auto_initialize)
                if training_module and hasattr(training_module, 'get_current_config'):
                    module_config = training_module.get_current_config()
                    
            elif module_name == 'evaluation':
                from smartcash.ui.model.evaluation.evaluation_uimodule import get_evaluation_uimodule
                evaluation_module = get_evaluation_uimodule(auto_initialize=auto_initialize)
                if evaluation_module and hasattr(evaluation_module, 'get_current_config'):
                    module_config = evaluation_module.get_current_config()
                    
            elif module_name == 'pretrained':
                from smartcash.ui.model.pretrained.pretrained_uimodule import get_pretrained_uimodule
                pretrained_module = get_pretrained_uimodule(auto_initialize=auto_initialize)
                if pretrained_module and hasattr(pretrained_module, 'get_current_config'):
                    module_config = pretrained_module.get_current_config()
            
            # Cache the result
            if module_config and use_cache:
                self._module_config_cache[module_name] = copy.deepcopy(module_config)
                self._config_sync_logger.debug(f"📋 Cached config for {module_name}")
            
            if module_config:
                self._config_sync_logger.debug(f"✅ Retrieved config from {module_name} module")
            else:
                self._config_sync_logger.warning(f"⚠️ Could not retrieve config from {module_name} module")
            
            return module_config
            
        except Exception as e:
            self._config_sync_logger.warning(f"⚠️ Error getting {module_name} config: {e}")
            return None
    
    def sync_config_to_ui(
        self, 
        config: Dict[str, Any], 
        target_sections: List[str] = None,
        update_method: str = 'merge'
    ) -> bool:
        """
        Synchronize configuration changes back to UI components.
        
        Args:
            config: Configuration dictionary to sync
            target_sections: Specific sections to update (None for all)
            update_method: 'merge', 'replace', or 'selective'
            
        Returns:
            True if sync was successful, False otherwise
        """
        try:
            # Get current UI components if available
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self._config_sync_logger.warning("⚠️ No UI components available for sync")
                return False
            
            widgets = self._ui_components.get('widgets', {})
            
            # Determine which sections to update
            sections_to_update = target_sections or list(config.keys())
            
            updated_count = 0
            for section in sections_to_update:
                if section not in config:
                    continue
                
                section_config = config[section]
                
                # Update section-specific UI components
                if self._update_section_widgets(section, section_config, widgets, update_method):
                    updated_count += 1
            
            # Update any computed fields or derived UI state
            self._update_derived_ui_state(config)
            
            self._config_sync_logger.info(f"📋 Synced {updated_count}/{len(sections_to_update)} config sections to UI")
            return True
            
        except Exception as e:
            self._config_sync_logger.error(f"❌ Error syncing config to UI: {e}")
            return False
    
    def merge_configs_deep(
        self, 
        base_config: Dict[str, Any], 
        override_config: Dict[str, Any],
        merge_strategy: str = 'deep'
    ) -> Dict[str, Any]:
        """
        Deep merge configuration dictionaries with validation.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration to merge in
            merge_strategy: 'deep', 'shallow', or 'replace'
            
        Returns:
            Merged configuration dictionary
        """
        if merge_strategy == 'replace':
            return copy.deepcopy(override_config)
        
        if merge_strategy == 'shallow':
            merged = copy.deepcopy(base_config)
            merged.update(override_config)
            return merged
        
        # Deep merge strategy
        merged = copy.deepcopy(base_config)
        
        def _deep_merge(target: Dict, source: Dict) -> None:
            """Recursively merge source into target."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    _deep_merge(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)
        
        _deep_merge(merged, override_config)
        return merged
    
    def validate_cross_module_dependencies(
        self, 
        required_modules: List[str],
        dependency_rules: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate that required module configurations are available and valid.
        
        Args:
            required_modules: List of module names that must be available
            dependency_rules: Optional rules defining interdependencies
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'missing_modules': [],
            'invalid_dependencies': [],
            'warnings': [],
            'module_status': {}
        }
        
        # Check each required module
        for module_name in required_modules:
            module_config = self.get_module_config(module_name, auto_initialize=False)
            
            if module_config is None:
                validation_result['missing_modules'].append(module_name)
                validation_result['valid'] = False
                validation_result['module_status'][module_name] = 'missing'
            else:
                validation_result['module_status'][module_name] = 'available'
        
        # Check dependency rules if provided
        if dependency_rules:
            for module, dependencies in dependency_rules.items():
                if module in validation_result['module_status']:
                    for dependency in dependencies:
                        if dependency not in validation_result['module_status'] or \
                           validation_result['module_status'][dependency] != 'available':
                            validation_result['invalid_dependencies'].append(f"{module} requires {dependency}")
                            validation_result['valid'] = False
        
        # Add warnings for partial availability
        available_count = sum(1 for status in validation_result['module_status'].values() if status == 'available')
        total_count = len(required_modules)
        
        if 0 < available_count < total_count:
            validation_result['warnings'].append(f"Partial module availability: {available_count}/{total_count}")
        
        self._config_sync_logger.debug(f"📋 Validated dependencies: {available_count}/{total_count} modules available")
        return validation_result
    
    def update_dependent_configs(
        self, 
        source_module: str, 
        updated_config: Dict[str, Any],
        propagation_rules: Dict[str, List[str]] = None
    ) -> None:
        """
        Update dependent module configurations when source config changes.
        
        Args:
            source_module: Name of module that was updated
            updated_config: New configuration from source module
            propagation_rules: Rules defining which changes should propagate where
        """
        try:
            # Clear cache for source module
            if source_module in self._module_config_cache:
                del self._module_config_cache[source_module]
                self._config_sync_logger.debug(f"📋 Cleared cache for {source_module}")
            
            # Default propagation rules if not provided
            if propagation_rules is None:
                propagation_rules = {
                    'backbone': ['training', 'evaluation'],
                    'pretrained': ['backbone', 'training'],
                    'training': ['evaluation'],
                    'evaluation': []
                }
            
            # Get modules that should be updated
            modules_to_update = propagation_rules.get(source_module, [])
            
            if not modules_to_update:
                self._config_sync_logger.debug(f"📋 No dependent modules for {source_module}")
                return
            
            # Clear cache for dependent modules to force refresh
            for module_name in modules_to_update:
                if module_name in self._module_config_cache:
                    del self._module_config_cache[module_name]
                    self._config_sync_logger.debug(f"📋 Cleared cache for dependent module {module_name}")
            
            # Notify dependent modules of the change
            self._notify_dependent_modules(source_module, updated_config, modules_to_update)
            
            self._config_sync_logger.info(f"📋 Updated {len(modules_to_update)} dependent modules for {source_module}")
            
        except Exception as e:
            self._config_sync_logger.error(f"❌ Error updating dependent configs: {e}")
    
    def register_config_dependency(
        self, 
        dependent_module: str, 
        source_modules: List[str],
        update_callback: Callable[[str, Dict[str, Any]], None] = None
    ) -> None:
        """
        Register a configuration dependency relationship.
        
        Args:
            dependent_module: Module that depends on others
            source_modules: Modules that this module depends on
            update_callback: Optional callback for when dependencies change
        """
        if dependent_module not in self._config_dependencies:
            self._config_dependencies[dependent_module] = {}
        
        for source_module in source_modules:
            self._config_dependencies[dependent_module][source_module] = {
                'callback': update_callback,
                'last_sync': None
            }
        
        self._config_sync_logger.debug(f"📋 Registered dependency: {dependent_module} -> {source_modules}")
    
    def get_synchronized_config(
        self, 
        module_configs: Dict[str, Dict[str, Any]],
        sync_rules: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Create a synchronized configuration from multiple module configs.
        
        Args:
            module_configs: Dictionary mapping module names to their configs
            sync_rules: Rules for resolving conflicts between modules
            
        Returns:
            Synchronized configuration dictionary
        """
        synchronized_config = {}
        
        # Default sync rules
        if sync_rules is None:
            sync_rules = {
                'paths': 'backbone',  # Use backbone module for path configurations
                'architecture': 'backbone',  # Use backbone for architecture settings
                'training': 'training',  # Use training module for training settings
                'evaluation': 'evaluation'  # Use evaluation module for evaluation settings
            }
        
        # Apply sync rules
        for config_section, source_module in sync_rules.items():
            if source_module in module_configs:
                source_config = module_configs[source_module]
                if config_section in source_config:
                    synchronized_config[config_section] = copy.deepcopy(source_config[config_section])
        
        # Add any remaining configuration sections
        for module_name, module_config in module_configs.items():
            for section_name, section_config in module_config.items():
                if section_name not in synchronized_config:
                    synchronized_config[section_name] = copy.deepcopy(section_config)
        
        self._config_sync_logger.debug(f"📋 Created synchronized config from {len(module_configs)} modules")
        return synchronized_config
    
    def _update_section_widgets(
        self, 
        section: str, 
        section_config: Dict[str, Any], 
        widgets: Dict[str, Any],
        update_method: str
    ) -> bool:
        """Update UI widgets for a specific configuration section."""
        try:
            # This method should be overridden by implementing classes
            # to provide section-specific UI update logic
            
            # Default implementation: try to find and update common widget types
            updated = False
            
            for key, value in section_config.items():
                widget_key = f"{section}_{key}"
                
                if widget_key in widgets:
                    widget = widgets[widget_key]
                    
                    # Update different widget types
                    if hasattr(widget, 'value'):
                        widget.value = value
                        updated = True
                    elif hasattr(widget, 'selected'):
                        widget.selected = value
                        updated = True
                    elif hasattr(widget, 'text'):
                        widget.text = str(value)
                        updated = True
            
            return updated
            
        except Exception as e:
            self._config_sync_logger.debug(f"⚠️ Error updating section {section}: {e}")
            return False
    
    def _update_derived_ui_state(self, config: Dict[str, Any]) -> None:
        """Update computed UI fields based on configuration changes."""
        # This method should be overridden by implementing classes
        # to provide module-specific derived state updates
        pass
    
    def _notify_dependent_modules(
        self, 
        source_module: str, 
        updated_config: Dict[str, Any], 
        dependent_modules: List[str]
    ) -> None:
        """Notify dependent modules of configuration changes."""
        for module_name in dependent_modules:
            try:
                # Try to call refresh method on dependent module if available
                module_config = self.get_module_config(module_name, auto_initialize=False, use_cache=False)
                
                # This is a basic notification - specific modules can override
                # this behavior to implement custom update logic
                self._config_sync_logger.debug(f"📋 Notified {module_name} of {source_module} config change")
                
            except Exception as e:
                self._config_sync_logger.debug(f"⚠️ Error notifying {module_name}: {e}")