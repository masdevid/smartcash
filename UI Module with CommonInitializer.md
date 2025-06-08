# 🏗️ Guide Pola UI Module SmartCash

## Struktur Wajib UI Module with CommonInitializer

### 📁 Directory Structure
```
smartcash/ui/[domain]/[module]/
├── __init__.py                    # Export initializer
├── [module]_initializer.py        # Main initializer class
├── handlers/
│   ├── __init__.py
│   ├── config_handler.py         # Config management
│   ├── config_extractor.py       # UI → Config
│   ├── config_updater.py         # Config → UI
│   ├── defaults.py               # Hardcoded defaults
│   └── [module]_handlers.py      # Business logic handlers
├── components/
│   ├── __init__.py
│   ├── ui_components.py          # Main UI assembly
│   └── input_options.py          # Form components
└── utils/
    ├── __init__.py
    ├── ui_utils.py               # UI display utilities
    ├── button_manager.py         # State management
    ├── dialog_utils.py           # Confirmation dialogs
    ├── progress_utils.py         # Progress tracking
    └── backend_utils.py          # Backend integration
```

## 📋 Konsistensi Penamaan File

### File Naming Pattern
- **Initializer**: `[module]_initializer.py`
- **Config Handler**: `config_handler.py` (standardized)
- **Main Handlers**: `[module]_handlers.py`
- **Main UI**: `ui_components.py` (standardized)
- **Utils**: `[function]_utils.py`

### Class Naming Pattern
- **Initializer**: `[Module]Initializer`
- **Config Handler**: `[Module]ConfigHandler`
- **UI Components**: Functional exports

## 🔧 Template Standar

### 1. Initializer Template

```python
"""
File: smartcash/ui/[domain]/[module]/[module]_initializer.py
Deskripsi: [Module] initializer yang mewarisi CommonInitializer
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.[domain].[module].handlers.config_handler import [Module]ConfigHandler
from smartcash.ui.[domain].[module].components.ui_components import create_[module]_main_ui
from smartcash.ui.[domain].[module].handlers.[module]_handlers import setup_[module]_handlers

class [Module]Initializer(CommonInitializer):
    """[Module] initializer dengan complete UI dan backend integration"""
    
    def __init__(self):
        super().__init__(
            module_name='[module]',
            config_handler_class=[Module]ConfigHandler,
            parent_module='[domain]'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create [module] UI components"""
        ui_components = create_[module]_main_ui(config)
        ui_components.update({
            '[module]_initialized': True,
            'module_name': '[module]',
            'data_dir': config.get('data', {}).get('dir', 'data')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan auto config load dan UI update"""
        # Setup handlers terlebih dahulu
        result = setup_[module]_handlers(ui_components, config, env)
        
        # CRITICAL: Load config dari file dan update UI
        self._load_and_update_ui(ui_components)
        
        return result
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """CRITICAL: Load config dari file dan update UI saat initialization"""
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                # Set UI components untuk logging
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                
                # Load config dari file dengan inheritance
                loaded_config = config_handler.load_config()
                
                # Update UI dengan loaded config
                config_handler.update_ui(ui_components, loaded_config)
                
                # Update config reference
                ui_components['config'] = loaded_config
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Error loading config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        return get_default_[module]_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', '[primary]_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]

# Global instance
_[module]_initializer = [Module]Initializer()

def initialize_[module]_ui(env=None, config=None, **kwargs):
    """Factory function untuk [module] UI dengan auto config load"""
    return _[module]_initializer.initialize(env=env, config=config, **kwargs)
```

### 2. Config Handler Template (CRITICAL PATTERNS)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_handler.py
Deskripsi: Config handler dengan proper logging dan auto UI refresh
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.[domain].[module].handlers.config_extractor import extract_[module]_config
from smartcash.ui.[domain].[module].handlers.config_updater import update_[module]_ui
from smartcash.common.config.manager import get_config_manager

class [Module]ConfigHandler(ConfigHandler):
    """Config handler dengan proper UI logging dan inheritance"""
    
    def __init__(self, module_name: str = '[module]', parent_module: str = '[domain]'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = '[module]_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dengan DRY approach"""
        return extract_[module]_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_[module]_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dari defaults.py"""
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        return get_default_[module]_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """CRITICAL: Load dengan inheritance handling"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_to_ui("⚠️ Config kosong, menggunakan default", "warning")
                return self.get_default_config()
            
            # CRITICAL: Handle inheritance dari _base_
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                self._log_to_ui(f"📂 Config loaded dari {filename} dengan inheritance", "info")
                return merged_config
            
            self._log_to_ui(f"📂 Config loaded dari {filename}", "info")
            return config
            
        except Exception as e:
            self._log_to_ui(f"❌ Error loading config: {str(e)}", "error")
            return self.get_default_config()
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """CRITICAL: Merge base config dengan override"""
        import copy
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key == '_base_':
                continue
                
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dictionaries"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """CRITICAL: Save dengan auto refresh"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_to_ui(f"✅ Config tersimpan ke {filename}", "success")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                self._log_to_ui(f"❌ Gagal simpan config ke {filename}", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error save config: {str(e)}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """CRITICAL: Reset dengan auto refresh"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui(f"🔄 Config direset ke default", "success")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self._log_to_ui(f"❌ Gagal reset config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error reset config: {str(e)}", "error")
            return False
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str):
        """CRITICAL: Auto refresh UI setelah save"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self._log_to_ui("🔄 UI direfresh dengan config tersimpan", "info")
        except Exception as e:
            self._log_to_ui(f"⚠️ Error refresh UI: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """CRITICAL: Log ke UI components dengan fallback"""
        try:
            ui_components = getattr(self, '_ui_components', {})
            logger = ui_components.get('logger')
            
            if logger and hasattr(logger, level):
                log_method = getattr(logger, level)
                log_method(message)
                return
            
            # Fallback ke log_to_accordion
            from smartcash.ui.[domain].[module].utils.ui_utils import log_to_accordion
            log_to_accordion(ui_components, message, level)
                
        except Exception:
            print(f"[{level.upper()}] {message}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """CRITICAL: Set UI components untuk logging"""
        self._ui_components = ui_components
```

### 3. Config Extractor Template (DRY APPROACH)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_extractor.py
Deskripsi: DRY config extraction dengan defaults sebagai base
"""

from typing import Dict, Any

def extract_[module]_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """CRITICAL: DRY approach - base dari defaults + form values"""
    from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
    
    # Base structure dari defaults (DRY)
    config = get_default_[module]_config()
    
    # Helper untuk get form values
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Update HANYA nilai dari form - examples:
    config['[module]']['option1'] = get_value('option1_input', 'default')
    config['[module]']['option2'] = get_value('option2_checkbox', False)
    config['performance']['num_workers'] = get_value('worker_slider', 8)
    
    return config
```

### 4. Config Updater Template (INHERITANCE AWARE)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_updater.py
Deskripsi: Config updater dengan inheritance handling
"""

from typing import Dict, Any

def update_[module]_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """CRITICAL: Update UI dengan inheritance handling"""
    # Extract sections dengan safe defaults (handle inheritance)
    [module]_config = config.get('[module]', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Field mappings dengan validation
    safe_update('option1_input', [module]_config.get('option1', 'default'))
    safe_update('option2_checkbox', [module]_config.get('option2', False))
    safe_update('worker_slider', min(max(performance_config.get('num_workers', 8), 1), 10))

def reset_[module]_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke defaults"""
    try:
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        default_config = get_default_[module]_config()
        update_[module]_ui(ui_components, default_config)
    except Exception:
        _apply_hardcoded_defaults(ui_components)

def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Hardcoded defaults fallback"""
    defaults = {'option1_input': 'default', 'worker_slider': 8}
    for key, value in defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass
```

### 5. Handlers Template (CONFIG HANDLER INTEGRATION)

```python
def setup_[module]_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan config handler UI integration"""
    
    # CRITICAL: Setup config handler dengan UI logger
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup other handlers...
    setup_config_handlers_fixed(ui_components, config)
    
    return ui_components

def setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """CRITICAL: Config handlers dengan proper UI logging"""
    
    def save_config(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            # CRITICAL: Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.save_config(ui_components)
            # Logger sudah handle di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error save: {str(e)}")
    
    def reset_config(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            # CRITICAL: Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.reset_config(ui_components)
            # Logger sudah handle di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset: {str(e)}")
    
    # Bind handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config)
    if reset_button:
        reset_button.on_click(reset_config)
```

## 🔄 Implementasi Checklist

### Setup Phase
- [ ] Create directory structure
- [ ] Implement initializer dengan CommonInitializer inheritance
- [ ] **CRITICAL**: Add `_load_and_update_ui()` di initializer
- [ ] Setup config handler dengan proper inheritance

### Config Management (CRITICAL FIXES)
- [ ] **CRITICAL**: Implement `_log_to_ui()` dengan fallback ke `log_to_accordion`
- [ ] **CRITICAL**: Add `set_ui_components()` method di config handler
- [ ] **CRITICAL**: Implement `load_config()` dengan `_merge_configs()` untuk inheritance
- [ ] **CRITICAL**: Add `_refresh_ui_after_save()` untuk auto UI update
- [ ] Implement DRY config extractor dengan defaults sebagai base
- [ ] Implement config updater dengan safe extraction dari inheritance

### Handler Integration (CRITICAL)
- [ ] **CRITICAL**: Call `config_handler.set_ui_components()` di setup handlers
- [ ] **CRITICAL**: Use `setup_config_handlers_fixed()` pattern
- [ ] Setup proper error handling dengan UI logging

### UI Components
- [ ] Create main UI components file
- [ ] Implement input options dengan responsive layout
- [ ] Setup progress tracking dengan dual level
- [ ] Add proper button management

## 💡 Critical Lessons Learned

### 1. **Config Loading & UI Update**
- **MUST** implement `_load_and_update_ui()` di initializer
- **MUST** handle inheritance dengan `_merge_configs()`
- **MUST** call `config_handler.set_ui_components()` untuk logging

### 2. **Logging to UI**
- **MUST** implement `_log_to_ui()` dengan fallback ke `log_to_accordion`
- **NEVER** rely only pada `print()` - logs won't show di UI

### 3. **DRY Principle**
- **ALWAYS** use defaults.py sebagai base structure
- **ONLY** update form values di extractor
- **AVOID** rewriting entire config structure

### 4. **Auto Refresh**
- **MUST** implement `_refresh_ui_after_save()` 
- **MUST** reload config dari file setelah save
- **MUST** call `update_ui()` dengan reloaded config

### 5. **Error Prevention**
- **ALWAYS** use safe_update dengan try/catch
- **ALWAYS** validate dropdown values before assignment
- **ALWAYS** handle missing keys dengan `.get()` dan defaults