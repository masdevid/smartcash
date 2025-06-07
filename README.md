# 🏗️ Guide Pola Backend Module SmartCash

## Struktur Wajib Backend Module

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
        """Setup handlers dengan backend integration"""
        return setup_[module]_handlers(ui_components, config, env)
    
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
    """Factory function untuk [module] UI"""
    return _[module]_initializer.initialize(env=env, config=config, **kwargs)
```

### 2. Config Handler Template

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_handler.py
Deskripsi: Config handler untuk [module] dengan inheritance
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.[domain].[module].handlers.config_extractor import extract_[module]_config
from smartcash.ui.[domain].[module].handlers.config_updater import update_[module]_ui
from smartcash.common.config.manager import get_config_manager

class [Module]ConfigHandler(ConfigHandler):
    """Config handler untuk [module] dengan [domain] integration"""
    
    def __init__(self, module_name: str = '[module]', parent_module: str = '[domain]'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = '[domain]_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        return extract_[module]_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_[module]_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan optimal workers"""
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        return get_default_[module]_config()
```

### 3. Config Extractor Template

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi [module] dari UI components
"""

from typing import Dict, Any
from datetime import datetime

def extract_[module]_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstraksi konfigurasi [module] sesuai dengan [domain]_config.yaml"""
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        '[module]': {
            'enabled': get_value('[module]_enabled', True),
            'option1': get_value('option1_input', 'default_value'),
            'option2': get_value('option2_checkbox', False),
            'workers': get_value('worker_slider', _get_optimal_workers())
        },
        
        'performance': {
            'num_workers': get_value('worker_slider', 8),
            'batch_size': get_value('batch_size', 32),
            'use_gpu': get_value('use_gpu', True)
        }
    }

def _get_optimal_workers() -> int:
    """Get optimal workers untuk [module]"""
    from smartcash.common.threadpools import get_optimal_thread_count
    return get_optimal_thread_count('io')
```

### 4. Config Updater Template

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi [module]
"""

from typing import Dict, Any

def update_[module]_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config [module]"""
    [module]_config = config.get('[module]', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    field_mappings = [
        ('[module]_enabled', [module]_config, 'enabled', True),
        ('option1_input', [module]_config, 'option1', 'default_value'),
        ('option2_checkbox', [module]_config, 'option2', False),
        ('worker_slider', performance_config, 'num_workers', 8),
        ('batch_size', performance_config, 'batch_size', 32),
        ('use_gpu', performance_config, 'use_gpu', True)
    ]
    
    [safe_update(component_key, source_config.get(config_key, default_value)) 
     for component_key, source_config, config_key, default_value in field_mappings]

def reset_[module]_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI components ke default konfigurasi [module]"""
    try:
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        default_config = get_default_[module]_config()
        update_[module]_ui(ui_components, default_config)
    except Exception:
        _apply_basic_defaults(ui_components)

def _apply_basic_defaults(ui_components: Dict[str, Any]) -> None:
    """Apply basic defaults jika config manager tidak tersedia"""
    basic_defaults = {
        '[module]_enabled': True,
        'option1_input': 'default_value',
        'option2_checkbox': False,
        'worker_slider': 8
    }
    
    for key, value in basic_defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            ui_components[key].value = value
```

### 5. Defaults Template

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/defaults.py
Deskripsi: Hardcoded default configuration untuk [module]
"""

from typing import Dict, Any

def get_default_[module]_config() -> Dict[str, Any]:
    """Get hardcoded default configuration untuk [module] reset operations"""
    return {
        'config_version': '1.0',
        '_base_': 'base_config.yaml',
        
        '[module]': {
            'enabled': True,
            'option1': 'default_value',
            'option2': False,
            'workers': 8
        },
        
        'performance': {
            'num_workers': 8,
            'batch_size': 32,
            'use_gpu': True,
            'max_memory_usage_gb': 4.0
        }
    }

# One-liner utilities
get_default_option1 = lambda: 'default_value'
get_default_workers = lambda: 8
is_enabled_by_default = lambda: True
```

## 🔄 Implementasi Checklist

### Setup Phase
- [ ] Create directory structure
- [ ] Implement initializer dengan CommonInitializer inheritance
- [ ] Setup config handler dengan proper inheritance
- [ ] Create defaults file dengan hardcoded values. 

### Config Management
- [ ] Implement config extractor dengan one-liner style
- [ ] Implement config updater dengan field mappings
- [ ] Setup proper config file naming ([domain]_config.yaml)
- [ ] Make sure config structure of loaded [domain]_config.yaml match with base_config.yaml
- [ ] If existing forms are not matching with [domain]_config.yaml, update and regenerate [domain]_config.yaml with missing keys but keep alignment with base_config.yaml. Config Override: base_config.yaml -> [domain]_config.yaml -> forms
- [ ] Add optimal workers integration

### UI Components
- [ ] Create main UI components file
- [ ] Implement input options dengan responsive layout
- [ ] Setup progress tracking dengan dual level
- [ ] Add proper button management

### Backend Integration
- [ ] Create backend utils untuk service integration
- [ ] Setup proper error handling
- [ ] Implement dialog confirmations
- [ ] Add progress callbacks

## 💡 Best Practices

1. **Consistent Naming**: Follow exact naming patterns
2. **DRY Principle**: Use existing utils dan components
3. **Error Safety**: Always include fallback mechanisms
4. **One-liner Style**: Optimize untuk readability
5. **Backend Integration**: Use existing services
6. **State Management**: Proper button/progress handling
7. **Config Consistency**: Follow parent config structure