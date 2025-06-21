# Common Initializer Guide - Complete Implementation

## 📁 Struktur File Module

```
[module]/
├── __init__.py
├── [module]_initializer.py          # Entry point
├── components/
│   ├── __init__.py
│   └── ui_components.py             # UI creation
├── handlers/
│   ├── __init__.py
│   ├── config_handler.py            # Config management
│   ├── [module]_handlers.py         # Operation handlers
│   └── defaults.py                  # Default values
└── utils/
    ├── __init__.py
    └── [module]_utils.py             # Helper functions
```

## 🚀 Initializer Implementation

### 1. Main Initializer Class
```python
# File: [module]_initializer.py
from smartcash.ui.initializers.common_initializer import CommonInitializer
from .handlers.[module]_handlers import setup_[module]_handlers
from .handlers.config_handler import [Module]ConfigHandler

class [Module]Initializer(CommonInitializer):
    def __init__(self):
        super().__init__(
            module_name='[module]',
            config_filename='[module]_config.yaml',
            config_handler_class=[Module]ConfigHandler
        )
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dari config"""
        from .components.ui_components import create_[module]_ui
        return create_[module]_ui(config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk module"""
        return setup_[module]_handlers(ui_components, config)

# Fungsi entry point
def initialize_[module]_ui(config=None, env=None):
    """Entry point untuk module initialization"""
    initializer = [Module]Initializer()
    return initializer.initialize(config=config, env=env)
```

### 2. Advanced Initializer dengan Custom Hooks
```python
class [Module]Initializer(CommonInitializer):
    def __init__(self):
        super().__init__(
            module_name='[module]',
            config_filename='[module]_config.yaml',
            config_handler_class=[Module]ConfigHandler
        )
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """UI creation dengan validation"""
        from .components.ui_components import create_[module]_ui
        ui_components = create_[module]_ui(config)
        
        # Validate required components
        required_widgets = ['main_button', 'status_panel', 'log_output']
        missing = [w for w in required_widgets if w not in ui_components]
        if missing:
            raise ValueError(f"Missing widgets: {missing}")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan environment check"""
        # Environment-specific setup
        if env == 'colab':
            ui_components['drive_enabled'] = True
        
        return setup_[module]_handlers(ui_components, config)
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs):
        """Post-init custom logic"""
        # Auto-load config ke UI
        config_handler = ui_components.get('config_handler')
        if config_handler and config:
            config_handler.update_ui(ui_components, config)
            self._log_to_ui(ui_components, "✅ Config loaded", "success")
        
        # Setup callbacks
        self._setup_auto_save_callbacks(ui_components)
```

## 🏗️ Config Handler Implementation

### Standard Config Handler
```python
# File: handlers/config_handler.py
from smartcash.ui.handlers.config_handlers import ConfigHandler
from .defaults import get_default_[module]_config

class [Module]ConfigHandler(ConfigHandler):
    def __init__(self):
        super().__init__('[module]_name', None)
        self.config_mapping = {
            'setting1': 'setting1_input',
            'setting2': 'setting2_dropdown',
            'enabled': 'enabled_checkbox'
        }
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        defaults = get_default_[module]_config()['[module]_name']
        return {'[module]_name': {
            config_key: getattr(ui_components.get(widget_key), 'value', defaults[config_key])
            for config_key, widget_key in self.config_mapping.items()
        }}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        module_config = config.get('[module]_name', {})
        [setattr(ui_components[widget_key], 'value', module_config.get(config_key))
         for config_key, widget_key in self.config_mapping.items()
         if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]
    
    def get_default_config(self) -> Dict[str, Any]:
        return get_default_[module]_config()
```

## 🎯 Handler Setup Patterns

### Config Handlers (Save/Reset)
```python
def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset dengan UI feedback"""
    
    def save_config(button=None):
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error save: {str(e)}", "error")
    
    def reset_config(button=None):
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error reset: {str(e)}", "error")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)
```

### Operation Handlers dengan Confirmation
```python
def _setup_operation_handlers(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan confirmation"""
    
    def operation_handler(button=None):
        return _handle_operation_with_confirmation(ui_components)
    
    if operation_button := ui_components.get('operation_button'):
        operation_button.on_click(operation_handler)

def _handle_operation_with_confirmation(ui_components: Dict[str, Any]) -> bool:
    """Operation dengan confirmation workflow"""
    try:
        _clear_outputs(ui_components)
        
        if _should_execute_operation(ui_components):
            return _execute_operation_with_progress(ui_components)
        
        if not _is_confirmation_pending(ui_components):
            _show_confirmation_area(ui_components)
            _log_to_ui(ui_components, "⏳ Menunggu konfirmasi...", "info")
            _show_operation_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Error operation: {str(e)}")
        return False
```

## 📊 Progress Tracker Integration

```python
def _execute_operation_with_progress(ui_components: Dict[str, Any]) -> bool:
    """Execute dengan progress tracking"""
    try:
        _disable_buttons(ui_components)
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("🚀 Memulai operasi...")
        else:
            _show_fallback_progress("🚀 Memulai operasi...")
        
        total_steps = 5
        for step in range(total_steps):
            if progress_tracker and hasattr(progress_tracker, 'update'):
                progress_tracker.update(f"📋 Step {step+1}...", step + 1, total_steps)
            else:
                _show_fallback_progress(f"📋 Step {step+1}/{total_steps}")
            
            _perform_operation_step(step, ui_components)
        
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete("✅ Operasi selesai!")
        else:
            _show_fallback_progress("✅ Operasi selesai!")
        
        _log_to_ui(ui_components, "🎉 Operasi berhasil!", "success")
        return True
        
    except Exception as e:
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(f"❌ Error: {str(e)}")
        else:
            _show_fallback_progress(f"❌ Error: {str(e)}")
        
        _handle_error(ui_components, f"💥 Operation failed: {str(e)}")
        return False
    finally:
        _enable_buttons(ui_components)

def _show_fallback_progress(message: str):
    print(f"📊 {message}")
```

## 🔔 UI Logging & Error Handling

```python
def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log ke UI dengan emoji context"""
    
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = message
    
    if log_output := ui_components.get('log_output'):
        with log_output:
            emoji_map = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            print(f"{emoji_map.get(level, 'ℹ️')} {message}")
    else:
        print(f"📝 {message}")

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear semua output widgets"""
    if log_output := ui_components.get('log_output'):
        log_output.clear_output()
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = ""

def _handle_error(ui_components: Dict[str, Any], error_message: str):
    """Comprehensive error handling"""
    _log_to_ui(ui_components, error_message, "error")
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = error_message
    _enable_buttons(ui_components)
    _hide_confirmation_area(ui_components)

def _disable_buttons(ui_components: Dict[str, Any]):
    """Disable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]):
    """Enable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = False
```

## ⚠️ Confirmation Dialog Management

```python
def _show_confirmation_area(ui_components: Dict[str, Any]):
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'block'

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'none'

def _show_operation_confirmation(ui_components: Dict[str, Any]):
    """Show config summary untuk konfirmasi"""
    config_handler = ui_components.get('config_handler')
    if config_handler:
        current_config = config_handler.extract_config(ui_components)
        summary = f"Config: {len(current_config)} settings"
        _log_to_ui(ui_components, f"📋 Konfirmasi: {summary}", "info")

def _should_execute_operation(ui_components: Dict[str, Any]) -> bool:
    confirm_checkbox = ui_components.get('confirm_operation_checkbox')
    return confirm_checkbox and getattr(confirm_checkbox, 'value', False)

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    confirmation_area = ui_components.get('confirmation_area')
    return confirmation_area and confirmation_area.layout.display != 'none'
```

## 📋 Widget Naming Convention

```python
widget_types = {
    'input': 'purpose_input',
    'dropdown': 'purpose_dropdown', 
    'checkbox': 'purpose_checkbox',
    'button': 'purpose_button',
    'output': 'purpose_output',
    'panel': 'purpose_panel',
    'area': 'purpose_area'
}
```