"""
File: smartcash/common/utils/one_liner_fixes.py
Deskripsi: Perbaikan untuk one-liner yang broken di berbagai modul
"""

from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

def safe_method_chain(obj, method_name: str, *args, **kwargs):
    """Safe method chaining yang tidak return None pada boolean operations"""
    if obj is None or not hasattr(obj, method_name):
        return obj
    
    method = getattr(obj, method_name)
    if callable(method):
        result = method(*args, **kwargs)
        return obj if result is None else result
    return obj

def safe_setattr_chain(obj, attr: str, value, return_obj: bool = True):
    """Safe setattr yang return object untuk chaining"""
    if obj is not None and hasattr(obj, attr):
        setattr(obj, attr, value)
    return obj if return_obj else value

def safe_mkdir_chain(path, parents: bool = True, exist_ok: bool = True):
    """Safe mkdir yang return path untuk chaining"""
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=parents, exist_ok=exist_ok)
        return path_obj
    except Exception:
        return Path(path)

def safe_operation_or_none(operation: Callable, *args, **kwargs):
    """Execute operation safely, return result or None"""
    try:
        return operation(*args, **kwargs)
    except Exception:
        return None

def safe_conditional_call(condition, operation: Callable, *args, **kwargs):
    """Execute operation only if condition is True"""
    return operation(*args, **kwargs) if condition else None

def safe_list_operation(items: List, operation: Callable, *args, **kwargs):
    """Apply operation to list items safely"""
    results = []
    for item in items:
        try:
            result = operation(item, *args, **kwargs)
            if result is not None:
                results.append(result)
        except Exception:
            continue
    return results

def safe_dict_update(target_dict: Dict, updates: Dict) -> Dict:
    """Safe dictionary update that returns the dict"""
    if target_dict is not None and updates is not None:
        target_dict.update(updates)
    return target_dict

def safe_widget_operation(widget, operation_name: str, *args, **kwargs):
    """Safe widget operation untuk UI components"""
    if widget is None or not hasattr(widget, operation_name):
        return widget
    
    operation = getattr(widget, operation_name)
    if callable(operation):
        try:
            result = operation(*args, **kwargs)
            return widget if result is None else result
        except Exception:
            return widget
    return widget

def safe_boolean_and_operation(*conditions) -> bool:
    """Safe boolean AND operation untuk multiple conditions"""
    return all(bool(condition) for condition in conditions if condition is not None)

def safe_boolean_or_operation(*conditions) -> bool:
    """Safe boolean OR operation untuk multiple conditions"""
    return any(bool(condition) for condition in conditions if condition is not None)

# Fixed one-liner patterns
def fix_broken_one_liner_pattern_1(obj, method: str, *args, **kwargs):
    """Fix pattern: obj.method() and obj -> broken jika method return None"""
    if obj is None:
        return None
    
    if hasattr(obj, method):
        getattr(obj, method)(*args, **kwargs)
    return obj

def fix_broken_one_liner_pattern_2(condition, true_action: Callable, false_action: Callable = None):
    """Fix pattern: condition and action() or fallback"""
    if condition:
        return true_action() if callable(true_action) else true_action
    elif false_action:
        return false_action() if callable(false_action) else false_action
    return None

def fix_broken_one_liner_pattern_3(items: List, action: Callable, condition: Callable = None):
    """Fix pattern: [action(item) for item in items if condition]"""
    if not items:
        return []
    
    results = []
    for item in items:
        try:
            if condition is None or condition(item):
                result = action(item)
                if result is not None:
                    results.append(result)
        except Exception:
            continue
    return results

# Specific fixes untuk common patterns
def fix_ui_component_setup(ui_components: Dict[str, Any], component_name: str, setup_func: Callable):
    """Fix UI component setup pattern"""
    if component_name in ui_components and ui_components[component_name] is not None:
        try:
            setup_func(ui_components[component_name])
        except Exception:
            pass
    return ui_components

def fix_path_operation(path_str: str, operation: str, *args, **kwargs):
    """Fix path operation pattern"""
    try:
        path = Path(path_str)
        if operation == 'mkdir':
            path.mkdir(parents=kwargs.get('parents', True), exist_ok=kwargs.get('exist_ok', True))
        elif operation == 'exists':
            return path.exists()
        elif operation == 'is_file':
            return path.is_file()
        elif operation == 'is_dir':
            return path.is_dir()
        return path
    except Exception:
        return Path(path_str) if path_str else None

def fix_config_operation(config: Dict[str, Any], key: str, default_value=None, operation: str = 'get'):
    """Fix config operation pattern"""
    if config is None:
        return default_value
    
    if operation == 'get':
        return config.get(key, default_value)
    elif operation == 'set':
        config[key] = default_value
        return config
    elif operation == 'update':
        if isinstance(default_value, dict):
            config.update(default_value)
        return config
    return config

def fix_logger_operation(logger, level: str, message: str):
    """Fix logger operation pattern"""
    if logger is None:
        return None
    
    try:
        method = getattr(logger, level.lower(), None)
        if callable(method):
            method(message)
    except Exception:
        pass
    return logger

# Pattern replacement helpers
def replace_broken_and_pattern(obj, method_name: str, *args, **kwargs):
    """Replace broken 'obj.method() and obj' patterns"""
    if obj is not None and hasattr(obj, method_name):
        method = getattr(obj, method_name)
        if callable(method):
            method(*args, **kwargs)
    return obj

def replace_broken_conditional_pattern(condition, action, *args, **kwargs):
    """Replace broken conditional patterns"""
    if condition:
        if callable(action):
            return action(*args, **kwargs)
        return action
    return None

def replace_broken_list_comprehension(items, action, condition=None):
    """Replace broken list comprehension patterns"""
    if not items:
        return []
    
    return [
        result for item in items
        if (condition is None or condition(item))
        for result in [action(item) if callable(action) else action]
        if result is not None
    ]

# Widget-specific fixes
def fix_widget_layout_operation(widget, **layout_params):
    """Fix widget layout operation yang sering broken"""
    if widget is None or not hasattr(widget, 'layout'):
        return widget
    
    try:
        for param, value in layout_params.items():
            if hasattr(widget.layout, param):
                setattr(widget.layout, param, value)
    except Exception:
        pass
    return widget

def fix_widget_on_click_operation(widget, handler):
    """Fix widget on_click operation"""
    if widget is not None and hasattr(widget, 'on_click') and callable(handler):
        try:
            widget.on_click(handler)
        except Exception:
            pass
    return widget

def fix_widget_value_operation(widget, value, get_mode: bool = False):
    """Fix widget value operation"""
    if widget is None or not hasattr(widget, 'value'):
        return None if get_mode else widget
    
    try:
        if get_mode:
            return getattr(widget, 'value', None)
        else:
            setattr(widget, 'value', value)
    except Exception:
        pass
    return widget

# Directory and file operation fixes
def fix_directory_operation(dir_path: str, operation: str, **kwargs):
    """Fix directory operations"""
    try:
        path = Path(dir_path)
        
        if operation == 'create':
            path.mkdir(parents=kwargs.get('parents', True), exist_ok=kwargs.get('exist_ok', True))
            return path
        elif operation == 'exists':
            return path.exists()
        elif operation == 'remove':
            if path.exists():
                import shutil
                shutil.rmtree(path, ignore_errors=kwargs.get('ignore_errors', True))
            return True
        elif operation == 'list':
            return list(path.iterdir()) if path.exists() else []
        
        return path
    except Exception:
        return Path(dir_path) if dir_path else None

# Context manager fixes
class SafeOperationContext:
    """Safe operation context untuk prevent broken one-liners"""
    
    def __init__(self, operation: Callable, *args, **kwargs):
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def __enter__(self):
        try:
            self.result = self.operation(*self.args, **self.kwargs)
            return self.result
        except Exception:
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress exceptions

# Utility functions untuk specific fixes
def fix_method_chain_result(obj, result):
    """Fix method chain result yang return None"""
    return obj if result is None else result

def fix_boolean_operation_result(operation_result, fallback=False):
    """Fix boolean operation result"""
    return bool(operation_result) if operation_result is not None else fallback

def fix_callback_operation(callback: Optional[Callable], *args, **kwargs):
    """Fix callback operation"""
    if callback is not None and callable(callback):
        try:
            return callback(*args, **kwargs)
        except Exception:
            pass
    return None

# Collection operation fixes
def fix_list_operation(items: List, operation: str, *args, **kwargs):
    """Fix list operations"""
    if not items:
        return [] if operation in ['filter', 'map'] else 0
    
    try:
        if operation == 'filter':
            condition = args[0] if args and callable(args[0]) else lambda x: True
            return [item for item in items if condition(item)]
        elif operation == 'map':
            mapper = args[0] if args and callable(args[0]) else lambda x: x
            return [mapper(item) for item in items if item is not None]
        elif operation == 'count':
            condition = args[0] if args and callable(args[0]) else lambda x: True
            return sum(1 for item in items if condition(item))
        elif operation == 'find':
            condition = args[0] if args and callable(args[0]) else lambda x: True
            return next((item for item in items if condition(item)), None)
    except Exception:
        pass
    
    return items

def fix_dict_operation(data: Dict, operation: str, key=None, value=None, default=None):
    """Fix dictionary operations"""
    if data is None:
        return {} if operation == 'create' else None
    
    try:
        if operation == 'get':
            return data.get(key, default)
        elif operation == 'set':
            data[key] = value
            return data
        elif operation == 'update':
            if isinstance(value, dict):
                data.update(value)
            return data
        elif operation == 'pop':
            return data.pop(key, default)
        elif operation == 'keys':
            return list(data.keys())
        elif operation == 'values':
            return list(data.values())
    except Exception:
        pass
    
    return data