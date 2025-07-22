# Troubleshooting Guide: UI Components (Progress Tracker, Dialog, Log Accordion)

## 🚨 Critical Issues & Solutions

### 1. Progress Tracker Issues

#### **Issue 1.1: `'ProgressTracker' object has no attribute 'update_progress'`**

**🔍 Symptoms:**
```python
AttributeError: 'ProgressTracker' object has no attribute 'update_progress'
```

**🎯 Root Cause:**
- `ProgressTracker` class uses `set_progress()` method
- `OperationContainer` was calling non-existent `update_progress()` method
- Method name mismatch between container and tracker

**✅ Solution:**
```python
# ❌ WRONG - in operation_container.py
self.progress_tracker.update_progress(value, message, level)

# ✅ CORRECT - in operation_container.py  
self.progress_tracker.set_progress(value, level, message)
```

**📝 Code Fix:**
```python
# File: smartcash/ui/components/operation_container.py
def _update_progress_bars(self) -> None:
    for level, data in self.progress_bars.items():
        if data['visible']:
            if data.get('error', False):
                self.progress_tracker.error(data['message'])  # Fixed method
            else:
                self.progress_tracker.set_progress(  # Fixed method
                    data['value'], 
                    level,
                    data['message']
                )
```

**⚠️ Gotcha:**
- Always check `ProgressTracker` class for actual method names
- Parameter order is different: `set_progress(value, level, message)` not `update_progress(value, message, level)`

---

#### **Issue 1.2: Progress Update Function Not Available**

**🔍 Symptoms:**
```python
update_progress = container.get('update_progress')
# Returns None or non-callable
```

**🎯 Root Cause:**
- Export conditions in `create_operation_container` were too restrictive
- Function not properly exposed in return dictionary

**✅ Solution:**
```python
# File: smartcash/ui/components/operation_container.py
return {
    'container': container.container,
    'progress_tracker': container.progress_tracker if show_progress else None,
    'update_progress': container.update_progress,  # Always export this
    # ... other exports
}
```

**⚠️ Gotcha:**
- Don't make `update_progress` export conditional on `show_progress`
- The method should handle disabled state internally

---

### 2. Dialog Functionality Issues

#### **Issue 2.1: Dialog Function Returns Boolean Instead of Method**

**🔍 Symptoms:**
```python
show_dialog = container.get('show_dialog')
print(show_dialog)  # Prints: True (should be: <bound method>)
callable(show_dialog)  # Returns: False (should be: True)
```

**🎯 Root Cause:**
- **CRITICAL NAMING CONFLICT**: Boolean parameter `show_dialog` overwrites method `show_dialog`
- `self.show_dialog = show_dialog` in `__init__` destroys the method

**✅ Solution:**
```python
# ❌ WRONG - in OperationContainer.__init__
def __init__(self, show_dialog: bool = True):
    self.show_dialog = show_dialog  # This overwrites the method!

# ✅ CORRECT - in OperationContainer.__init__  
def __init__(self, show_dialog: bool = True):
    self._show_dialog_enabled = show_dialog  # Use different attribute name
```

**📝 Complete Fix:**
```python
# File: smartcash/ui/components/operation_container.py
class OperationContainer(BaseUIComponent):
    def __init__(self, show_dialog: bool = True, **kwargs):
        # Store configuration with different name to avoid method conflict
        self._show_dialog_enabled = show_dialog  # ✅ RENAMED
        
        # Initialize dialog area if enabled
        if self._show_dialog_enabled:  # ✅ Use renamed attribute
            self._init_dialog_area()
    
    def show_dialog(self, title: str, message: str, **kwargs):  # ✅ Method preserved
        if not hasattr(self, 'dialog_area') or not self.dialog_area:
            if kwargs.get('on_confirm'):
                kwargs['on_confirm']()  # Graceful fallback
            return
        # ... dialog implementation
```

**⚠️ Critical Gotcha:**
- **NEVER** use the same name for a boolean parameter and a method
- Parameter: `_show_dialog_enabled` (attribute)
- Method: `show_dialog()` (method)
- This is a common Python anti-pattern that destroys methods

---

#### **Issue 2.2: Dialog Export Conditions Wrong**

**🔍 Symptoms:**
```python
# Container creation
container = create_operation_container(show_dialog=True)
show_dialog = container.get('show_dialog')  # Returns None
```

**🎯 Root Cause:**
- Export condition logic was evaluating method as boolean
- Conditional export was too complex

**✅ Solution:**
```python
# File: smartcash/ui/components/operation_container.py
return {
    'show_dialog': container.show_dialog,  # Always export the method
    # Don't use: container.show_dialog if show_dialog else None
}
```

**⚠️ Gotcha:**
- The method itself should handle disabled state
- Don't make method export conditional - let the method decide behavior

---

#### **Issue 2.3: Dialog Implementation Missing Parameters**

**🔍 Symptoms:**
```python
ConfirmationDialog._show_dialog() missing 1 required positional argument: 'html_content'
```

**🎯 Root Cause:**
- Deep implementation issue in dialog component chain
- Legacy vs new dialog API mismatch

**✅ Workaround:**
```python
def show_dialog(self, title, message, on_confirm=None, **kwargs):
    if not hasattr(self, 'dialog_area') or not self.dialog_area:
        # Graceful fallback when dialog UI not available
        if on_confirm:
            on_confirm()  # Just execute callback
        return
    
    try:
        # Attempt dialog display
        show_confirmation_dialog(...)
    except Exception as e:
        # Log error but don't fail operation
        self.logger.warning(f"Dialog display failed: {e}")
        if on_confirm:
            on_confirm()  # Execute callback anyway
```

**⚠️ Gotcha:**
- Always provide fallback behavior for missing dialog UI
- Don't let dialog implementation errors break main functionality

---

### 3. Log Accordion Issues

#### **Issue 3.1: Color Variable Undefined in HTML Template**

**🔍 Symptoms:**
```python
NameError: name 'color' is not defined
# In log entry HTML generation
```

**🎯 Root Cause:**
- F-string template used undefined `{color}` variable
- Should use `{style['color']}` directly

**✅ Solution:**
```python
# ❌ WRONG - in log_accordion.py
html = f"""
<div class='log-entry' style='border-left-color: {color} !important;'>
    <!-- content -->
</div>
""".format(color=style['color'], ...)  # Mixed f-string and .format()

# ✅ CORRECT - in log_accordion.py
html = f"""
<div class='log-entry' style='border-left-color: {style['color']} !important;'>
    <!-- content -->
</div>
"""  # Pure f-string, no .format() needed
```

**⚠️ Gotcha:**
- Don't mix f-string syntax with `.format()` method
- Use f-string variables directly: `{style['color']}`

---

#### **Issue 3.2: Logger Attribute Name Mismatch**

**🔍 Symptoms:**
```python
AttributeError: 'LogAccordion' object has no attribute '_logger'
```

**🎯 Root Cause:**
- `LogAccordion` inherits from `BaseUIComponent`
- Base class provides `self.logger`, not `self._logger`

**✅ Solution:**
```python
# ❌ WRONG - in log_accordion.py
except Exception as e:
    self._logger.error(f"Error creating log widget: {str(e)}")

# ✅ CORRECT - in log_accordion.py  
except Exception as e:
    self.logger.error(f"Error creating log widget: {str(e)}")  # Use inherited logger
```

**⚠️ Gotcha:**
- Check base class for correct logger attribute name
- `BaseUIComponent` provides `self.logger` (public)

---

#### **Issue 3.3: Log Messages Not Appearing**

**🔍 Symptoms:**
- Log function calls don't error
- No messages appear in log accordion
- Log accordion UI exists but empty

**🎯 Root Cause:**
- Log function export not properly connected
- Method chaining broken between container and accordion

**✅ Solution:**
```python
# File: smartcash/ui/components/operation_container.py
return {
    'log_message': container.log_message,  # Ensure proper method export
    'log_accordion': container.log_accordion if show_logs else None,
}

# Ensure container.log_message properly delegates to accordion
def log_message(self, message: str, level: str = "info") -> None:
    if self.log_accordion:
        self.log_accordion.log(message, LogLevel(level))
```

**⚠️ Gotcha:**
- Check the entire method call chain: `container.log_message` → `log_accordion.log`
- Verify log level conversion between string and enum

---

## 🛠️ General Troubleshooting Patterns

### Pattern 1: Method vs Attribute Naming Conflicts

**🚨 CRITICAL RULE:** Never use the same name for:
- Boolean/config parameters in `__init__`
- Method names in the same class

```python
# ❌ DANGEROUS PATTERN
class Component:
    def __init__(self, show_feature: bool):
        self.show_feature = show_feature  # Overwrites method!
    
    def show_feature(self):  # Method gets destroyed
        pass

# ✅ SAFE PATTERN  
class Component:
    def __init__(self, show_feature: bool):
        self._show_feature_enabled = show_feature  # Different name
    
    def show_feature(self):  # Method preserved
        if not self._show_feature_enabled:
            return
        # ... implementation
```

### Pattern 2: Export Function Conditions

**🚨 RULE:** Always export methods, let methods handle disabled state

```python
# ❌ WRONG - Conditional export
return {
    'method': container.method if enabled else None
}

# ✅ CORRECT - Always export, method handles state
return {
    'method': container.method  # Method checks self._enabled internally
}
```

### Pattern 3: Graceful Fallbacks

**🚨 RULE:** Always provide fallback behavior for disabled/missing features

```python
def feature_method(self, **kwargs):
    if not self._feature_enabled:
        # Graceful fallback
        callback = kwargs.get('on_complete')
        if callback:
            callback()
        return
    
    try:
        # Main implementation
        self._do_feature()
    except Exception as e:
        # Error fallback
        self.logger.warning(f"Feature failed: {e}")
        callback = kwargs.get('on_complete')
        if callback:
            callback()
```

## 🔍 Debugging Commands

### Check Component Structure
```python
# Debug component attributes
component = SomeComponent()
print([attr for attr in dir(component) if not attr.startswith('_')])

# Check for method conflicts
print(f"Attribute type: {type(getattr(component, 'method_name'))}")
print(f"Is callable: {callable(getattr(component, 'method_name'))}")
```

### Test Export Structure
```python
# Debug container exports
container = create_operation_container()
for key, value in container.items():
    print(f"{key}: {type(value)} - Callable: {callable(value)}")
```

### Verify Method Chains
```python
# Debug method delegation
container = create_operation_container(show_logs=True)
log_accordion = container.get('log_accordion')
log_message = container.get('log_message')

print(f"Log accordion: {log_accordion}")
print(f"Log message method: {log_message}")
print(f"Accordion has log method: {hasattr(log_accordion, 'log')}")
```

## 📋 Prevention Checklist

### Before Adding New UI Components:

- [ ] **Naming:** No conflicts between parameters and methods
- [ ] **Exports:** Methods always exported, handle disabled state internally  
- [ ] **Fallbacks:** Graceful behavior when features disabled/missing
- [ ] **Method Chains:** Verify delegation from container to child components
- [ ] **Error Handling:** Try/catch with meaningful fallbacks
- [ ] **Testing:** Test both enabled and disabled states
- [ ] **Documentation:** Document gotchas and common pitfalls

### Code Review Red Flags:

- [ ] `self.method_name = parameter` (overwrites method)
- [ ] `return method if enabled else None` (conditional export)
- [ ] Missing fallback behavior in disabled state
- [ ] F-string + .format() mixing
- [ ] Assuming methods exist without checking
- [ ] No error handling in UI operations

---

*This troubleshooting guide is based on real issues encountered during SmartCash UI development. Keep this updated as new patterns emerge.*