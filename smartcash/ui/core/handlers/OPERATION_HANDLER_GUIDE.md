# Operation Handler Best Practices

## 🔧 Extending OperationHandler

When creating new operation handlers, follow these patterns to avoid common issues:

### ✅ Correct Implementation

```python
from smartcash.ui.core.handlers.operation_handler import OperationHandler

class MyOperationManager(OperationHandler):
    def __init__(self, config, operation_container=None):
        super().__init__(
            module_name='my_module',
            parent_module='setup',
            operation_container=operation_container
        )
        self.config = config
    
    def get_operations(self) -> Dict[str, Callable]:
        """Required: Implement abstract method."""
        return {
            'execute': self.execute_operation,
            'cancel': self.cancel_operation
        }
    
    async def execute_operation(self) -> Dict[str, Any]:
        """Your operation logic here."""
        # Use self.log() - it automatically handles operation container
        self.log("Starting operation", 'info')
        
        # Use self.update_progress() for progress updates
        self.update_progress(50, "Halfway done", "primary")
        
        return {"success": True}
```

### ❌ Common Mistakes to Avoid

1. **Don't override the log() method unless absolutely necessary**
   - The base class handles operation container vs logger automatically
   - Overriding can cause duplicate logging

2. **Don't use both logger and operation_container**
   ```python
   # BAD - causes duplicate logging
   self.logger.info("Message")
   self.operation_container.log("Message")
   
   # GOOD - use only self.log()
   self.log("Message", 'info')
   ```

3. **Don't create custom operation container wrappers**
   - Pass the operation container directly to the parent constructor
   - The base class handles all the integration

4. **Always implement get_operations()**
   - This is an abstract method and must be implemented
   - Return a dict mapping operation names to callable methods

### 🔄 Migration Guide

If you have existing operation handlers with issues:

1. Remove custom log() method overrides
2. Remove operation container wrapper classes  
3. Pass operation_container directly to parent __init__
4. Use self.log() instead of logger/operation_container calls
5. Implement get_operations() if missing

### 📊 Progress Updates

```python
# Use the built-in progress system
self.update_progress(
    progress=75,           # 0-100
    message="Processing",  # Status message
    level="primary"        # UI level (primary/secondary/tertiary)
)
```

### 🚨 Error Handling

```python
try:
    # Your operation logic
    pass
except Exception as e:
    self.log(f"Operation failed: {e}", 'error')
    return {"success": False, "error": str(e)}
```