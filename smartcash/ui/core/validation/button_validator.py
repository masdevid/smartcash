"""
Button-Handler Validation System

This module provides core-level validation to ensure button IDs 
and handlers are properly synchronized across all UI modules.
"""

import re
from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"      # Critical issues that prevent functionality
    WARNING = "warning"  # Issues that may cause problems
    INFO = "info"       # Suggestions for improvement


@dataclass
class ValidationIssue:
    """Represents a button validation issue."""
    level: ValidationLevel
    message: str
    button_id: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ButtonValidationResult:
    """Result of button-handler validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_handlers: List[str] = field(default_factory=list)
    orphaned_handlers: List[str] = field(default_factory=list)
    button_ids: Set[str] = field(default_factory=set)
    handler_ids: Set[str] = field(default_factory=set)
    auto_fixes_applied: List[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)
    
    def add_issue(self, level: ValidationLevel, message: str, 
                  button_id: Optional[str] = None, suggestion: Optional[str] = None,
                  auto_fixable: bool = False) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            level=level,
            message=message,
            button_id=button_id,
            suggestion=suggestion,
            auto_fixable=auto_fixable
        ))


class ButtonHandlerValidator:
    """
    Core-level validation for button-handler synchronization.
    
    This validator ensures that:
    1. All buttons have corresponding handlers
    2. All handlers have corresponding buttons  
    3. Button IDs follow naming conventions
    4. No reserved button IDs are used incorrectly
    """
    
    # Reserved button IDs that have special handling
    RESERVED_IDS = {'save', 'reset', 'primary', 'save_reset'}
    
    # Standard button ID naming pattern (snake_case)
    BUTTON_ID_PATTERN = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$')
    
    # Common button action patterns
    COMMON_ACTIONS = {
        'start', 'stop', 'pause', 'resume', 'cancel',
        'install', 'uninstall', 'update', 'upgrade',
        'check', 'validate', 'verify', 'test',
        'load', 'save', 'export', 'import',
        'build', 'compile', 'deploy', 'run',
        'create', 'delete', 'edit', 'view',
        'refresh', 'reload', 'sync', 'reset'
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
    
    def validate_module(self, ui_module) -> ButtonValidationResult:
        """
        Validate button-handler synchronization for a UI module.
        
        Args:
            ui_module: UI module instance to validate
            
        Returns:
            ButtonValidationResult with validation details
        """
        result = ButtonValidationResult(is_valid=True)
        
        # Extract button IDs and handler IDs
        button_ids = self._extract_button_ids(ui_module)
        handler_ids = self._extract_handler_ids(ui_module)
        
        result.button_ids = button_ids
        result.handler_ids = handler_ids
        
        # Find missing and orphaned handlers with button naming variant support
        result.missing_handlers = self._find_missing_handlers(button_ids, handler_ids)
        result.orphaned_handlers = self._find_orphaned_handlers(button_ids, handler_ids)
        
        # Validate button ID naming conventions
        self._validate_button_naming(button_ids, result)
        
        # Validate handler-button synchronization
        self._validate_handler_sync(result)
        
        # Validate reserved ID usage
        self._validate_reserved_ids(button_ids, handler_ids, result)
        
        # Check for auto-fixable issues
        self._identify_auto_fixes(ui_module, result)
        
        # Set overall validation status
        result.is_valid = not result.has_errors
        
        return result
    
    def _find_missing_handlers(self, button_ids: Set[str], handler_ids: Set[str]) -> List[str]:
        """Find buttons that don't have corresponding handlers, accounting for naming variants."""
        missing = []
        
        for button_id in button_ids:
            if button_id in self.RESERVED_IDS:
                continue
                
            # Check for exact match first
            if button_id in handler_ids:
                continue
                
            # Check for button variant (e.g., 'setup' -> 'setup_button')
            button_variant = f"{button_id}_button"
            if button_variant in handler_ids:
                continue
                
            # Check for btn variant (e.g., 'setup' -> 'btn_setup')  
            btn_variant = f"btn_{button_id}"
            if btn_variant in handler_ids:
                continue
                
            # If none of the variants exist, it's missing
            missing.append(button_id)
            
        return missing
    
    def _find_orphaned_handlers(self, button_ids: Set[str], handler_ids: Set[str]) -> List[str]:
        """Find handlers that don't have corresponding buttons, accounting for naming variants."""
        orphaned = []
        
        for handler_id in handler_ids:
            if handler_id in self.RESERVED_IDS:
                continue
                
            # Check for exact match first
            if handler_id in button_ids:
                continue
                
            # Check if this handler is a variant of a button
            # e.g., handler 'setup_button' for button 'setup'
            if handler_id.endswith('_button'):
                base_name = handler_id[:-7]  # Remove '_button'
                if base_name in button_ids:
                    continue
                    
            # Check if this handler is a btn variant
            # e.g., handler 'btn_setup' for button 'setup'
            if handler_id.startswith('btn_'):
                base_name = handler_id[4:]  # Remove 'btn_'
                if base_name in button_ids:
                    continue
                    
            # If no matching button found, it's orphaned
            orphaned.append(handler_id)
            
        return orphaned
    
    def auto_fix_issues(self, ui_module, validation_result: ButtonValidationResult) -> ButtonValidationResult:
        """
        Automatically fix common button-handler issues.
        
        Args:
            ui_module: UI module instance
            validation_result: Previous validation result
            
        Returns:
            Updated validation result after fixes
        """
        fixes_applied = []
        
        # Auto-register missing handlers for common patterns
        for button_id in validation_result.missing_handlers:
            if self._can_auto_create_handler(ui_module, button_id):
                handler = self._create_default_handler(ui_module, button_id)
                if handler:
                    ui_module.register_button_handler(button_id, handler)
                    fixes_applied.append(f"Auto-registered handler for '{button_id}'")
        
        # Log fixes applied
        if fixes_applied:
            validation_result.auto_fixes_applied = fixes_applied
            if hasattr(ui_module, 'logger'):
                for fix in fixes_applied:
                    ui_module.logger.info(f"🔧 Auto-fix applied: {fix}")
        
        # Re-validate after fixes
        return self.validate_module(ui_module)
    
    def suggest_handler_name(self, button_id: str) -> List[str]:
        """
        Suggest handler method names for a button ID.
        
        Args:
            button_id: Button identifier
            
        Returns:
            List of suggested handler method names
        """
        suggestions = []
        
        # Standard pattern: _handle_{button_id}
        suggestions.append(f"_handle_{button_id}")
        
        # Alternative patterns
        suggestions.append(f"{button_id}_operation")
        suggestions.append(f"_{button_id}_handler")
        suggestions.append(f"on_{button_id}_clicked")
        
        return suggestions
    
    def _extract_button_ids(self, ui_module) -> Set[str]:
        """Extract button IDs from UI module components."""
        button_ids = set()
        
        try:
            if not hasattr(ui_module, '_ui_components') or not ui_module._ui_components:
                return button_ids
            
            # Get action container first as it's the main source of buttons
            action_container = ui_module._ui_components.get('action_container')
            
            # If we have an action container, get buttons from it
            if action_container:
                # Get buttons from action container's buttons dictionary if available
                if hasattr(action_container, 'buttons') and isinstance(action_container.buttons, dict):
                    for btn_id, btn in action_container.buttons.items():
                        if hasattr(btn, 'on_click'):
                            # Only add base button IDs, ignore any variants
                            base_btn_id = btn_id.replace('_button', '').replace('btn_', '')
                            button_ids.add(base_btn_id)
                
                # Get buttons from container children if available
                if hasattr(action_container, 'container') and hasattr(action_container.container, 'children'):
                    for child in action_container.container.children:
                        if hasattr(child, 'on_click'):
                            # Try to get button ID from _button_id attribute or description
                            btn_id = getattr(child, '_button_id', None)
                            if not btn_id and hasattr(child, 'description'):
                                btn_id = child.description.lower().replace(' ', '_')
                            if btn_id:
                                # Only add base button IDs, ignore any variants
                                base_btn_id = btn_id.replace('_button', '').replace('btn_', '')
                                button_ids.add(base_btn_id)
            
            # Also check for buttons directly in ui_components
            for key, widget in ui_module._ui_components.items():
                # Skip non-button widgets and the action container
                if key == 'action_container' or not hasattr(widget, 'on_click'):
                    continue
                
                # Get button ID from _button_id attribute or key
                button_id = getattr(widget, '_button_id', key)
                if button_id:  # Avoid duplicates
                    # Only add base button IDs, ignore any variants
                    base_btn_id = button_id.replace('_button', '').replace('btn_', '')
                    if base_btn_id not in button_ids:
                        button_ids.add(base_btn_id)
            
            # Log extracted button IDs for debugging
            if hasattr(ui_module, 'logger'):
                ui_module.logger.debug(f"Extracted button IDs: {sorted(button_ids)}")
            
        except Exception as e:
            # Log error but continue validation
            if hasattr(ui_module, 'logger'):
                ui_module.logger.error(f"Failed to extract button IDs: {e}", exc_info=True)
        
        return button_ids
    
    def _extract_handler_ids(self, ui_module) -> Set[str]:
        """Extract registered handler IDs from UI module."""
        handler_ids = set()
        
        try:
            # Get handlers from _button_handlers if available
            if hasattr(ui_module, '_button_handlers') and ui_module._button_handlers:
                # Only add exact matches, no variants
                handler_ids.update(ui_module._button_handlers.keys())
            
            # Also check for handlers registered directly on the module
            if hasattr(ui_module, 'handlers') and callable(getattr(ui_module, 'handlers', None)):
                handler_func = getattr(ui_module, 'handlers')
                if hasattr(handler_func, 'handlers'):
                    # Only add exact matches, no variants
                    handler_ids.update(handler_func.handlers.keys())
                    
            # Log extracted handler IDs for debugging
            if hasattr(ui_module, 'logger') and handler_ids:
                ui_module.logger.debug(f"Extracted handler IDs: {sorted(handler_ids)}")
                    
        except Exception as e:
            if hasattr(ui_module, 'logger'):
                ui_module.logger.error(f"Failed to extract handler IDs: {e}", exc_info=True)
        
        return handler_ids
    
    def _validate_button_naming(self, button_ids: Set[str], result: ButtonValidationResult) -> None:
        """Validate button ID naming conventions."""
        for button_id in button_ids:
            if button_id in self.RESERVED_IDS:
                continue  # Skip reserved IDs
            
            if not self.BUTTON_ID_PATTERN.match(button_id):
                result.add_issue(
                    ValidationLevel.WARNING,
                    f"Button ID '{button_id}' doesn't follow snake_case convention",
                    button_id=button_id,
                    suggestion="Use snake_case format (e.g., 'start_training', 'check_status')"
                )
    
    def _validate_handler_sync(self, result: ButtonValidationResult) -> None:
        """Validate handler-button synchronization.
        
        This checks that:
        1. All buttons have corresponding handlers
        2. All handlers have corresponding buttons
        """
        # Track buttons that have been handled to avoid duplicate warnings
        handled_buttons = set()
        
        # Check for missing handlers
        for button_id in result.missing_handlers:
            # Skip reserved buttons that have special handling
            if button_id in self.RESERVED_IDS:
                continue
                
            # If we get here, the button has no handler
            result.add_issue(
                ValidationLevel.ERROR,
                f"Button '{button_id}' has no registered handler",
                button_id=button_id,
                suggestion=f"Register handler with: self.register_button_handler('{button_id}', handler_method)",
                auto_fixable=True
            )
        
        # Check for orphaned handlers
        for handler_id in result.orphaned_handlers:
            # Skip reserved handlers
            if handler_id in self.RESERVED_IDS:
                continue
                
            # If we get here, the handler has no corresponding button
            result.add_issue(
                ValidationLevel.INFO,  # Downgrade to INFO level
                f"Handler '{handler_id}' has no corresponding button",
                button_id=handler_id,
                suggestion=f"Add button with ID '{handler_id}' or remove unused handler"
            )
    
    def _validate_reserved_ids(self, button_ids: Set[str], handler_ids: Set[str], 
                             result: ButtonValidationResult) -> None:
        """Validate reserved ID usage."""
        for reserved_id in self.RESERVED_IDS:
            if reserved_id in button_ids and reserved_id not in handler_ids:
                # Reserved buttons should have automatic handlers
                if reserved_id not in ['save', 'reset']:  # These are handled by BaseUIModule
                    result.add_issue(
                        ValidationLevel.INFO,
                        f"Reserved button '{reserved_id}' detected",
                        button_id=reserved_id,
                        suggestion=f"Reserved buttons are handled automatically"
                    )
    
    def _identify_auto_fixes(self, ui_module, result: ButtonValidationResult) -> None:
        """Identify issues that can be automatically fixed."""
        for issue in result.issues:
            if issue.button_id and issue.level == ValidationLevel.ERROR:
                # Check if we can auto-create a handler
                if self._can_auto_create_handler(ui_module, issue.button_id):
                    issue.auto_fixable = True
    
    def _can_auto_create_handler(self, ui_module, button_id: str) -> bool:
        """Check if we can auto-create a handler for a button."""
        # Look for existing handler methods that match common patterns
        potential_methods = self.suggest_handler_name(button_id)
        
        for method_name in potential_methods:
            if hasattr(ui_module, method_name):
                method = getattr(ui_module, method_name)
                if callable(method):
                    return True
        
        # Check if we can create a default operation handler
        operation_name = f"{button_id}_operation"
        if hasattr(ui_module, operation_name):
            return True
        
        return False
    
    def _create_default_handler(self, ui_module, button_id: str) -> Optional[Callable]:
        """Create a default handler for a button."""
        # Try to find existing method
        potential_methods = self.suggest_handler_name(button_id)
        
        for method_name in potential_methods:
            if hasattr(ui_module, method_name):
                method = getattr(ui_module, method_name)
                if callable(method):
                    return lambda btn: method()
        
        # Try operation handler
        operation_name = f"{button_id}_operation"
        if hasattr(ui_module, operation_name):
            operation = getattr(ui_module, operation_name)
            if callable(operation):
                return lambda btn: operation()
        
        # Create placeholder handler
        def placeholder_handler(button):
            if hasattr(ui_module, 'logger'):
                ui_module.logger.warning(f"Button '{button_id}' clicked but no handler implemented")
            return {'success': False, 'message': f'Handler for {button_id} not implemented'}
        
        return placeholder_handler


def validate_button_handlers(ui_module, strict_mode: bool = False, 
                           auto_fix: bool = True) -> ButtonValidationResult:
    """
    Convenience function to validate button handlers for a UI module.
    
    Args:
        ui_module: UI module instance to validate
        strict_mode: Apply stricter validation rules
        auto_fix: Automatically fix common issues
        
    Returns:
        ButtonValidationResult with validation details
    """
    validator = ButtonHandlerValidator(strict_mode=strict_mode)
    result = validator.validate_module(ui_module)
    
    if auto_fix and not result.is_valid:
        result = validator.auto_fix_issues(ui_module, result)
    
    return result