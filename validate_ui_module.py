#!/usr/bin/env python3
"""
SmartCash UI Module Validation Script

This script validates that UI modules follow the standardized template structure
and container ordering requirements.

Usage:
    python validate_ui_module.py <module_ui_file>
    python validate_ui_module.py smartcash/ui/dataset/preprocess/components/preprocess_ui.py
"""

import ast
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Required container order according to UI module structure
REQUIRED_CONTAINER_ORDER = [
    'header_container',
    'form_container', 
    'action_container',
    'summary_container',  # Optional but if present, should be in this position
    'operation_container',
    'footer_container'
]

# Required imports for standardized UI modules
REQUIRED_IMPORTS = [
    'smartcash.ui.components.header_container',
    'smartcash.ui.components.form_container',
    'smartcash.ui.components.action_container',
    'smartcash.ui.components.operation_container',
    'smartcash.ui.components.footer_container'
]

# Required return keys for UI components
REQUIRED_RETURN_KEYS = [
    'ui',
    'header_container',
    'form_container',
    'action_container',
    'operation_container',
    'footer_container'
]

# Required metadata keys
REQUIRED_METADATA_KEYS = [
    'module_name',
    'parent_module',
    'ui_initialized',
    'config'
]

class UIModuleValidator:
    """Validates UI modules against the standardized template."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.tree = None
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate(self) -> Dict[str, Any]:
        """
        Validate the UI module file.
        
        Returns:
            Dictionary containing validation results
        """
        if not self.file_path.exists():
            self.errors.append(f"File not found: {self.file_path}")
            return self._get_results()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.tree = ast.parse(content)
            
            # Run validation checks
            self._validate_imports()
            self._validate_main_function()
            self._validate_constants()
            self._validate_helper_functions()
            self._validate_docstrings()
            self._validate_error_handling()
            
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
        except Exception as e:
            self.errors.append(f"Error parsing file: {e}")
        
        return self._get_results()
    
    def _validate_imports(self):
        """Validate that required imports are present."""
        imports = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check for required imports
        missing_imports = []
        for required_import in REQUIRED_IMPORTS:
            if required_import not in imports:
                missing_imports.append(required_import)
        
        if missing_imports:
            self.errors.append(f"Missing required imports: {', '.join(missing_imports)}")
        else:
            self.info.append("✓ All required imports present")
    
    def _validate_main_function(self):
        """Validate the main UI creation function."""
        main_functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('create_') and node.name.endswith('_ui'):
                    main_functions.append(node)
        
        if not main_functions:
            self.errors.append("No main UI creation function found (should be create_[module]_ui)")
            return
        
        if len(main_functions) > 1:
            self.warnings.append(f"Multiple UI creation functions found: {[f.name for f in main_functions]}")
        
        main_func = main_functions[0]
        self.info.append(f"✓ Main UI function found: {main_func.name}")
        
        # Check function signature
        self._validate_function_signature(main_func)
        
        # Check function body for container creation
        self._validate_container_creation(main_func)
        
        # Check return statement
        self._validate_return_statement(main_func)
    
    def _validate_function_signature(self, func_node: ast.FunctionDef):
        """Validate function signature."""
        args = func_node.args
        
        # Check for config parameter
        config_param_found = False
        for arg in args.args:
            if arg.arg == 'config':
                config_param_found = True
                break
        
        if not config_param_found:
            self.errors.append("Main function missing 'config' parameter")
        else:
            self.info.append("✓ Config parameter present")
        
        # Check for **kwargs
        if not args.kwarg:
            self.warnings.append("Main function missing **kwargs parameter")
    
    def _validate_container_creation(self, func_node: ast.FunctionDef):
        """Validate container creation in function body."""
        container_assignments = {}
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.endswith('_container'):
                            container_assignments[target.id] = node
        
        # Check for required containers
        missing_containers = []
        for container in REQUIRED_CONTAINER_ORDER:
            if container not in container_assignments and container != 'summary_container':
                missing_containers.append(container)
        
        if missing_containers:
            self.errors.append(f"Missing required containers: {', '.join(missing_containers)}")
        else:
            self.info.append("✓ All required containers present")
        
        # Check container creation calls
        for container_name, assignment_node in container_assignments.items():
            if isinstance(assignment_node.value, ast.Call):
                func_name = None
                if isinstance(assignment_node.value.func, ast.Name):
                    func_name = assignment_node.value.func.id
                elif isinstance(assignment_node.value.func, ast.Attribute):
                    func_name = assignment_node.value.func.attr
                
                expected_func = f"create_{container_name}"
                if func_name != expected_func:
                    self.warnings.append(f"Container {container_name} not created with {expected_func}")
    
    def _validate_return_statement(self, func_node: ast.FunctionDef):
        """Validate return statement structure."""
        return_nodes = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return_nodes.append(node)
        
        if not return_nodes:
            self.errors.append("Main function missing return statement")
            return
        
        last_return = return_nodes[-1]
        
        # Check if returning a dictionary
        if not isinstance(last_return.value, ast.Name):
            self.warnings.append("Return statement should return ui_components dictionary")
            return
        
        # Check if ui_components is built correctly
        self._validate_ui_components_dict(func_node)
    
    def _validate_ui_components_dict(self, func_node: ast.FunctionDef):
        """Validate ui_components dictionary structure."""
        # This is a simplified check - could be more comprehensive
        ui_components_assignments = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'ui_components':
                        ui_components_assignments.append(node)
        
        if not ui_components_assignments:
            self.errors.append("ui_components dictionary not found")
            return
        
        self.info.append("✓ ui_components dictionary found")
    
    def _validate_constants(self):
        """Validate module constants."""
        constants = {}
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants[target.id] = node
        
        required_constants = ['UI_CONFIG', 'BUTTON_CONFIG']
        missing_constants = []
        
        for const in required_constants:
            if const not in constants:
                missing_constants.append(const)
        
        if missing_constants:
            self.warnings.append(f"Missing recommended constants: {', '.join(missing_constants)}")
        else:
            self.info.append("✓ Required constants present")
    
    def _validate_helper_functions(self):
        """Validate helper functions."""
        helper_functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('_create_module_'):
                    helper_functions.append(node.name)
        
        expected_helpers = [
            '_create_module_form_widgets',
            '_create_module_summary_content',
            '_create_module_info_box'
        ]
        
        missing_helpers = []
        for helper in expected_helpers:
            if helper not in helper_functions:
                missing_helpers.append(helper)
        
        if missing_helpers:
            self.warnings.append(f"Missing helper functions: {', '.join(missing_helpers)}")
        else:
            self.info.append("✓ Helper functions present")
    
    def _validate_docstrings(self):
        """Validate docstring presence."""
        functions_with_docstrings = 0
        total_functions = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    functions_with_docstrings += 1
        
        if total_functions > 0:
            docstring_ratio = functions_with_docstrings / total_functions
            if docstring_ratio < 0.8:
                self.warnings.append(f"Low docstring coverage: {docstring_ratio:.1%}")
            else:
                self.info.append(f"✓ Good docstring coverage: {docstring_ratio:.1%}")
    
    def _validate_error_handling(self):
        """Validate error handling decorator."""
        decorators = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('create_') and node.name.endswith('_ui'):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorators.append(decorator.id)
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name):
                                decorators.append(decorator.func.id)
        
        if 'handle_ui_errors' not in decorators:
            self.warnings.append("Main function missing error handling decorator")
        else:
            self.info.append("✓ Error handling decorator present")
    
    def _get_results(self) -> Dict[str, Any]:
        """Get validation results."""
        return {
            'file_path': str(self.file_path),
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'score': self._calculate_score()
        }
    
    def _calculate_score(self) -> float:
        """Calculate validation score."""
        total_checks = len(self.errors) + len(self.warnings) + len(self.info)
        if total_checks == 0:
            return 0.0
        
        passed_checks = len(self.info)
        warning_penalty = len(self.warnings) * 0.5
        error_penalty = len(self.errors) * 1.0
        
        score = max(0.0, (passed_checks - warning_penalty - error_penalty) / total_checks)
        return score * 100


def print_validation_results(results: Dict[str, Any]):
    """Print validation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"UI Module Validation Results")
    print(f"{'='*60}")
    print(f"File: {results['file_path']}")
    print(f"Valid: {'✓ PASS' if results['valid'] else '✗ FAIL'}")
    print(f"Score: {results['score']:.1f}%")
    print()
    
    if results['errors']:
        print("🔴 ERRORS:")
        for error in results['errors']:
            print(f"  • {error}")
        print()
    
    if results['warnings']:
        print("🟡 WARNINGS:")
        for warning in results['warnings']:
            print(f"  • {warning}")
        print()
    
    if results['info']:
        print("🟢 PASSED CHECKS:")
        for info in results['info']:
            print(f"  • {info}")
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if results['score'] < 50:
        print("  • This module needs significant restructuring to follow the template")
    elif results['score'] < 80:
        print("  • Address warnings to improve compliance with the template")
    else:
        print("  • Great job! This module follows the template well")
    
    print(f"{'='*60}")


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_ui_module.py <module_ui_file>")
        print("Example: python validate_ui_module.py smartcash/ui/dataset/preprocess/components/preprocess_ui.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"Validating UI module: {file_path}")
    
    validator = UIModuleValidator(file_path)
    results = validator.validate()
    
    print_validation_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results['valid'] else 1)


if __name__ == "__main__":
    main()