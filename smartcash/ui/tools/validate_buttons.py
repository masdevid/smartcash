#!/usr/bin/env python3
"""
Button validation CLI tool for SmartCash UI modules.

This tool validates button-handler synchronization across all UI modules
and provides detailed reports and auto-fixing capabilities.

Usage:
    python -m smartcash.ui.tools.validate_buttons [module_name] [options]
    
Examples:
    python -m smartcash.ui.tools.validate_buttons                    # Validate all modules
    python -m smartcash.ui.tools.validate_buttons dependency        # Validate specific module  
    python -m smartcash.ui.tools.validate_buttons --auto-fix        # Validate and auto-fix issues
    python -m smartcash.ui.tools.validate_buttons --strict          # Strict validation mode
"""

import sys
import os
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.ui.core.validation.button_validator import ButtonHandlerValidator, ValidationLevel


class ButtonValidationCLI:
    """CLI tool for button validation."""
    
    # Available UI modules for validation
    AVAILABLE_MODULES = {
        'dependency': ('smartcash.ui.setup.dependency.dependency_uimodule', 'DependencyUIModule'),
        'colab': ('smartcash.ui.setup.colab.colab_uimodule', 'ColabUIModule'),
        'downloader': ('smartcash.ui.dataset.downloader.downloader_uimodule', 'DownloaderUIModule'),
        'preprocessing': ('smartcash.ui.dataset.preprocessing.preprocessing_uimodule', 'PreprocessingUIModule'),
        'split': ('smartcash.ui.dataset.split.split_uimodule', 'SplitUIModule'),
        'augmentation': ('smartcash.ui.dataset.augmentation.augmentation_uimodule', 'AugmentationUIModule'),
        'backbone': ('smartcash.ui.model.backbone.backbone_uimodule', 'BackboneUIModule'),
        'pretrained': ('smartcash.ui.model.pretrained.pretrained_uimodule', 'PretrainedUIModule'),
        'hyperparameters': ('smartcash.ui.model.hyperparameters.hyperparameters_uimodule', 'HyperparametersUIModule'),
        'training': ('smartcash.ui.model.training.training_uimodule', 'TrainingUIModule'),
        'evaluation': ('smartcash.ui.model.evaluation.evaluation_uimodule', 'EvaluationUIModule')
    }
    
    def __init__(self):
        self.validator = ButtonHandlerValidator()
    
    def validate_module(self, module_name: str, auto_fix: bool = False, 
                       strict_mode: bool = False) -> Dict[str, Any]:
        """
        Validate a specific UI module.
        
        Args:
            module_name: Name of the module to validate
            auto_fix: Whether to auto-fix issues
            strict_mode: Whether to use strict validation
            
        Returns:
            Validation results dictionary
        """
        if module_name not in self.AVAILABLE_MODULES:
            return {
                'success': False,
                'error': f"Unknown module '{module_name}'. Available modules: {list(self.AVAILABLE_MODULES.keys())}"
            }
        
        try:
            # Import and create module
            module_path, class_name = self.AVAILABLE_MODULES[module_name]
            
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            module_class = getattr(module, class_name)
            
            # Create instance and initialize to get accurate button/handler data
            ui_module = module_class()
            
            # We need to initialize to get button registrations, but suppress UI output
            import io
            import contextlib
            
            # Capture output during initialization
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                initialized = ui_module.initialize()
            
            if not initialized:
                return {
                    'success': False,
                    'module_name': module_name,
                    'error': 'Failed to initialize module for validation'
                }
            
            # Set up validator with strict mode
            self.validator.strict_mode = strict_mode
            
            # Perform validation
            result = self.validator.validate_module(ui_module)
            
            # Auto-fix if requested
            if auto_fix:
                result = self.validator.auto_fix_issues(ui_module, result)
            
            return {
                'success': True,
                'module_name': module_name,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'module_name': module_name,
                'error': str(e)
            }
    
    def validate_all_modules(self, auto_fix: bool = False, 
                           strict_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Validate all available UI modules.
        
        Args:
            auto_fix: Whether to auto-fix issues
            strict_mode: Whether to use strict validation
            
        Returns:
            List of validation results for each module
        """
        results = []
        
        print(f"🔍 Validating {len(self.AVAILABLE_MODULES)} UI modules...")
        print("=" * 60)
        
        for module_name in self.AVAILABLE_MODULES:
            print(f"Validating {module_name}...")
            result = self.validate_module(module_name, auto_fix, strict_mode)
            results.append(result)
        
        return results
    
    def print_validation_report(self, results: List[Dict[str, Any]]) -> None:
        """Print a comprehensive validation report."""
        
        total_modules = len(results)
        successful_validations = sum(1 for r in results if r['success'])
        failed_validations = total_modules - successful_validations
        
        valid_modules = 0
        modules_with_errors = 0
        modules_with_warnings = 0
        total_auto_fixes = 0
        
        print("\n" + "=" * 80)
        print("🔍 BUTTON VALIDATION REPORT")
        print("=" * 80)
        
        for result in results:
            if not result['success']:
                print(f"\n❌ {result['module_name']}: FAILED TO VALIDATE")
                print(f"   Error: {result['error']}")
                continue
            
            validation_result = result['result']
            module_name = result['module_name']
            
            # Count statistics
            if validation_result.is_valid:
                valid_modules += 1
            if validation_result.has_errors:
                modules_with_errors += 1
            if validation_result.has_warnings:
                modules_with_warnings += 1
            total_auto_fixes += len(validation_result.auto_fixes_applied)
            
            # Print module status
            status_icon = "✅" if validation_result.is_valid else "❌"
            print(f"\n{status_icon} {module_name.upper()}")
            
            # Print statistics
            print(f"   Buttons: {len(validation_result.button_ids)}")
            print(f"   Handlers: {len(validation_result.handler_ids)}")
            
            if validation_result.missing_handlers:
                print(f"   Missing handlers: {validation_result.missing_handlers}")
            
            if validation_result.orphaned_handlers:
                print(f"   Orphaned handlers: {validation_result.orphaned_handlers}")
            
            # Print issues
            error_issues = [i for i in validation_result.issues if i.level == ValidationLevel.ERROR]
            warning_issues = [i for i in validation_result.issues if i.level == ValidationLevel.WARNING]
            
            if error_issues:
                print(f"   🔴 Errors ({len(error_issues)}):")
                for issue in error_issues:
                    print(f"      • {issue.message}")
                    if issue.suggestion:
                        print(f"        💡 {issue.suggestion}")
            
            if warning_issues:
                print(f"   🟡 Warnings ({len(warning_issues)}):")
                for issue in warning_issues:
                    print(f"      • {issue.message}")
                    if issue.suggestion:
                        print(f"        💡 {issue.suggestion}")
            
            # Print auto-fixes
            if validation_result.auto_fixes_applied:
                print(f"   🔧 Auto-fixes applied ({len(validation_result.auto_fixes_applied)}):")
                for fix in validation_result.auto_fixes_applied:
                    print(f"      • {fix}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"Total modules: {total_modules}")
        print(f"Successful validations: {successful_validations}")
        print(f"Failed validations: {failed_validations}")
        print(f"Valid modules: {valid_modules}")
        print(f"Modules with errors: {modules_with_errors}")
        print(f"Modules with warnings: {modules_with_warnings}")
        print(f"Total auto-fixes applied: {total_auto_fixes}")
        
        # Overall status
        if failed_validations > 0:
            print(f"\n❌ VALIDATION FAILED: {failed_validations} modules could not be validated")
        elif modules_with_errors > 0:
            print(f"\n❌ VALIDATION ISSUES: {modules_with_errors} modules have errors")
        elif modules_with_warnings > 0:
            print(f"\n⚠️ VALIDATION WARNINGS: {modules_with_warnings} modules have warnings")
        else:
            print(f"\n✅ ALL VALIDATIONS PASSED: All {valid_modules} modules are valid")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Validate button-handler synchronization in SmartCash UI modules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Validate all modules
  %(prog)s dependency               # Validate specific module
  %(prog)s --auto-fix               # Auto-fix common issues
  %(prog)s dependency --strict      # Strict validation for specific module
        """
    )
    
    parser.add_argument(
        'module',
        nargs='?',
        help='Specific module to validate (if not provided, validates all modules)'
    )
    
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help='Automatically fix common issues'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict validation mode'
    )
    
    parser.add_argument(
        '--list-modules',
        action='store_true',
        help='List available modules'
    )
    
    args = parser.parse_args()
    
    cli = ButtonValidationCLI()
    
    # List modules if requested
    if args.list_modules:
        print("Available modules for validation:")
        for module_name in sorted(cli.AVAILABLE_MODULES.keys()):
            print(f"  • {module_name}")
        return
    
    # Validate specific module or all modules
    if args.module:
        results = [cli.validate_module(args.module, args.auto_fix, args.strict)]
    else:
        results = cli.validate_all_modules(args.auto_fix, args.strict)
    
    # Print report
    cli.print_validation_report(results)
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r['success'])
    error_count = sum(1 for r in results if r['success'] and r['result'].has_errors)
    
    if failed_count > 0 or error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()