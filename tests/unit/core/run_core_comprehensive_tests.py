#!/usr/bin/env python3
"""
Comprehensive Core Module Test Runner

This script runs all comprehensive tests for the core infrastructure modules,
providing detailed reporting and validation of the core SmartCash UI system.

Modules Tested:
- Core Initializers (BaseInitializer and implementations)
- Core Handlers (BaseHandler and implementations)  
- Core Shared Components (SharedConfigManager)
- UI Components (ActionContainer, OperationContainer, etc.)

The runner provides:
- Individual module test results
- Integration validation
- Performance benchmarks
- Error analysis
- Comprehensive reporting
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_test_file(test_file: str, description: str) -> Tuple[bool, float, Dict[str, Any]]:
    """Run a specific test file and return results.
    
    Args:
        test_file: Path to the test file
        description: Description of the test
        
    Returns:
        Tuple of (success, success_rate, details)
    """
    print(f"\n🧪 Running {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run pytest on the specific file
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=project_root, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        output = result.stdout + result.stderr
        
        # Parse test results
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        
        for line in output.split('\n'):
            if " PASSED " in line:
                passed_tests += 1
            elif " FAILED " in line:
                failed_tests += 1
            elif " ERROR " in line:
                error_tests += 1
            elif " SKIPPED " in line:
                skipped_tests += 1
        
        total_tests = passed_tests + failed_tests + error_tests + skipped_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine success
        success = result.returncode == 0 and failed_tests == 0 and error_tests == 0
        
        details = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'skipped': skipped_tests,
            'duration': duration,
            'output': output
        }
        
        # Print summary
        status = "✅ PASSED" if success else "❌ FAILED" if failed_tests > 0 or error_tests > 0 else "⚠️ PARTIAL"
        print(f"{status}: {description}")
        print(f"   📊 Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print(f"   ⏱️ Duration: {duration:.2f}s")
        
        if failed_tests > 0:
            print(f"   ❌ Failed: {failed_tests}")
        if error_tests > 0:
            print(f"   💥 Errors: {error_tests}")
        if skipped_tests > 0:
            print(f"   ⏭️ Skipped: {skipped_tests}")
        
        return success, success_rate, details
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (exceeded 5 minutes)")
        return False, 0, {'error': 'timeout', 'duration': 300}
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False, 0, {'error': str(e), 'duration': time.time() - start_time}


def run_direct_import_tests() -> Tuple[bool, float, Dict[str, Any]]:
    """Run direct import tests for core modules.
    
    Returns:
        Tuple of (success, success_rate, details)
    """
    print(f"\n🔧 Running Direct Import Tests")
    print("=" * 60)
    
    import_results = []
    start_time = time.time()
    
    # Test core initializers
    try:
        from smartcash.ui.core.initializers.base_initializer import BaseInitializer
        
        class TestInit(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "success"}
        
        test_init = TestInit("test")
        result = test_init.initialize()
        
        if result and result.get("status") == "success":
            import_results.append(("Core Initializers", True, "Direct instantiation successful"))
        else:
            import_results.append(("Core Initializers", False, "Instantiation failed"))
            
    except Exception as e:
        import_results.append(("Core Initializers", False, f"Import error: {e}"))
    
    # Test core handlers
    try:
        from smartcash.ui.core.handlers.base_handler import BaseHandler
        
        handler = BaseHandler("test_handler", "test_parent")
        result = handler.initialize()
        
        if result and result.get("status") == "success":
            import_results.append(("Core Handlers", True, "Direct instantiation successful"))
        else:
            import_results.append(("Core Handlers", False, "Instantiation failed"))
            
    except Exception as e:
        import_results.append(("Core Handlers", False, f"Import error: {e}"))
    
    # Test shared config manager
    try:
        from smartcash.ui.core.shared.shared_config_manager import SharedConfigManager
        
        manager = SharedConfigManager("test_module")
        manager.update_config("test", {"key": "value"})
        config = manager.get_config("test")
        
        if config and config.get("key") == "value":
            import_results.append(("Shared Config Manager", True, "Basic operations successful"))
        else:
            import_results.append(("Shared Config Manager", False, "Operations failed"))
            
    except Exception as e:
        import_results.append(("Shared Config Manager", False, f"Import error: {e}"))
    
    # Test UI components
    ui_component_results = []
    
    # ActionContainer
    try:
        from smartcash.ui.components.action_container import ActionContainer
        container = ActionContainer()
        ui_component_results.append(("ActionContainer", True))
    except Exception as e:
        ui_component_results.append(("ActionContainer", False))
    
    # OperationContainer
    try:
        from smartcash.ui.components.operation_container import create_operation_container
        container = create_operation_container("test")
        ui_component_results.append(("OperationContainer", True))
    except Exception as e:
        ui_component_results.append(("OperationContainer", False))
    
    # ChartContainer
    try:
        from smartcash.ui.components.chart_container import create_chart_container
        container = create_chart_container("test")
        ui_component_results.append(("ChartContainer", True))
    except Exception as e:
        ui_component_results.append(("ChartContainer", False))
    
    # LogAccordion
    try:
        from smartcash.ui.components.log_accordion.log_accordion import LogAccordion
        accordion = LogAccordion()
        ui_component_results.append(("LogAccordion", True))
    except Exception as e:
        ui_component_results.append(("LogAccordion", False))
    
    # ConfirmationDialog
    try:
        from smartcash.ui.components.dialog.confirmation_dialog import ConfirmationDialog
        dialog = ConfirmationDialog("test")
        ui_component_results.append(("ConfirmationDialog", True))
    except Exception as e:
        ui_component_results.append(("ConfirmationDialog", False))
    
    # Calculate UI components success
    ui_success_count = sum(1 for _, success in ui_component_results if success)
    ui_total_count = len(ui_component_results)
    ui_success_rate = (ui_success_count / ui_total_count * 100) if ui_total_count > 0 else 0
    
    if ui_success_rate >= 80:
        import_results.append(("UI Components", True, f"{ui_success_count}/{ui_total_count} components working"))
    else:
        import_results.append(("UI Components", False, f"Only {ui_success_count}/{ui_total_count} components working"))
    
    # Calculate overall results
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_imports = len(import_results)
    success_rate = (successful_imports / total_imports * 100) if total_imports > 0 else 0
    duration = time.time() - start_time
    
    # Print results
    for name, success, message in import_results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}: {message}")
    
    # Print UI component details
    print(f"\n   🧩 UI Component Details:")
    for name, success in ui_component_results:
        status = "✅" if success else "❌"
        print(f"      {status} {name}")
    
    overall_success = success_rate >= 75
    status = "✅ PASSED" if overall_success else "❌ FAILED"
    print(f"\n{status}: Direct Import Tests ({success_rate:.1f}% success rate)")
    print(f"   ⏱️ Duration: {duration:.2f}s")
    
    details = {
        'import_results': import_results,
        'ui_component_results': ui_component_results,
        'successful_imports': successful_imports,
        'total_imports': total_imports,
        'duration': duration
    }
    
    return overall_success, success_rate, details


def run_integration_validation() -> Tuple[bool, float, Dict[str, Any]]:
    """Run integration validation tests.
    
    Returns:
        Tuple of (success, success_rate, details)
    """
    print(f"\n🔗 Running Integration Validation")
    print("=" * 60)
    
    validation_results = []
    start_time = time.time()
    
    # Test cross-module integration
    try:
        from smartcash.ui.core.initializers.base_initializer import BaseInitializer
        from smartcash.ui.core.handlers.base_handler import BaseHandler
        from smartcash.ui.core.shared.shared_config_manager import SharedConfigManager
        
        # Test integration workflow
        manager = SharedConfigManager("integration_test")
        
        class IntegrationInitializer(BaseInitializer):
            def _initialize_impl(self, **kwargs):
                return {"status": "integrated", "data": kwargs}
        
        handler = BaseHandler("integration_handler", "test")
        initializer = IntegrationInitializer("integration_init", "test")
        
        # Test workflow
        config = {"integration": True, "test_data": "value"}
        manager.update_config("test_config", config)
        
        init_result = initializer.initialize(config_manager=manager)
        handler_result = handler.initialize()
        
        retrieved_config = manager.get_config("test_config")
        
        if (init_result and handler_result and retrieved_config and 
            init_result.get("status") == "integrated" and
            handler_result.get("status") == "success" and
            retrieved_config.get("integration")):
            validation_results.append(("Core Module Integration", True, "All modules integrated successfully"))
        else:
            validation_results.append(("Core Module Integration", False, "Integration workflow failed"))
            
    except Exception as e:
        validation_results.append(("Core Module Integration", False, f"Integration error: {e}"))
    
    # Test error handling integration
    try:
        from smartcash.ui.core.errors import SmartCashUIError, ErrorContext
        
        # Test error context creation
        context = ErrorContext(component="test", operation="integration", details={"test": True})
        
        # Test error handling in handler
        handler = BaseHandler("error_test", "test")
        
        try:
            handler.handle_error("Integration test error", test_context="integration")
        except SmartCashUIError as e:
            if "Integration test error" in str(e):
                validation_results.append(("Error Handling Integration", True, "Error handling works correctly"))
            else:
                validation_results.append(("Error Handling Integration", False, "Error message incorrect"))
        except Exception as e:
            validation_results.append(("Error Handling Integration", False, f"Unexpected error: {e}"))
        else:
            validation_results.append(("Error Handling Integration", False, "Error not raised"))
            
    except Exception as e:
        validation_results.append(("Error Handling Integration", False, f"Error handling test failed: {e}"))
    
    # Test component integration with core
    component_integration_success = True
    component_errors = []
    
    try:
        from smartcash.ui.components.action_container import ActionContainer
        from smartcash.ui.core.shared.shared_config_manager import get_shared_config_manager
        
        # Test component with shared config
        manager = get_shared_config_manager("component_test")
        container = ActionContainer()
        
        # Add action that uses shared config
        def config_action():
            config = manager.get_config("action_config") or {}
            config["actions_executed"] = config.get("actions_executed", 0) + 1
            manager.update_config("action_config", config)
            return config
        
        container.add_action("Config Action", config_action)
        
        # Execute action
        result = container.execute_action("Config Action")
        if result and result.get("actions_executed") == 1:
            validation_results.append(("Component Integration", True, "UI components integrate with core"))
        else:
            validation_results.append(("Component Integration", False, "Component integration failed"))
            
    except Exception as e:
        validation_results.append(("Component Integration", False, f"Component integration error: {e}"))
    
    # Calculate results
    successful_validations = sum(1 for _, success, _ in validation_results if success)
    total_validations = len(validation_results)
    success_rate = (successful_validations / total_validations * 100) if total_validations > 0 else 0
    duration = time.time() - start_time
    
    # Print results
    for name, success, message in validation_results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}: {message}")
    
    overall_success = success_rate >= 80
    status = "✅ PASSED" if overall_success else "❌ FAILED"
    print(f"\n{status}: Integration Validation ({success_rate:.1f}% success rate)")
    print(f"   ⏱️ Duration: {duration:.2f}s")
    
    details = {
        'validation_results': validation_results,
        'successful_validations': successful_validations,
        'total_validations': total_validations,
        'duration': duration
    }
    
    return overall_success, success_rate, details


def main():
    """Main test runner function."""
    print("🚀 SmartCash Core Module Comprehensive Test Suite")
    print("=" * 80)
    print("Testing Core Infrastructure: Initializers → Handlers → Shared → UI Components")
    print("=" * 80)
    
    # Store all test results
    test_results = {}
    total_start_time = time.time()
    
    # 1. Direct Import Tests
    import_success, import_rate, import_details = run_direct_import_tests()
    test_results['direct_imports'] = (import_success, import_rate, import_details)
    
    # 2. Core Initializers Comprehensive Tests
    initializers_success, initializers_rate, initializers_details = run_test_file(
        "tests/unit/core/test_core_initializers_comprehensive.py",
        "Core Initializers Comprehensive Tests"
    )
    test_results['core_initializers'] = (initializers_success, initializers_rate, initializers_details)
    
    # 3. Core Handlers Comprehensive Tests
    handlers_success, handlers_rate, handlers_details = run_test_file(
        "tests/unit/core/test_core_handlers_comprehensive.py", 
        "Core Handlers Comprehensive Tests"
    )
    test_results['core_handlers'] = (handlers_success, handlers_rate, handlers_details)
    
    # 4. Core Shared Components Comprehensive Tests
    shared_success, shared_rate, shared_details = run_test_file(
        "tests/unit/core/test_core_shared_comprehensive.py",
        "Core Shared Components Comprehensive Tests"
    )
    test_results['core_shared'] = (shared_success, shared_rate, shared_details)
    
    # 5. UI Components Comprehensive Tests
    ui_success, ui_rate, ui_details = run_test_file(
        "tests/unit/core/test_ui_components_comprehensive.py",
        "UI Components Comprehensive Tests"
    )
    test_results['ui_components'] = (ui_success, ui_rate, ui_details)
    
    # 6. Integration Validation
    integration_success, integration_rate, integration_details = run_integration_validation()
    test_results['integration'] = (integration_success, integration_rate, integration_details)
    
    # Calculate overall statistics
    total_duration = time.time() - total_start_time
    
    all_success_rates = [rate for success, rate, details in test_results.values() if rate > 0]
    overall_success_rate = sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0
    
    successful_modules = sum(1 for success, rate, details in test_results.values() if success)
    total_modules = len(test_results)
    
    # Calculate total test counts
    total_tests_run = 0
    total_tests_passed = 0
    total_tests_failed = 0
    
    for name, (success, rate, details) in test_results.items():
        if isinstance(details, dict) and 'total_tests' in details:
            total_tests_run += details['total_tests']
            total_tests_passed += details.get('passed', 0)
            total_tests_failed += details.get('failed', 0) + details.get('errors', 0)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 CORE MODULE COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Test Summary:")
    print(f"   • Total Duration: {total_duration:.2f}s")
    print(f"   • Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"   • Successful Modules: {successful_modules}/{total_modules}")
    if total_tests_run > 0:
        print(f"   • Total Tests: {total_tests_run} ({total_tests_passed} passed, {total_tests_failed} failed)")
    
    print(f"\n🧪 Individual Module Results:")
    for name, (success, rate, details) in test_results.items():
        status = "✅" if success else "❌"
        duration = details.get('duration', 0) if isinstance(details, dict) else 0
        
        module_name = name.replace('_', ' ').title()
        print(f"   {status} {module_name}: {rate:.1f}% ({duration:.2f}s)")
        
        # Show test details if available
        if isinstance(details, dict) and 'total_tests' in details:
            passed = details.get('passed', 0)
            total = details.get('total_tests', 0)
            failed = details.get('failed', 0)
            errors = details.get('errors', 0)
            
            if total > 0:
                print(f"      📝 Tests: {passed}/{total} passed", end="")
                if failed > 0:
                    print(f", {failed} failed", end="")
                if errors > 0:
                    print(f", {errors} errors", end="")
                print()
    
    # Integration status
    print(f"\n🔗 Integration Status:")
    integration_success, integration_rate, integration_details = test_results['integration']
    if isinstance(integration_details, dict) and 'validation_results' in integration_details:
        for name, success, message in integration_details['validation_results']:
            status = "✅" if success else "❌"
            print(f"   {status} {name}")
    
    # Determine overall result
    core_modules_success = (
        initializers_success and handlers_success and 
        shared_success and ui_success
    )
    
    print(f"\n🎯 Final Assessment:")
    if integration_success and core_modules_success and overall_success_rate >= 90:
        print("🎉 EXCELLENT: Core infrastructure is production ready!")
        print("✅ All core modules pass comprehensive testing")
        print("✅ Integration validation successful")
        print("✅ Ready for full SmartCash UI deployment")
        exit_code = 0
    elif integration_success and core_modules_success and overall_success_rate >= 75:
        print("✅ GOOD: Core infrastructure is solid with minor issues")
        print("🔧 Core functionality working, some edge cases need attention")
        print("✅ Safe to proceed with development")
        exit_code = 0
    elif core_modules_success and overall_success_rate >= 60:
        print("⚠️ ACCEPTABLE: Core infrastructure functional but needs improvement")
        print("🔧 Basic functionality working, integration needs work")
        print("🚧 Continue development with caution")
        exit_code = 1
    else:
        print("❌ CRITICAL: Significant issues in core infrastructure")
        print("🚨 Core modules have serious problems")
        print("🛑 Fix critical issues before proceeding")
        exit_code = 2
    
    # Module status breakdown
    print(f"\n📋 Module Status Breakdown:")
    completed = sum(1 for success, rate, _ in test_results.values() if success and rate >= 90)
    partial = sum(1 for success, rate, _ in test_results.values() if success and 60 <= rate < 90)
    failing = sum(1 for success, rate, _ in test_results.values() if not success or rate < 60)
    
    print(f"   • ✅ Excellent (≥90%): {completed}/{total_modules} modules")
    print(f"   • ⚠️ Good (60-89%): {partial}/{total_modules} modules") 
    print(f"   • ❌ Needs Work (<60%): {failing}/{total_modules} modules")
    
    if exit_code == 0:
        print(f"\n🚀 CORE INFRASTRUCTURE TESTING SUCCESSFUL!")
    else:
        print(f"\n🔧 Core infrastructure needs attention before production use")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()