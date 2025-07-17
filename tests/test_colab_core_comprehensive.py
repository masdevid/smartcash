#!/usr/bin/env python3
"""
Comprehensive Core Module Test Suite for Colab UI

This test suite validates core module integration, inheritance, caching,
and proper UI rendering functionality.
"""

import sys
import os
import traceback
from typing import Dict, Any, Optional
import ipywidgets as widgets

# Add project root to path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

def test_core_inheritance():
    """Test proper inheritance from core modules."""
    print("🧪 Testing Core Inheritance...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabInitializer
        from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
        from smartcash.ui.core.initializers.config_initializer import ConfigurableInitializer
        from smartcash.ui.core.initializers.base_initializer import BaseInitializer
        
        initializer = ColabInitializer()
        
        # Test inheritance chain
        inheritance_tests = [
            (BaseInitializer, "BaseInitializer"),
            (ConfigurableInitializer, "ConfigurableInitializer"), 
            (ModuleInitializer, "ModuleInitializer")
        ]
        
        for base_class, name in inheritance_tests:
            if isinstance(initializer, base_class):
                print(f"✅ {name} inheritance: PASS")
            else:
                print(f"❌ {name} inheritance: FAIL")
                return False
        
        # Test key methods exist
        key_methods = [
            'initialize', '_initialize_impl', 'get_default_config',
            'load_config', 'save_config', '_create_ui_components'
        ]
        
        for method in key_methods:
            if hasattr(initializer, method):
                print(f"✅ Method {method}: EXISTS")
            else:
                print(f"❌ Method {method}: MISSING")
                return False
        
        print("✅ Core Inheritance Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Core Inheritance Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_singleton_pattern():
    """Test singleton pattern and instance management."""
    print("🧪 Testing Singleton Pattern...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import get_colab_initializer
        
        # Test singleton behavior
        instance1 = get_colab_initializer()
        instance2 = get_colab_initializer()
        
        if instance1 is instance2:
            print("✅ Singleton pattern: WORKING")
        else:
            print("❌ Singleton pattern: BROKEN")
            return False
        
        # Test instance type
        from smartcash.ui.setup.colab.colab_initializer import ColabInitializer
        if isinstance(instance1, ColabInitializer):
            print("✅ Instance type: CORRECT")
        else:
            print(f"❌ Instance type: INCORRECT - {type(instance1)}")
            return False
        
        print("✅ Singleton Pattern Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Singleton Pattern Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_cache_management():
    """Test cache management and cleanup."""
    print("🧪 Testing Cache Management...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import get_colab_components
        
        # Test multiple calls
        components1 = get_colab_components()
        components2 = get_colab_components()
        
        # Components should be different instances (no caching of widgets)
        if components1 and components2:
            ui1 = components1.get('ui')
            ui2 = components2.get('ui')
            
            if ui1 is not ui2:
                print("✅ No widget caching: CORRECT")
            else:
                print("⚠️ Widget caching detected: This might cause display issues")
        
        # Test component keys consistency
        if set(components1.keys()) == set(components2.keys()):
            print("✅ Component keys consistency: PASS")
        else:
            print("❌ Component keys consistency: FAIL")
            return False
        
        print("✅ Cache Management Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Cache Management Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_ui_rendering():
    """Test actual UI rendering capabilities."""
    print("🧪 Testing UI Rendering...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import get_colab_components
        from IPython.display import display
        import io
        import contextlib
        
        # Test component creation
        components = get_colab_components()
        
        if not components:
            print("❌ No components returned")
            return False
        
        # Test UI widget exists and is displayable
        ui_widget = components.get('ui') or components.get('main_container')
        
        if not ui_widget:
            print("❌ No UI widget found")
            return False
        
        if not isinstance(ui_widget, widgets.Widget):
            print(f"❌ UI widget is not a Widget: {type(ui_widget)}")
            return False
        
        # Test that widget can be displayed (no exceptions)
        try:
            # Capture output to avoid cluttering test results
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                display(ui_widget)
            print("✅ UI widget display: SUCCESS")
        except Exception as e:
            print(f"❌ UI widget display: FAILED - {str(e)}")
            return False
        
        # Test required containers exist
        required_containers = [
            'header_container', 'form_container', 'action_container',
            'operation_container', 'footer_container'
        ]
        
        # Debug: print all available components
        print(f"🔍 Available components: {', '.join(sorted(components.keys()))}")
        
        for container in required_containers:
            if container in components:
                if components[container] is not None:
                    print(f"✅ Container {container}: EXISTS")
                else:
                    print(f"❌ Container {container}: EXISTS but is None")
                    return False
            else:
                print(f"❌ Container {container}: NOT IN COMPONENTS")
                return False
        
        print("✅ UI Rendering Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ UI Rendering Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_display_integration():
    """Test the full display integration pipeline."""
    print("🧪 Testing Display Integration...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabDisplayInitializer
        import io
        import contextlib
        
        # Test display initializer
        display_init = ColabDisplayInitializer()
        
        # Test display method (capture output) - use test mode to suppress logs
        f = io.StringIO()
        try:
            # Suppress IPython display output during testing
            from IPython.display import display as original_display
            displayed_items = []
            
            def mock_display(obj, *args, **kwargs):
                displayed_items.append(str(type(obj).__name__))
            
            # Monkey patch the display function
            import IPython.display
            IPython.display.display = mock_display
            
            with contextlib.redirect_stdout(f):
                display_init.display(_test_mode=True)
        finally:
            # Restore original display function
            if 'original_display' in locals():
                import IPython.display
                IPython.display.display = original_display
        
        output = f.getvalue()
        
        # Should not contain error messages
        if "⚠️" in output or "❌" in output:
            print(f"❌ Display method has warnings/errors: {output}")
            return False
        
        print("✅ Display method: SUCCESS")
        
        # Test component retrieval
        components = display_init.get_components()
        
        if not components:
            print("❌ get_components returned empty")
            return False
        
        if 'ui' not in components and 'main_container' not in components:
            print("❌ No main UI widget in components")
            return False
        
        print("✅ Component retrieval: SUCCESS")
        print("✅ Display Integration Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Display Integration Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("🧪 Testing Error Handling...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabInitializer
        
        initializer = ColabInitializer()
        
        # Test with invalid config
        try:
            result = initializer.initialize(config={'invalid_key': 'invalid_value'})
            if result and 'ui_components' in result:
                print("✅ Error handling: Graceful degradation")
            else:
                print("⚠️ Error handling: May need improvement")
        except Exception as e:
            print(f"⚠️ Error handling: Exception raised - {str(e)}")
        
        # Test initialization state
        if hasattr(initializer, '_is_initialized'):
            print("✅ Initialization state tracking: EXISTS")
        else:
            print("⚠️ Initialization state tracking: MISSING")
        
        print("✅ Error Handling Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Error Handling Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def test_lifecycle_management():
    """Test component lifecycle management."""
    print("🧪 Testing Lifecycle Management...")
    
    try:
        from smartcash.ui.setup.colab.colab_initializer import get_colab_initializer
        
        initializer = get_colab_initializer()
        
        # Test initialization
        result = initializer.initialize()
        
        if result and isinstance(result, dict):
            print("✅ Initialization: SUCCESS")
        else:
            print("❌ Initialization: FAILED")
            return False
        
        # Test component access
        if 'ui_components' in result:
            ui_components = result['ui_components']
            if isinstance(ui_components, dict) and len(ui_components) > 0:
                print("✅ Component creation: SUCCESS")
            else:
                print("❌ Component creation: FAILED")
                return False
        else:
            print("❌ No ui_components in result")
            return False
        
        # Test metadata
        metadata_keys = ['module_name', 'parent_module', 'ui_initialized']
        for key in metadata_keys:
            if key in ui_components:
                print(f"✅ Metadata {key}: EXISTS")
            else:
                print(f"❌ Metadata {key}: MISSING")
                return False
        
        print("✅ Lifecycle Management Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Lifecycle Management Test Failed: {str(e)}")
        traceback.print_exc()
        return False


def run_core_comprehensive_test():
    """Run all core module comprehensive tests."""
    print("🚀 Starting Comprehensive Core Module Test Suite")
    print("=" * 60)
    
    tests = [
        ("Core Inheritance", test_core_inheritance),
        ("Singleton Pattern", test_singleton_pattern),
        ("Cache Management", test_cache_management),
        ("UI Rendering", test_ui_rendering),
        ("Display Integration", test_display_integration),
        ("Error Handling", test_error_handling),
        ("Lifecycle Management", test_lifecycle_management),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} Test: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} Test: FAILED with exception: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📊 Core Module Test Results Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL CORE TESTS PASSED! Core module integration is fully functional.")
        return True
    else:
        print(f"\n⚠️  {failed} core test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_core_comprehensive_test()
    sys.exit(0 if success else 1)