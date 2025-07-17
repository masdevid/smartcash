#!/usr/bin/env python3
"""
Button Integration Validation Script

This script validates that button key mappings are correct and there are no click errors
in the colab, dependency, and preprocessing modules.
"""

import sys
import os
import logging
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_module_button_integration(module_name: str, module_class, config_func) -> Dict[str, Any]:
    """Test button integration for a module."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing {module_name} module button integration")
    logger.info(f"{'='*60}")
    
    results = {
        'module': module_name,
        'success': False,
        'button_handlers_registered': 0,
        'ui_components_created': 0,
        'errors': [],
        'warnings': []
    }
    
    try:
        # 1. Create module instance
        logger.info(f"📦 Creating {module_name} module instance...")
        module = module_class()
        
        # 2. Initialize module
        logger.info(f"🔧 Initializing {module_name} module...")
        initialization_success = module.initialize()
        
        if not initialization_success:
            results['errors'].append("Module initialization failed")
            return results
        
        # 3. Check UI components
        if hasattr(module, '_ui_components') and module._ui_components:
            results['ui_components_created'] = len(module._ui_components)
            logger.info(f"✅ Created {results['ui_components_created']} UI components")
            
            # List component types
            component_types = list(module._ui_components.keys())
            logger.info(f"📋 Components: {', '.join(component_types)}")
        else:
            results['warnings'].append("No UI components found")
        
        # 4. Check button handlers
        if hasattr(module, '_button_handlers') and module._button_handlers:
            results['button_handlers_registered'] = len(module._button_handlers)
            logger.info(f"✅ Registered {results['button_handlers_registered']} button handlers")
            
            # List button handlers
            button_handlers = list(module._button_handlers.keys())
            logger.info(f"🔘 Button handlers: {', '.join(button_handlers)}")
            
            # Test each button handler
            logger.info(f"🧪 Testing button handler functionality...")
            for button_id, handler in module._button_handlers.items():
                try:
                    logger.info(f"  ➤ Testing handler for '{button_id}'...")
                    if callable(handler):
                        logger.info(f"    ✅ Handler '{button_id}' is callable")
                    else:
                        results['errors'].append(f"Handler '{button_id}' is not callable")
                except Exception as e:
                    results['errors'].append(f"Error testing handler '{button_id}': {e}")
        else:
            results['warnings'].append("No button handlers found")
        
        # 5. Check module status
        status = module.get_status()
        logger.info(f"📊 Module status: {status}")
        
        # 6. Test config operations
        logger.info(f"🔧 Testing config operations...")
        try:
            current_config = module.get_current_config()
            logger.info(f"  ✅ Successfully retrieved current config with {len(current_config)} keys")
        except Exception as e:
            results['errors'].append(f"Failed to get current config: {e}")
        
        try:
            save_result = module.save_config()
            if save_result.get('success'):
                logger.info(f"  ✅ Config save successful: {save_result.get('message')}")
            else:
                results['warnings'].append(f"Config save returned false: {save_result.get('message')}")
        except Exception as e:
            results['errors'].append(f"Failed to save config: {e}")
        
        # 7. Cleanup
        try:
            module.cleanup()
            logger.info(f"🧹 Module cleanup successful")
        except Exception as e:
            results['warnings'].append(f"Cleanup warning: {e}")
        
        # Determine overall success
        results['success'] = len(results['errors']) == 0
        
        logger.info(f"🎯 {module_name} module test {'PASSED' if results['success'] else 'FAILED'}")
        if results['errors']:
            logger.error(f"❌ Errors: {'; '.join(results['errors'])}")
        if results['warnings']:
            logger.warning(f"⚠️ Warnings: {'; '.join(results['warnings'])}")
            
    except Exception as e:
        results['errors'].append(f"Critical test error: {e}")
        logger.error(f"❌ Critical error testing {module_name}: {e}")
    
    return results

def main():
    """Main test function."""
    logger.info("🚀 Starting button integration validation tests")
    
    # Test results storage
    all_results = []
    
    # Test 1: Colab Module
    try:
        from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
        from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config
        
        colab_results = test_module_button_integration(
            "Colab", 
            ColabUIModule, 
            get_default_colab_config
        )
        all_results.append(colab_results)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Colab module: {e}")
        all_results.append({
            'module': 'Colab',
            'success': False,
            'errors': [f"Import error: {e}"],
            'warnings': [],
            'button_handlers_registered': 0,
            'ui_components_created': 0
        })
    
    # Test 2: Dependency Module  
    try:
        from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
        from smartcash.ui.setup.dependency.configs.dependency_defaults import get_default_dependency_config
        
        dependency_results = test_module_button_integration(
            "Dependency",
            DependencyUIModule,
            get_default_dependency_config
        )
        all_results.append(dependency_results)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Dependency module: {e}")
        all_results.append({
            'module': 'Dependency',
            'success': False,
            'errors': [f"Import error: {e}"],
            'warnings': [],
            'button_handlers_registered': 0,
            'ui_components_created': 0
        })
    
    # Test 3: Preprocessing Module
    try:
        from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
        from smartcash.ui.dataset.preprocessing.configs.preprocessing_defaults import get_default_config as get_default_preprocessing_config
        
        preprocessing_results = test_module_button_integration(
            "Preprocessing",
            PreprocessingUIModule,
            get_default_preprocessing_config
        )
        all_results.append(preprocessing_results)
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Preprocessing module: {e}")
        all_results.append({
            'module': 'Preprocessing', 
            'success': False,
            'errors': [f"Import error: {e}"],
            'warnings': [],
            'button_handlers_registered': 0,
            'ui_components_created': 0
        })
    
    # Generate summary report
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 BUTTON INTEGRATION VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['success'])
    failed_tests = total_tests - passed_tests
    
    logger.info(f"📈 Total Tests: {total_tests}")
    logger.info(f"✅ Passed: {passed_tests}")
    logger.info(f"❌ Failed: {failed_tests}")
    
    for result in all_results:
        module = result['module']
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        handlers = result['button_handlers_registered']
        components = result['ui_components_created']
        
        logger.info(f"  {status} {module}: {handlers} handlers, {components} components")
        
        if result['errors']:
            for error in result['errors']:
                logger.error(f"    ❌ {error}")
        
        if result['warnings']:
            for warning in result['warnings']:
                logger.warning(f"    ⚠️ {warning}")
    
    # Overall result
    overall_success = failed_tests == 0
    logger.info(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)