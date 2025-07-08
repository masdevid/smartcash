#!/usr/bin/env python3
"""
Simple test runner for evaluation module tests
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_evaluation_service():
    """Test evaluation service basic functionality."""
    print("Testing evaluation service...")
    
    try:
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        from smartcash.ui.model.evaluate.constants import DEFAULT_CONFIG, EvaluationPhase
        
        # Test initialization
        service = EvaluationService()
        assert service.config == DEFAULT_CONFIG
        assert service.current_phase == EvaluationPhase.IDLE
        assert service.evaluation_results == {}
        print("✅ Service initialization test passed")
        
        # Test status retrieval
        status = service.get_current_status()
        assert isinstance(status, dict)
        assert 'phase' in status
        assert 'backend_available' in status
        print("✅ Service status test passed")
        
        # Test callback setup
        progress_calls = []
        log_calls = []
        
        def progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        def log_callback(message, level):
            log_calls.append((message, level))
        
        service.set_callbacks(
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        service._update_progress(50, "Test progress")
        service._log_message("Test log", "info")
        
        assert len(progress_calls) == 1
        assert len(log_calls) == 1
        print("✅ Service callbacks test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation service test failed: {e}")
        return False

async def test_evaluation_service_async():
    """Test evaluation service async functionality."""
    print("Testing evaluation service async operations...")
    
    try:
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        
        service = EvaluationService()
        service._backend_available = False  # Force simulation mode
        
        # Test scenario evaluation
        result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet",
            selected_metrics=["map", "precision"]
        )
        
        assert result["success"] is True
        assert result["scenario"] == "position_variation"
        assert result["model"] == "cspdarknet"
        assert "results" in result
        print("✅ Scenario evaluation test passed")
        
        # Test comprehensive evaluation
        comp_result = await service.run_comprehensive_evaluation(
            scenarios=["position_variation"],
            models=["cspdarknet"],
            selected_metrics=["map"]
        )
        
        assert comp_result["success"] is True
        assert comp_result["total_tests"] == 1
        assert "results" in comp_result
        print("✅ Comprehensive evaluation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation service async test failed: {e}")
        return False

def test_evaluation_handler():
    """Test evaluation UI handler."""
    print("Testing evaluation UI handler...")
    
    try:
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        from smartcash.ui.model.evaluate.constants import (
            DEFAULT_CONFIG, AVAILABLE_SCENARIOS, AVAILABLE_MODELS, DEFAULT_ENABLED_METRICS
        )
        
        # Test initialization
        handler = EvaluationUIHandler()
        assert handler.module_name == 'evaluate'
        assert handler.parent_module == 'model'
        assert handler.current_config == DEFAULT_CONFIG
        assert handler.selected_scenarios == set(AVAILABLE_SCENARIOS)
        assert handler.selected_models == set(AVAILABLE_MODELS)
        print("✅ Handler initialization test passed")
        
        # Test handler initialize method
        handler.initialize()
        assert handler.evaluation_service is not None
        assert handler.evaluation_active is False
        print("✅ Handler initialize method test passed")
        
        # Test status retrieval
        status = handler.get_evaluation_status()
        assert isinstance(status, dict)
        assert 'evaluation_active' in status
        assert 'selected_scenarios' in status
        print("✅ Handler status test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation handler test failed: {e}")
        return False

async def test_evaluation_operations():
    """Test evaluation operations."""
    print("Testing evaluation operations...")
    
    try:
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        from smartcash.ui.model.evaluate.operations.scenario_evaluation_operation import ScenarioEvaluationOperation
        from smartcash.ui.model.evaluate.operations.comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
        
        service = EvaluationService()
        service._backend_available = False  # Force simulation mode
        
        # Test scenario operation
        scenario_op = ScenarioEvaluationOperation(service)
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet",
            "selected_metrics": ["map"]
        }
        
        result = await scenario_op.execute(config=config)
        assert result["success"] is True
        assert "result" in result
        print("✅ Scenario operation test passed")
        
        # Test comprehensive operation
        comprehensive_op = ComprehensiveEvaluationOperation(service)
        comp_config = {
            "scenarios": ["position_variation"],
            "models": ["cspdarknet"],
            "selected_metrics": ["map"]
        }
        
        comp_result = await comprehensive_op.execute(config=comp_config)
        assert comp_result["success"] is True
        assert "result" in comp_result
        print("✅ Comprehensive operation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation operations test failed: {e}")
        return False

def test_evaluation_constants():
    """Test evaluation constants."""
    print("Testing evaluation constants...")
    
    try:
        from smartcash.ui.model.evaluate.constants import (
            DEFAULT_CONFIG, EvaluationOperation, EvaluationPhase,
            TestScenario, BackboneModel, AVAILABLE_SCENARIOS,
            AVAILABLE_MODELS, AVAILABLE_METRICS, DEFAULT_ENABLED_METRICS
        )
        
        # Test enums
        assert EvaluationOperation.TEST_SCENARIO.value == "test_scenario"
        assert EvaluationPhase.IDLE.value == "idle"
        assert TestScenario.POSITION_VARIATION.value == "position_variation"
        assert BackboneModel.CSPDARKNET.value == "cspdarknet"
        print("✅ Enums test passed")
        
        # Test config structure
        assert 'evaluation' in DEFAULT_CONFIG
        assert 'inference' in DEFAULT_CONFIG
        assert 'scenarios' in DEFAULT_CONFIG['evaluation']
        print("✅ Config structure test passed")
        
        # Test lists
        assert 'position_variation' in AVAILABLE_SCENARIOS
        assert 'lighting_variation' in AVAILABLE_SCENARIOS
        assert 'cspdarknet' in AVAILABLE_MODELS
        assert 'efficientnet_b4' in AVAILABLE_MODELS
        assert len(DEFAULT_ENABLED_METRICS) > 0
        print("✅ Lists test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation constants test failed: {e}")
        return False

def test_module_imports():
    """Test all module imports work."""
    print("Testing module imports...")
    
    try:
        from smartcash.ui.model.evaluate import constants
        from smartcash.ui.model.evaluate.services import evaluation_service
        from smartcash.ui.model.evaluate.handlers import evaluation_ui_handler
        from smartcash.ui.model.evaluate.operations import (
            scenario_evaluation_operation,
            comprehensive_evaluation_operation,
            checkpoint_operation
        )
        
        # Verify key classes exist
        assert hasattr(evaluation_service, 'EvaluationService')
        assert hasattr(evaluation_ui_handler, 'EvaluationUIHandler')
        assert hasattr(scenario_evaluation_operation, 'ScenarioEvaluationOperation')
        assert hasattr(comprehensive_evaluation_operation, 'ComprehensiveEvaluationOperation')
        assert hasattr(checkpoint_operation, 'CheckpointOperation')
        print("✅ Module imports test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Module imports test failed: {e}")
        return False

async def run_async_tests():
    """Run all async tests."""
    print("\n=== Running Async Tests ===")
    
    tests = [
        test_evaluation_service_async(),
        test_evaluation_operations()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    passed = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Async test {i+1} failed with exception: {result}")
        elif result:
            passed += 1
        else:
            print(f"❌ Async test {i+1} failed")
    
    return passed, len(tests)

def run_sync_tests():
    """Run all sync tests."""
    print("=== Running Sync Tests ===")
    
    tests = [
        test_module_imports,
        test_evaluation_constants,
        test_evaluation_service,
        test_evaluation_handler
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    return passed, len(tests)

async def main():
    """Main test runner."""
    print("🚀 Starting Evaluation Module Test Suite")
    print("=" * 50)
    
    # Run sync tests
    sync_passed, sync_total = run_sync_tests()
    
    # Run async tests
    async_passed, async_total = await run_async_tests()
    
    # Summary
    total_passed = sync_passed + async_passed
    total_tests = sync_total + async_total
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results Summary")
    print(f"Sync Tests: {sync_passed}/{sync_total} passed")
    print(f"Async Tests: {async_passed}/{async_total} passed")
    print(f"Total: {total_passed}/{total_tests} passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️ {total_tests - total_passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)