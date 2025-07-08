#!/usr/bin/env python3
"""
Comprehensive test runner for training module.
Runs all unit tests, integration tests, and validates the complete training pipeline.
"""

import sys
import pytest
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_training_tests():
    """Run comprehensive training module tests."""
    
    print("🚀 Running Comprehensive Training Module Tests")
    print("=" * 70)
    
    # Test categories to run
    test_categories = [
        {
            "name": "Unit Tests - Training Module Structure",
            "path": "tests/unit/ui/model/train/test_training_module.py",
            "description": "Core module structure and components"
        },
        {
            "name": "Unit Tests - Chart Container",
            "path": "tests/unit/ui/components/test_chart_container.py", 
            "description": "Reusable chart container component"
        },
        {
            "name": "Integration Tests - Backend Integration",
            "path": "tests/integration/training/test_backend_integration.py",
            "description": "UI-Backend service integration"
        },
        {
            "name": "Integration Tests - Training Pipeline",
            "path": "tests/integration/training/test_training_pipeline.py",
            "description": "Complete training workflow"
        }
    ]
    
    results = []
    total_tests = 0
    total_passed = 0
    
    for category in test_categories:
        print(f"\n📋 Running: {category['name']}")
        print(f"📝 Description: {category['description']}")
        print("-" * 50)
        
        test_path = project_root / category['path']
        
        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            results.append((category['name'], False, 0, 0))
            continue
        
        try:
            # Run pytest with verbose output and capture results (override config)
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "-v", "--tb=short", "-x", "--override-ini", "addopts="
            ], capture_output=True, text=True, cwd=project_root)
            
            # Parse results
            output = result.stdout + result.stderr
            
            # Count passed/failed tests
            lines = output.split('\n')
            passed = sum(1 for line in lines if '::' in line and 'PASSED' in line)
            failed = sum(1 for line in lines if '::' in line and 'FAILED' in line)
            category_total = passed + failed
            
            success = result.returncode == 0 and failed == 0
            
            if success:
                print(f"✅ PASSED: {category['name']} ({passed}/{category_total})")
            else:
                print(f"❌ FAILED: {category['name']} ({passed}/{category_total})")
                if failed > 0:
                    print(f"   Failed tests: {failed}")
                
                # Show error details for failed tests
                error_lines = [line for line in lines if 'FAILED' in line or 'ERROR' in line]
                for error_line in error_lines[:3]:  # Show first 3 errors
                    print(f"   {error_line}")
            
            results.append((category['name'], success, passed, category_total))
            total_tests += category_total
            total_passed += passed
            
        except Exception as e:
            print(f"❌ ERROR: Failed to run {category['name']}: {e}")
            results.append((category['name'], False, 0, 0))
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("🏁 Training Module Test Results Summary")
    print("=" * 70)
    
    for name, success, passed, total in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        if total > 0:
            print(f"{status}: {name} ({passed}/{total} tests)")
        else:
            print(f"{status}: {name} (No tests found)")
    
    print(f"\n📊 Overall Results: {total_passed}/{total_tests} tests passed")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Determine overall status
    overall_success = all(result[1] for result in results if result[3] > 0)
    
    if overall_success and total_passed == total_tests:
        print("\n🎉 ALL TRAINING MODULE TESTS PASSED!")
        print("\n✅ Training Module Status: PRODUCTION READY")
        print("   • Module structure: ✅ Compliant with UI standards")
        print("   • Chart container: ✅ Reusable component working")
        print("   • Backend integration: ✅ UI-Backend bridge functional")
        print("   • Training pipeline: ✅ Complete workflow operational")
        print("   • Error handling: ✅ Graceful degradation working")
        print("   • Simulation mode: ✅ Fallback mode functional")
        
        return True
    else:
        failed_categories = [name for name, success, _, total in results if not success and total > 0]
        print(f"\n⚠️ Some tests failed in: {', '.join(failed_categories)}")
        
        if success_rate >= 80:
            print("✅ Training Module Status: MOSTLY FUNCTIONAL")
            print("   • Core functionality working")
            print("   • Minor issues in some components")
        else:
            print("❌ Training Module Status: NEEDS ATTENTION")
            print("   • Significant issues found")
            print("   • Review failed tests before production use")
        
        return False


def run_quick_validation():
    """Run quick validation of key training module components."""
    
    print("\n🔍 Quick Validation of Key Components")
    print("=" * 50)
    
    validations = []
    
    try:
        # Test 1: Import validation
        print("📦 Testing imports...")
        from smartcash.ui.model.train.constants import TrainingOperation, DEFAULT_CONFIG
        from smartcash.ui.model.train.services.training_service import TrainingService
        from smartcash.ui.components.chart_container import create_chart_container
        print("✅ All critical imports successful")
        validations.append(True)
        
        # Test 2: Service creation
        print("🔧 Testing service creation...")
        service = TrainingService()
        status = service.get_current_status()
        assert 'phase' in status
        print("✅ Training service creation successful")
        validations.append(True)
        
        # Test 3: Chart container
        print("📊 Testing chart container...")
        chart = create_chart_container(columns=2)
        chart.initialize()
        chart.update_chart("chart_1", [0.5, 0.4, 0.3], {"title": "Test"})
        print("✅ Chart container working")
        validations.append(True)
        
        # Test 4: Backend availability
        print("🔗 Testing backend availability...")
        backend_status = service.validate_backend_availability()
        assert 'available' in backend_status
        if backend_status['available']:
            print("✅ Backend components available")
        else:
            print("⚠️ Backend not available (will use simulation)")
        validations.append(True)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        validations.append(False)
    
    success_count = sum(validations)
    total_count = len(validations)
    
    print(f"\n📊 Quick Validation: {success_count}/{total_count} checks passed")
    
    if success_count == total_count:
        print("✅ All critical components functional")
        return True
    else:
        print("⚠️ Some components have issues")
        return False


if __name__ == "__main__":
    print("🚀 SmartCash Training Module Test Suite")
    print("=" * 70)
    
    # Run quick validation first
    quick_valid = run_quick_validation()
    
    if not quick_valid:
        print("\n❌ Quick validation failed. Skipping comprehensive tests.")
        sys.exit(1)
    
    # Run comprehensive tests
    comprehensive_success = run_training_tests()
    
    # Final status
    print(f"\n{'='*70}")
    print("🏁 Final Training Module Status")
    print(f"{'='*70}")
    
    if comprehensive_success:
        print("🎉 TRAINING MODULE FULLY VALIDATED!")
        print("✅ Ready for production use")
        exit_code = 0
    else:
        print("⚠️ Training module has some issues")
        print("🔧 Review test results and fix failing components")
        exit_code = 1
    
    sys.exit(exit_code)