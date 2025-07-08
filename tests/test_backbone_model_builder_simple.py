#!/usr/bin/env python3
"""
Simple backbone model builder test runner.
Focuses on core functionality without complex async operations.
"""

import sys
import pytest
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_backbone_service_basics():
    """Test basic backbone service functionality."""
    print("🧪 Testing Backbone Service Basics")
    print("=" * 50)
    
    try:
        from smartcash.ui.model.backbone.services.backbone_service import BackboneService
        
        # Test service creation
        service = BackboneService()
        assert service is not None
        print("✅ Service creation: PASSED")
        
        # Test backend components
        assert hasattr(service, 'backbone_factory')
        assert hasattr(service, 'model_builder')
        print("✅ Backend components: PASSED")
        
        # Test available backbones
        backbones = service.get_available_backbones()
        assert isinstance(backbones, list)
        assert len(backbones) > 0
        print(f"✅ Available backbones: {backbones}")
        
        # Test device info
        device_info = service.get_device_info()
        assert isinstance(device_info, dict)
        print(f"✅ Device info: {device_info.get('device', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backbone service test failed: {e}")
        return False


def test_model_builder_integration():
    """Test model builder integration."""
    print("\\n🏗️ Testing Model Builder Integration")
    print("=" * 50)
    
    try:
        from smartcash.ui.model.backbone.services.backbone_service import BackboneService
        from smartcash.ui.model.backbone.operations.build_operation import BuildOperation
        
        # Test service integration
        service = BackboneService()
        build_op = BuildOperation()
        
        assert hasattr(build_op, 'backbone_service')
        assert build_op.backbone_service is not None
        print("✅ Build operation integration: PASSED")
        
        # Test operations
        operations = build_op.get_operations()
        assert 'build' in operations
        assert callable(operations['build'])
        print("✅ Build operations: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Model builder integration test failed: {e}")
        return False


def test_backbone_factory_integration():
    """Test backbone factory integration."""
    print("\\n🏭 Testing Backbone Factory Integration")
    print("=" * 50)
    
    try:
        from smartcash.ui.model.backbone.services.backbone_service import BackboneService
        
        service = BackboneService()
        factory = service.backbone_factory
        
        # Test factory methods
        assert hasattr(factory, 'create_backbone')
        assert hasattr(factory, 'list_available_backbones')
        print("✅ Factory methods: PASSED")
        
        # Test listing backbones
        backbones = factory.list_available_backbones()
        assert isinstance(backbones, list)
        assert len(backbones) > 0
        print(f"✅ Factory backbones: {backbones}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backbone factory test failed: {e}")
        return False


def test_configuration_validation():
    """Test configuration validation."""
    print("\\n⚙️ Testing Configuration Validation")
    print("=" * 50)
    
    try:
        from smartcash.ui.model.backbone.services.backbone_service import BackboneService
        
        service = BackboneService()
        
        # Test valid configuration
        valid_config = {
            'backbone_type': 'efficientnet_b4',
            'pretrained': True,
            'feature_optimization': False
        }
        
        # Note: We'll test the internal validation method rather than async
        result = service._validate_config_format(valid_config)
        assert result['valid'] is True
        print("✅ Valid config validation: PASSED")
        
        # Test invalid configuration
        invalid_config = {
            'backbone_type': 'invalid_backbone'
        }
        
        result = service._validate_config_format(invalid_config)
        assert result['valid'] is False
        assert len(result['errors']) > 0
        print("✅ Invalid config validation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False


def run_simple_model_builder_tests():
    """Run simple model builder tests."""
    print("🚀 Running Simple Model Builder Tests")
    print("=" * 70)
    
    tests = [
        ("Backbone Service Basics", test_backbone_service_basics),
        ("Model Builder Integration", test_model_builder_integration),
        ("Backbone Factory Integration", test_backbone_factory_integration),
        ("Configuration Validation", test_configuration_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 70)
    print("🏁 Simple Model Builder Test Results")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\\n📊 Results: {passed}/{total} tests passed")
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\\n🎉 ALL SIMPLE MODEL BUILDER TESTS PASSED!")
        print("✅ Core model builder functionality is working")
        return True
    else:
        print(f"\\n⚠️ {total - passed} tests failed")
        print("🔧 Review failed tests")
        return False


if __name__ == "__main__":
    success = run_simple_model_builder_tests()
    
    if success:
        print("\\n✅ Model builder integration confirmed working!")
        sys.exit(0)
    else:
        print("\\n❌ Model builder integration has issues")
        sys.exit(1)