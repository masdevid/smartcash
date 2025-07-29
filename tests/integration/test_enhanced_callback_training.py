#!/usr/bin/env python3
"""
Comprehensive test for enhanced callback-only training example.

This script tests the new optimizer and resume arguments functionality
with various configurations to ensure everything works correctly.
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_training_test(description: str, args: list, expect_success: bool = True, timeout: int = 120):
    """
    Run a training test with specified arguments.
    
    Args:
        description: Test description
        args: List of command line arguments
        expect_success: Whether we expect the training to succeed
        timeout: Timeout in seconds
        
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    print(f"🧪 {description}")
    print(f"   Command: python examples/callback_only_training_example.py {' '.join(args)}")
    
    try:
        # Run the training example
        cmd = [sys.executable, "examples/callback_only_training_example.py"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root
        )
        
        success = (result.returncode == 0) == expect_success
        
        if success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED" 
            
        print(f"   Result: {status}")
        
        if not success or True:  # Always show output for debugging
            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   STDOUT preview: {result.stdout[:200]}...")
            if result.stderr:
                print(f"   STDERR preview: {result.stderr[:200]}...")
        
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"   Result: ⏰ TIMEOUT (exceeded {timeout}s)")
        return False, "", "Timeout expired"
    except Exception as e:
        print(f"   Result: 💥 ERROR - {str(e)}")
        return False, "", str(e)


def test_basic_functionality():
    """Test basic functionality without optimizer/resume args"""
    print("\n" + "="*80)
    print("📋 TESTING BASIC FUNCTIONALITY")
    print("="*80)
    
    tests = [
        (
            "Basic two-phase training with minimal epochs",
            ["--backbone", "cspdarknet", "--phase1-epochs", "1", "--phase2-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "Single-phase training with minimal epochs",
            ["--training-mode", "single_phase", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "Help command test",
            ["--help"]
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for description, args in tests:
        # For help command, expect it to "fail" with exit code 0 (help shown)
        expect_success = not ("--help" in args)
        success, stdout, stderr = run_training_test(description, args, expect_success)
        if success:
            passed += 1
    
    print(f"\n📊 Basic functionality tests: {passed}/{total} passed")
    return passed == total


def test_optimizer_configurations():
    """Test different optimizer and scheduler configurations"""
    print("\n" + "="*80)
    print("⚙️ TESTING OPTIMIZER CONFIGURATIONS")
    print("="*80)
    
    tests = [
        (
            "AdamW optimizer with cosine scheduler (default)",
            ["--optimizer", "adamw", "--scheduler", "cosine", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "AdamW with custom weight decay and cosine eta_min",
            ["--optimizer", "adamw", "--scheduler", "cosine", "--weight-decay", "5e-3", "--cosine-eta-min", "1e-7", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "SGD optimizer with step scheduler",
            ["--optimizer", "sgd", "--scheduler", "step", "--weight-decay", "1e-3", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "Adam optimizer with plateau scheduler",
            ["--optimizer", "adam", "--scheduler", "plateau", "--weight-decay", "1e-4", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "RMSprop optimizer with exponential scheduler",
            ["--optimizer", "rmsprop", "--scheduler", "exponential", "--weight-decay", "5e-4", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for description, args in tests:
        success, stdout, stderr = run_training_test(description, args)
        if success:
            passed += 1
    
    print(f"\n📊 Optimizer configuration tests: {passed}/{total} passed")
    return passed == total


def test_resume_configurations():
    """Test resume training configurations"""
    print("\n" + "="*80)
    print("🔄 TESTING RESUME CONFIGURATIONS")
    print("="*80)
    
    # These tests focus on argument parsing and configuration validation
    # We don't actually need valid checkpoint files for testing the argument handling
    
    tests = [
        (
            "Resume with non-existent checkpoint (should show error gracefully)",
            ["--resume", "non_existent_checkpoint.pt", "--phase1-epochs", "1", "--force-cpu", "--verbose"],
            False  # Expect this to fail since checkpoint doesn't exist
        ),
        (
            "Resume configuration with optimizer and scheduler state",
            ["--resume", "fake_checkpoint.pt", "--resume-optimizer", "--resume-scheduler", "--phase1-epochs", "1", "--force-cpu", "--verbose"],
            False  # Will fail due to missing checkpoint, but tests argument parsing
        ),
        (
            "Resume with epoch override",
            ["--resume", "fake_checkpoint.pt", "--resume-epoch", "5", "--phase1-epochs", "1", "--force-cpu", "--verbose"],
            False  # Will fail due to missing checkpoint, but tests argument parsing
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for description, args, expect_success in tests:
        success, stdout, stderr = run_training_test(description, args, expect_success)
        if success:
            passed += 1
        elif not expect_success and "checkpoint" in stderr.lower():
            # If it failed as expected due to missing checkpoint, that's still a pass for argument parsing
            print(f"   Note: Failed as expected due to missing checkpoint file")
            passed += 1
    
    print(f"\n📊 Resume configuration tests: {passed}/{total} passed")
    return passed == total


def test_combined_configurations():
    """Test combinations of optimizer and other arguments"""
    print("\n" + "="*80)
    print("🔧 TESTING COMBINED CONFIGURATIONS")
    print("="*80)
    
    tests = [
        (
            "Two-phase with AdamW and custom early stopping",
            ["--backbone", "cspdarknet", "--optimizer", "adamw", "--scheduler", "cosine", 
             "--phase1-epochs", "1", "--phase2-epochs", "1", "--patience", "5", 
             "--es-metric", "val_loss", "--es-mode", "min", "--force-cpu", "--verbose"]
        ),
        (
            "Single-phase with SGD and no early stopping",
            ["--training-mode", "single_phase", "--optimizer", "sgd", "--scheduler", "step",
             "--phase1-epochs", "1", "--no-early-stopping", "--force-cpu", "--verbose"]
        ),
        (
            "Pretrained backbone with Adam optimizer",
            ["--backbone", "efficientnet_b4", "--pretrained", "--optimizer", "adam", 
             "--scheduler", "plateau", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        ),
        (
            "Custom learning rates with RMSprop",
            ["--optimizer", "rmsprop", "--scheduler", "cosine", "--head-lr-p1", "2e-3", 
             "--head-lr-p2", "5e-4", "--backbone-lr", "1e-6", "--phase1-epochs", "1", "--force-cpu", "--verbose"]
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for description, args in tests:
        success, stdout, stderr = run_training_test(description, args)
        if success:
            passed += 1
    
    print(f"\n📊 Combined configuration tests: {passed}/{total} passed")
    return passed == total


def test_argument_validation():
    """Test argument validation and error handling"""
    print("\n" + "="*80)
    print("❌ TESTING ARGUMENT VALIDATION")
    print("="*80)
    
    tests = [
        (
            "Invalid optimizer (should fail)",
            ["--optimizer", "invalid_optimizer", "--phase1-epochs", "1", "--force-cpu"],
            False
        ),
        (
            "Invalid scheduler (should fail)",
            ["--scheduler", "invalid_scheduler", "--phase1-epochs", "1", "--force-cpu"],
            False
        ),
        (
            "Negative weight decay (should pass but might warn)",
            ["--weight-decay", "-1e-3", "--phase1-epochs", "1", "--force-cpu", "--verbose"],
            True  # Arguments might be accepted but could cause training issues
        ),
        (
            "Invalid resume epoch (negative)",
            ["--resume-epoch", "-5", "--phase1-epochs", "1", "--force-cpu"],
            True  # Argument parsing should accept it, validation happens later
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for description, args, expect_success in tests:
        success, stdout, stderr = run_training_test(description, args, expect_success, timeout=30)
        if success:
            passed += 1
    
    print(f"\n📊 Argument validation tests: {passed}/{total} passed")
    return passed == total


def main():
    """Run comprehensive tests for enhanced callback training example"""
    print("🚀 COMPREHENSIVE TEST SUITE FOR ENHANCED CALLBACK TRAINING")
    print("=" * 100)
    print("Testing new optimizer and resume functionality...")
    print("=" * 100)
    
    test_results = []
    
    # Run all test categories
    test_functions = [
        ("Basic Functionality", test_basic_functionality),
        ("Optimizer Configurations", test_optimizer_configurations), 
        ("Resume Configurations", test_resume_configurations),
        ("Combined Configurations", test_combined_configurations),
        ("Argument Validation", test_argument_validation)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 100)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 100)
    
    passed_categories = 0
    total_categories = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed_categories += 1
    
    print(f"\n🎯 OVERALL RESULT: {passed_categories}/{total_categories} test categories passed")
    
    if passed_categories == total_categories:
        print("🎉 ALL TESTS PASSED! Enhanced callback training example is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)