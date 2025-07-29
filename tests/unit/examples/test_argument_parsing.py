#!/usr/bin/env python3
"""
Simple test to verify argument parsing for optimizer and resume functionality.
This test focuses on validating that the arguments are correctly parsed and 
processed without running actual training.
"""

# Fix OpenMP duplicate library issue
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from examples.training_args_helper import create_training_arg_parser, get_training_kwargs, print_training_configuration


def test_optimizer_arguments():
    """Test optimizer argument parsing"""
    print("🧪 Testing optimizer argument parsing...")
    
    parser = create_training_arg_parser("Test Parser")
    
    test_cases = [
        # Test case: (description, args_list, expected_values)
        (
            "Default optimizer arguments", 
            [], 
            {"optimizer": "adamw", "scheduler": "cosine", "weight_decay": 1e-2, "cosine_eta_min": 1e-6}
        ),
        (
            "Custom AdamW with different weight decay",
            ["--optimizer", "adamw", "--weight-decay", "5e-3", "--cosine-eta-min", "1e-7"],
            {"optimizer": "adamw", "scheduler": "cosine", "weight_decay": 5e-3, "cosine_eta_min": 1e-7}
        ),
        (
            "SGD with step scheduler",
            ["--optimizer", "sgd", "--scheduler", "step", "--weight-decay", "1e-3"],
            {"optimizer": "sgd", "scheduler": "step", "weight_decay": 1e-3}
        ),
        (
            "Adam with plateau scheduler", 
            ["--optimizer", "adam", "--scheduler", "plateau"],
            {"optimizer": "adam", "scheduler": "plateau", "weight_decay": 1e-2}
        )
    ]
    
    passed = 0
    total = len(test_cases)
    
    for description, args_list, expected in test_cases:
        try:
            args = parser.parse_args(args_list + ["--phase1-epochs", "1"])  # Add required arg
            kwargs = get_training_kwargs(args)
            
            # Check if all expected values match
            all_match = True
            for key, expected_value in expected.items():
                actual_value = kwargs.get(key)
                if actual_value != expected_value:
                    print(f"   ❌ {description}: {key} expected {expected_value}, got {actual_value}")
                    all_match = False
            
            if all_match:
                print(f"   ✅ {description}")
                passed += 1
            
        except Exception as e:
            print(f"   ❌ {description}: Exception - {str(e)}")
    
    print(f"📊 Optimizer argument tests: {passed}/{total} passed")
    return passed == total


def test_resume_arguments():
    """Test resume argument parsing"""
    print("\n🧪 Testing resume argument parsing...")
    
    parser = create_training_arg_parser("Test Parser")
    
    test_cases = [
        (
            "No resume arguments",
            [],
            {"resume_checkpoint": None, "resume_optimizer_state": False, "resume_scheduler_state": False, "resume_epoch": None}
        ),
        (
            "Basic resume with checkpoint",
            ["--resume", "path/to/checkpoint.pt"],
            {"resume_checkpoint": "path/to/checkpoint.pt", "resume_optimizer_state": False, "resume_scheduler_state": False, "resume_epoch": None}
        ),
        (
            "Resume with optimizer and scheduler state",
            ["--resume", "checkpoint.pt", "--resume-optimizer", "--resume-scheduler"],
            {"resume_checkpoint": "checkpoint.pt", "resume_optimizer_state": True, "resume_scheduler_state": True, "resume_epoch": None}
        ),
        (
            "Resume with epoch override",
            ["--resume", "checkpoint.pt", "--resume-epoch", "10"],
            {"resume_checkpoint": "checkpoint.pt", "resume_optimizer_state": False, "resume_scheduler_state": False, "resume_epoch": 10}
        ),
        (
            "Complete resume configuration",
            ["--resume", "full_checkpoint.pt", "--resume-optimizer", "--resume-scheduler", "--resume-epoch", "5"],
            {"resume_checkpoint": "full_checkpoint.pt", "resume_optimizer_state": True, "resume_scheduler_state": True, "resume_epoch": 5}
        )
    ]
    
    passed = 0
    total = len(test_cases)
    
    for description, args_list, expected in test_cases:
        try:
            args = parser.parse_args(args_list + ["--phase1-epochs", "1"])  # Add required arg
            kwargs = get_training_kwargs(args)
            
            # Check if all expected values match
            all_match = True
            for key, expected_value in expected.items():
                actual_value = kwargs.get(key)
                if actual_value != expected_value:
                    print(f"   ❌ {description}: {key} expected {expected_value}, got {actual_value}")
                    all_match = False
            
            if all_match:
                print(f"   ✅ {description}")
                passed += 1
                
        except Exception as e:
            print(f"   ❌ {description}: Exception - {str(e)}")
    
    print(f"📊 Resume argument tests: {passed}/{total} passed")
    return passed == total


def test_configuration_display():
    """Test configuration display functionality"""
    print("\n🧪 Testing configuration display...")
    
    parser = create_training_arg_parser("Test Parser")
    
    try:
        # Test with a comprehensive set of arguments
        args = parser.parse_args([
            "--backbone", "cspdarknet",
            "--optimizer", "adamw",
            "--scheduler", "cosine", 
            "--weight-decay", "1e-2",
            "--cosine-eta-min", "1e-6",
            "--resume", "test_checkpoint.pt",
            "--resume-optimizer",
            "--resume-scheduler",
            "--resume-epoch", "3",
            "--phase1-epochs", "2",
            "--phase2-epochs", "1",
            "--verbose"
        ])
        
        print("   📋 Configuration display test:")
        print_training_configuration(args)
        
        print("   ✅ Configuration display completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration display failed: {str(e)}")
        return False


def test_invalid_arguments():
    """Test invalid argument handling"""
    print("\n🧪 Testing invalid argument handling...")
    
    parser = create_training_arg_parser("Test Parser")
    
    invalid_test_cases = [
        ("Invalid optimizer", ["--optimizer", "invalid_opt", "--phase1-epochs", "1"]),
        ("Invalid scheduler", ["--scheduler", "invalid_sched", "--phase1-epochs", "1"])
    ]
    
    passed = 0
    total = len(invalid_test_cases)
    
    for description, args_list in invalid_test_cases:
        try:
            args = parser.parse_args(args_list)
            print(f"   ❌ {description}: Should have failed but didn't")
        except SystemExit:
            # This is expected for invalid arguments
            print(f"   ✅ {description}: Correctly rejected invalid argument")
            passed += 1
        except Exception as e:
            print(f"   ❌ {description}: Unexpected exception - {str(e)}")
    
    print(f"📊 Invalid argument tests: {passed}/{total} passed")
    return passed == total


def main():
    """Run all argument parsing tests"""
    print("🚀 ARGUMENT PARSING TESTS FOR ENHANCED CALLBACK TRAINING")
    print("=" * 80)
    
    test_functions = [
        ("Optimizer Arguments", test_optimizer_arguments),
        ("Resume Arguments", test_resume_arguments),
        ("Configuration Display", test_configuration_display),
        ("Invalid Arguments", test_invalid_arguments)
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"💥 {test_name} crashed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(f"🎯 Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL ARGUMENT PARSING TESTS PASSED!")
        print("✅ Optimizer and resume arguments are correctly implemented")
        return True
    else:
        print("⚠️  Some argument parsing tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)