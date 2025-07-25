#!/usr/bin/env python3
"""
Script to run unified training pipeline tests with comprehensive coverage.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all training pipeline tests."""
    
    # Activate virtual environment and run tests
    venv_python = Path("venv-test/bin/python")
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please create venv-test first.")
        return False
    
    test_files = [
        "tests/unit/model/training/test_unified_training_pipeline.py",
        "tests/integration/test_unified_training_pipeline_integration.py", 
        "tests/unit/model/training/test_unified_training_pipeline_resume.py",
        "tests/unit/model/training/test_unified_training_pipeline_error_handling.py",
        "tests/unit/model/training/test_unified_training_pipeline_callbacks.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠️ Test file not found: {test_file}")
            continue
            
        print(f"\n🧪 Running tests in {test_file}")
        print("=" * 60)
        
        cmd = [
            str(venv_python), "-m", "pytest", 
            test_file, 
            "-v", "--tb=short", 
            "--maxfail=5"  # Stop after 5 failures
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ All tests passed in {test_file}")
                # Show just the summary
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and ('warning' in line or 'failed' in line or '====' in line):
                        print(line)
            else:
                print(f"❌ Some tests failed in {test_file}")
                all_passed = False
                
                # Show failures and summary
                lines = result.stdout.split('\n')
                in_failure = False
                for line in lines:
                    if 'FAILURES' in line or 'FAILED' in line:
                        in_failure = True
                    if in_failure or 'passed' in line or 'failed' in line or '====' in line:
                        if line.strip():
                            print(line)
                            
                # Also show stderr if there are errors
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ Tests timed out in {test_file}")
            all_passed = False
        except Exception as e:
            print(f"💥 Error running tests in {test_file}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All training pipeline tests passed!")
    else:
        print("⚠️ Some tests failed. See details above.")
        
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)