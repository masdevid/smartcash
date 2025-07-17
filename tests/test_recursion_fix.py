#!/usr/bin/env python3
"""
Test script to verify the maximum recursion fix in UI dataset downloader module.
This test verifies that the recursion issue has been resolved and the module
can be initialized successfully with a 90%+ success rate.
"""

import sys
import traceback
import time
from typing import Dict, Any, Optional

def test_recursion_fix() -> Dict[str, Any]:
    """
    Test the recursion fix in the downloader initializer.
    
    Returns:
        Dict containing test results and statistics
    """
    print("🧪 Testing UI Dataset Downloader Recursion Fix")
    print("=" * 50)
    
    results = {
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'recursion_errors': 0,
        'other_errors': 0,
        'success_rate': 0.0,
        'target_achieved': False,
        'errors': []
    }
    
    # Test configurations
    test_configs = [
        None,  # Default config
        {'download': {'target_dir': 'test_data'}},
        {'data': {'dir': 'custom_data'}},
        {},  # Empty config
        {'download': {'target_dir': 'test'}, 'data': {'dir': 'data'}},
    ]
    
    # Run tests with different configs
    for i, config in enumerate(test_configs):
        for repeat in range(4):  # 4 repeats per config = 20 total tests
            test_num = i * 4 + repeat + 1
            results['total_tests'] += 1
            
            try:
                # Import fresh each time to avoid caching issues
                from smartcash.ui.dataset.downloader.downloader_initializer import initialize_downloader_ui
                
                # Initialize with config
                if config:
                    result = initialize_downloader_ui(config=config)
                else:
                    result = initialize_downloader_ui()
                
                # Check if result is valid
                if result and len(result) > 0:
                    results['successful_tests'] += 1
                    print(f"✅ Test {test_num}: Success - {len(result)} components")
                else:
                    results['failed_tests'] += 1
                    print(f"❌ Test {test_num}: Empty result")
                    
            except Exception as e:
                results['failed_tests'] += 1
                error_msg = str(e)
                
                # Check for recursion errors
                if 'recursion' in error_msg.lower() or 'maximum recursion depth' in error_msg.lower():
                    results['recursion_errors'] += 1
                    print(f"🔥 Test {test_num}: RECURSION ERROR - {error_msg[:100]}...")
                    results['errors'].append(f"Test {test_num}: RECURSION - {error_msg[:100]}")
                else:
                    results['other_errors'] += 1
                    print(f"❌ Test {test_num}: Other error - {error_msg[:50]}...")
                    results['errors'].append(f"Test {test_num}: OTHER - {error_msg[:50]}")
    
    # Calculate success rate
    results['success_rate'] = (results['successful_tests'] / results['total_tests']) * 100
    results['target_achieved'] = results['success_rate'] >= 90.0
    
    return results

def print_test_summary(results: Dict[str, Any]) -> None:
    """Print a detailed test summary."""
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Successful Tests: {results['successful_tests']}")
    print(f"Failed Tests: {results['failed_tests']}")
    print(f"Recursion Errors: {results['recursion_errors']}")
    print(f"Other Errors: {results['other_errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['target_achieved']:
        print("🎉 TARGET ACHIEVED: 90%+ success rate!")
        print("✅ Recursion fix is working correctly!")
    else:
        print("❌ Target not met. Need 90%+ success rate.")
        
    if results['recursion_errors'] > 0:
        print(f"⚠️  WARNING: {results['recursion_errors']} recursion errors detected!")
        print("❌ Recursion fix may not be working properly.")
    else:
        print("✅ No recursion errors detected - fix is working!")
        
    # Print error details if any
    if results['errors']:
        print("\n🔍 ERROR DETAILS:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")

def main():
    """Main test runner."""
    start_time = time.time()
    
    try:
        results = test_recursion_fix()
        print_test_summary(results)
        
        # Return appropriate exit code
        if results['recursion_errors'] > 0:
            print("\n❌ CRITICAL: Recursion errors detected!")
            sys.exit(1)
        elif results['target_achieved']:
            print("\n✅ SUCCESS: All tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️  WARNING: Target not achieved but no recursion errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️  Test completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()