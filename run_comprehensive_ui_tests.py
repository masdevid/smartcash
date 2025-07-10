#!/usr/bin/env python3
"""
SmartCash UI Module Comprehensive Test Suite

This script runs validation tests on all UI modules and provides a comprehensive
report of compliance levels, errors, and recommendations for improvement.

Usage:
    python run_comprehensive_ui_tests.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from validate_ui_module import UIModuleValidator, print_validation_results

def find_all_ui_modules() -> List[str]:
    """Find all UI module files in the project."""
    base_path = Path(__file__).parent / "smartcash" / "ui"
    ui_files = []
    
    # Find all *_ui.py files, excluding configs directories
    for file_path in base_path.rglob("*_ui.py"):
        # Skip files in configs directories
        if "configs" not in str(file_path):
            ui_files.append(str(file_path))
    
    return sorted(ui_files)

def get_module_info(file_path: str) -> Dict[str, str]:
    """Extract module information from file path."""
    path_parts = Path(file_path).parts
    
    # Find the ui index
    ui_index = None
    for i, part in enumerate(path_parts):
        if part == "ui":
            ui_index = i
            break
    
    if ui_index is None:
        return {"parent_module": "unknown", "module_name": "unknown"}
    
    # Get parent module and module name
    if ui_index + 1 < len(path_parts):
        parent_module = path_parts[ui_index + 1]
    else:
        parent_module = "unknown"
    
    if ui_index + 2 < len(path_parts):
        module_name = path_parts[ui_index + 2]
    else:
        module_name = "unknown"
    
    return {
        "parent_module": parent_module,
        "module_name": module_name,
        "file_name": Path(file_path).name
    }

def run_validation_tests() -> Dict[str, Any]:
    """Run validation tests on all UI modules."""
    ui_modules = find_all_ui_modules()
    results = {
        "total_modules": len(ui_modules),
        "modules": {},
        "summary": {
            "setup": {"total": 0, "passed": 0, "scores": []},
            "dataset": {"total": 0, "passed": 0, "scores": []},
            "model": {"total": 0, "passed": 0, "scores": []},
            "overall": {"total": 0, "passed": 0, "scores": []}
        }
    }
    
    print(f"Found {len(ui_modules)} UI modules to validate\n")
    
    for file_path in ui_modules:
        print(f"Validating {Path(file_path).name}...")
        
        # Get module info
        module_info = get_module_info(file_path)
        parent_module = module_info["parent_module"]
        
        # Run validation
        validator = UIModuleValidator(file_path)
        validation_result = validator.validate()
        
        # Store results
        module_key = f"{parent_module}/{module_info['module_name']}"
        results["modules"][module_key] = {
            "file_path": file_path,
            "module_info": module_info,
            "validation": validation_result
        }
        
        # Update summary statistics
        if parent_module in results["summary"]:
            results["summary"][parent_module]["total"] += 1
            results["summary"][parent_module]["scores"].append(validation_result["score"])
            if validation_result["valid"]:
                results["summary"][parent_module]["passed"] += 1
        
        # Update overall statistics
        results["summary"]["overall"]["total"] += 1
        results["summary"]["overall"]["scores"].append(validation_result["score"])
        if validation_result["valid"]:
            results["summary"]["overall"]["passed"] += 1
        
        # Print quick status
        status = "✅ PASS" if validation_result["valid"] else "❌ FAIL"
        print(f"  {status} - Score: {validation_result['score']:.1f}%")
    
    return results

def print_comprehensive_report(results: Dict[str, Any]) -> None:
    """Print a comprehensive test report."""
    print("\n" + "="*80)
    print("SMARTCASH UI MODULE COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Overall Summary
    overall = results["summary"]["overall"]
    avg_score = sum(overall["scores"]) / len(overall["scores"]) if overall["scores"] else 0
    compliance_rate = (overall["passed"] / overall["total"]) * 100 if overall["total"] > 0 else 0
    
    print(f"\n📊 OVERALL SUMMARY")
    print(f"Total Modules: {overall['total']}")
    print(f"Modules Passing: {overall['passed']}")
    print(f"Compliance Rate: {compliance_rate:.1f}%")
    print(f"Average Score: {avg_score:.1f}%")
    
    # Module Group Summary
    print(f"\n📋 MODULE GROUP SUMMARY")
    for group_name in ["setup", "dataset", "model"]:
        if group_name in results["summary"]:
            group = results["summary"][group_name]
            if group["total"] > 0:
                group_avg = sum(group["scores"]) / len(group["scores"])
                group_compliance = (group["passed"] / group["total"]) * 100
                print(f"  {group_name.title()}: {group['passed']}/{group['total']} passed ({group_compliance:.1f}%) - Avg: {group_avg:.1f}%")
    
    # Detailed Results by Module
    print(f"\n📝 DETAILED RESULTS BY MODULE")
    
    for group_name in ["setup", "dataset", "model"]:
        group_modules = [(k, v) for k, v in results["modules"].items() if v["module_info"]["parent_module"] == group_name]
        
        if group_modules:
            print(f"\n🔧 {group_name.upper()} MODULES:")
            
            for module_key, module_data in sorted(group_modules):
                validation = module_data["validation"]
                module_info = module_data["module_info"]
                
                status = "✅ PASS" if validation["valid"] else "❌ FAIL"
                print(f"  {module_info['file_name']} - {status} ({validation['score']:.1f}%)")
                
                # Show errors if any
                if validation["errors"]:
                    print(f"    🔴 Errors: {len(validation['errors'])}")
                    for error in validation["errors"][:2]:  # Show first 2 errors
                        print(f"      • {error}")
                    if len(validation["errors"]) > 2:
                        print(f"      • ... and {len(validation['errors']) - 2} more")
                
                # Show warnings if any
                if validation["warnings"]:
                    print(f"    🟡 Warnings: {len(validation['warnings'])}")
                    for warning in validation["warnings"][:2]:  # Show first 2 warnings
                        print(f"      • {warning}")
                    if len(validation["warnings"]) > 2:
                        print(f"      • ... and {len(validation['warnings']) - 2} more")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    
    # Find modules that need attention
    needs_attention = []
    minor_fixes = []
    compliant = []
    
    for module_key, module_data in results["modules"].items():
        score = module_data["validation"]["score"]
        if score < 50:
            needs_attention.append((module_key, score))
        elif score < 90:
            minor_fixes.append((module_key, score))
        else:
            compliant.append((module_key, score))
    
    if needs_attention:
        print(f"\n🚨 PRIORITY FIXES NEEDED ({len(needs_attention)} modules):")
        for module_key, score in sorted(needs_attention, key=lambda x: x[1]):
            print(f"  • {module_key} ({score:.1f}%) - Needs major restructuring")
    
    if minor_fixes:
        print(f"\n🔧 MINOR FIXES NEEDED ({len(minor_fixes)} modules):")
        for module_key, score in sorted(minor_fixes, key=lambda x: x[1]):
            print(f"  • {module_key} ({score:.1f}%) - Address warnings and missing features")
    
    if compliant:
        print(f"\n✅ COMPLIANT MODULES ({len(compliant)} modules):")
        for module_key, score in sorted(compliant, key=lambda x: x[1], reverse=True):
            print(f"  • {module_key} ({score:.1f}%) - Well structured")
    
    # Next Steps
    print(f"\n🎯 NEXT STEPS")
    print("1. Focus on modules with scores < 50% first")
    print("2. Add missing constants (UI_CONFIG, BUTTON_CONFIG) to modules with warnings")
    print("3. Add error handling decorators to main functions")
    print("4. Implement missing helper functions (_create_module_*)")
    print("5. Ensure all modules follow standard container order")
    print("6. Add **kwargs parameter to function signatures")
    
    print("\n" + "="*80)

def save_results_to_file(results: Dict[str, Any], output_file: str = "ui_test_results.json") -> None:
    """Save test results to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

def main():
    """Main test runner function."""
    print("🧪 SmartCash UI Module Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        # Run validation tests
        results = run_validation_tests()
        
        # Print comprehensive report
        print_comprehensive_report(results)
        
        # Save results to file
        save_results_to_file(results)
        
        # Exit with appropriate code
        overall_passed = results["summary"]["overall"]["passed"]
        overall_total = results["summary"]["overall"]["total"]
        
        if overall_passed == overall_total:
            print("🎉 All modules are compliant!")
            sys.exit(0)
        else:
            print(f"⚠️  {overall_total - overall_passed} modules need attention.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()