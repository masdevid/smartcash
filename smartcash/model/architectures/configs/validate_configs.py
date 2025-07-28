#!/usr/bin/env python3
"""
Configuration Validation Script
Validates that all model architecture configurations comply with MODEL_ARC.md specifications.
"""

import yaml
from pathlib import Path
import sys

def validate_config_file(config_path: Path) -> dict:
    """Validate a single configuration file against MODEL_ARC.md requirements."""
    results = {
        'file': config_path.name,
        'valid': True,
        'issues': []
    }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Failed to load YAML: {e}")
        return results
    
    # Check required sections
    required_sections = ['nc', 'depth_multiple', 'width_multiple', 'anchors', 
                        'backbone', 'head', 'layer_specs', 'training']
    
    for section in required_sections:
        if section not in config:
            results['valid'] = False
            results['issues'].append(f"Missing required section: {section}")
    
    # Validate layer specifications
    if 'layer_specs' in config:
        expected_layers = ['layer_1', 'layer_2', 'layer_3']
        for layer in expected_layers:
            if layer not in config['layer_specs']:
                results['valid'] = False
                results['issues'].append(f"Missing layer specification: {layer}")
            else:
                layer_spec = config['layer_specs'][layer]
                if 'nc' not in layer_spec or 'classes' not in layer_spec:
                    results['valid'] = False
                    results['issues'].append(f"Invalid layer specification for {layer}")
    
    # Validate training configuration
    if 'training' in config:
        training = config['training']
        
        # Check for uncertainty-based loss weighting (MODEL_ARC.md compliance)
        if 'loss_function' not in training:
            results['valid'] = False
            results['issues'].append("Missing loss_function in training config")
        elif "λ1 * loss_layer1" not in training['loss_function']:
            results['valid'] = False
            results['issues'].append("Incorrect loss_function format")
        
        if 'loss_weighting' not in training:
            results['valid'] = False
            results['issues'].append("Missing loss_weighting in training config")
        elif training['loss_weighting'] != "uncertainty-based dynamic weighting":
            results['valid'] = False
            results['issues'].append("Incorrect loss_weighting - must be 'uncertainty-based dynamic weighting'")
        
        # Check for deprecated static loss weights
        if 'loss_weights' in training:
            results['valid'] = False
            results['issues'].append("Found deprecated static loss_weights - should use uncertainty-based weighting")
        
        # Check for two-phase training
        required_training_params = ['phase_1_epochs', 'phase_2_epochs', 'optimizer']
        for param in required_training_params:
            if param not in training:
                results['valid'] = False
                results['issues'].append(f"Missing training parameter: {param}")
        
        # Validate optimizer configuration
        if 'optimizer' in training:
            optimizer = training['optimizer']
            if 'backbone_lr' not in optimizer or 'head_lr' not in optimizer:
                results['valid'] = False
                results['issues'].append("Missing backbone_lr or head_lr in optimizer config")
    
    # Validate class counts match MODEL_ARC.md
    if 'layer_specs' in config:
        layer_specs = config['layer_specs']
        expected_classes = {
            'layer_1': 7,  # ["001", "002", "005", "010", "020", "050", "100"]
            'layer_2': 7,  # ["l2_001", "l2_002", "l2_005", "l2_010", "l2_020", "l2_050", "l2_100"]
            'layer_3': 3   # ["l3_sign", "l3_text", "l3_thread"]
        }
        
        for layer, expected_count in expected_classes.items():
            if layer in layer_specs:
                if layer_specs[layer]['nc'] != expected_count:
                    results['valid'] = False
                    results['issues'].append(f"{layer} has {layer_specs[layer]['nc']} classes, expected {expected_count}")
                
                if len(layer_specs[layer]['classes']) != expected_count:
                    results['valid'] = False
                    results['issues'].append(f"{layer} class list length mismatch")
    
    return results

def main():
    """Main validation function."""
    config_dir = Path(__file__).parent / "models" / "yolov5"
    
    print("🔍 Validating SmartCash Model Architecture Configurations")
    print("=" * 60)
    print(f"📂 Configuration directory: {config_dir}")
    print()
    
    all_valid = True
    total_configs = 0
    valid_configs = 0
    
    # Find all configuration files
    for backbone_dir in config_dir.iterdir():
        if not backbone_dir.is_dir():
            continue
            
        print(f"📋 Validating {backbone_dir.name} configurations:")
        
        for config_file in backbone_dir.glob("*.yaml"):
            total_configs += 1
            result = validate_config_file(config_file)
            
            if result['valid']:
                print(f"  ✅ {result['file']} - Valid")
                valid_configs += 1
            else:
                print(f"  ❌ {result['file']} - Invalid")
                for issue in result['issues']:
                    print(f"     • {issue}")
                all_valid = False
        print()
    
    # Summary
    print("=" * 60)
    if all_valid:
        print(f"🎉 All {total_configs} configurations are valid!")
        print("✅ MODEL_ARC.md compliance: PASSED")
        print("✅ Uncertainty-based loss weighting: ENABLED")
        print("✅ Multi-layer specifications: CORRECT")
    else:
        print(f"❌ {total_configs - valid_configs} out of {total_configs} configurations have issues")
        print("❌ MODEL_ARC.md compliance: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()