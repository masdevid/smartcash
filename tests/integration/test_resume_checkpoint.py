#!/usr/bin/env python3
"""
Test script to verify resume checkpoint functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the resume function
from examples.callback_only_training_example import load_legacy_checkpoint_for_resume

def test_resume_functionality():
    """Test the resume checkpoint functionality"""
    
    # Test CSPDarkNet checkpoint
    csp_checkpoint = "data/checkpoints/best_cspdarknet_two_phase_multi_unfrozen_pretrained_20250729.pt"
    print("🧪 Testing CSPDarkNet checkpoint resume...")
    
    if Path(csp_checkpoint).exists():
        resume_info = load_legacy_checkpoint_for_resume(csp_checkpoint)
        if resume_info:
            print("✅ CSPDarkNet checkpoint loaded successfully!")
            print(f"   Original epoch: {resume_info.get('saved_epoch', 'N/A')}")
            print(f"   Resume epoch: {resume_info['epoch']}")
            print(f"   Resume phase: {resume_info['phase']}")
        else:
            print("❌ Failed to load CSPDarkNet checkpoint")
    else:
        print(f"⚠️ CSPDarkNet checkpoint not found: {csp_checkpoint}")
    
    print()
    
    # Test EfficientNet checkpoint
    eff_checkpoint = "data/checkpoints/best_efficientnet_b4_two_phase_multi_unfrozen_pretrained_20250729.pt"
    print("🧪 Testing EfficientNet-B4 checkpoint resume...")
    
    if Path(eff_checkpoint).exists():
        resume_info = load_legacy_checkpoint_for_resume(eff_checkpoint)
        if resume_info:
            print("✅ EfficientNet-B4 checkpoint loaded successfully!")
            print(f"   Original epoch: {resume_info.get('saved_epoch', 'N/A')}")
            print(f"   Resume epoch: {resume_info['epoch']}")
            print(f"   Resume phase: {resume_info['phase']}")
        else:
            print("❌ Failed to load EfficientNet-B4 checkpoint")
    else:
        print(f"⚠️ EfficientNet-B4 checkpoint not found: {eff_checkpoint}")

if __name__ == "__main__":
    test_resume_functionality()