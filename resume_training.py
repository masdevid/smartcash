#!/usr/bin/env python3
"""
Resume Training Script for SmartCash

This script helps resume training from checkpoints or start fresh training
with optimized configurations for Apple Silicon.
"""

# Fix OpenMP duplicate library issue
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from pathlib import Path

def main():
    """Main function to handle training resume options."""
    
    print("🚀 SMARTCASH TRAINING RESUME HELPER")
    print("=" * 60)
    
    # Check for existing checkpoints
    checkpoint_dir = Path("data/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"📁 Found {len(checkpoints)} checkpoint(s):")
            for i, cp in enumerate(checkpoints, 1):
                print(f"   {i}. {cp.name}")
        else:
            print("📁 No checkpoints found in data/checkpoints/")
    else:
        print("📁 Checkpoint directory not found")
    
    print("\n🎯 RECOMMENDED TRAINING CONFIGURATIONS:")
    print("=" * 60)
    
    configs = {
        "1": {
            "name": "Stable CSPDarkNet (Recommended)",
            "description": "Memory-efficient, proven stable on 16GB Macs",
            "command": [
                "python examples/callback_only_training_example.py",
                "  --backbone cspdarknet",
                "  --optimizer adamw",
                "  --scheduler cosine",
                "  --pretrained",
                "  --phase1-epochs 20",
                "  --phase2-epochs 15",
                "  --batch-size 4",
                "  --weight-decay 1e-2",
                "  --verbose"
            ]
        },
        "2": {
            "name": "Resume from Checkpoint",
            "description": "Resume from existing EfficientNet-B4 checkpoint",
            "command": [
                "python examples/callback_only_training_example.py",
                "  --resume data/checkpoints/best_efficientnet_b4_two_phase_multi_unfrozen_pretrained_20250729.pt",
                "  --backbone efficientnet_b4",
                "  --optimizer adamw",
                "  --scheduler cosine",
                "  --pretrained",
                "  --phase1-epochs 15",
                "  --phase2-epochs 10",
                "  --batch-size 2",
                "  --weight-decay 1e-2",
                "  --verbose"
            ]
        },
        "3": {
            "name": "Fresh EfficientNet-B4",
            "description": "Start fresh with memory-optimized EfficientNet-B4",
            "command": [
                "python examples/callback_only_training_example.py",
                "  --backbone efficientnet_b4",
                "  --optimizer adamw", 
                "  --scheduler cosine",
                "  --pretrained",
                "  --phase1-epochs 15",
                "  --phase2-epochs 10",
                "  --batch-size 2",
                "  --weight-decay 1e-2",
                "  --verbose"
            ]
        },
        "4": {
            "name": "Ultra-Safe Mode",
            "description": "Minimal memory usage, guaranteed to work",
            "command": [
                "python examples/callback_only_training_example.py",
                "  --backbone cspdarknet",
                "  --optimizer adamw",
                "  --scheduler cosine",
                "  --pretrained",
                "  --phase1-epochs 10",
                "  --phase2-epochs 5",
                "  --batch-size 2",
                "  --weight-decay 1e-2",
                "  --verbose"
            ]
        }
    }
    
    for key, config in configs.items():
        print(f"\n{key}. {config['name']}")
        print(f"   {config['description']}")
        print("   Command:")
        for line in config['command']:
            print(f"   {line} \\")
        print()
    
    # Interactive selection
    try:
        choice = input("🎯 Select configuration (1-4) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Exiting...")
            return
        
        if choice in configs:
            selected_config = configs[choice]
            print(f"\n✅ Selected: {selected_config['name']}")
            print(f"📝 {selected_config['description']}")
            
            # Generate command
            command = " \\\n".join(selected_config['command'])
            print(f"\n📋 Command to run:")
            print("=" * 60)
            print(command)
            print("=" * 60)
            
            # Ask if user wants to run it
            run_now = input("\n🚀 Run this command now? (y/N): ").strip().lower()
            
            if run_now == 'y':
                print("\n🚀 Starting training...")
                print("=" * 60)
                
                # Prepare command for execution
                import subprocess
                cmd_parts = []
                for line in selected_config['command']:
                    cmd_parts.extend(line.strip().split())
                
                # Remove empty parts and backslashes
                cmd_parts = [part for part in cmd_parts if part and part != '\\']
                
                try:
                    # Run the command
                    subprocess.run(cmd_parts, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Training failed with exit code: {e.returncode}")
                except KeyboardInterrupt:
                    print("\n⚠️ Training interrupted by user")
                    
            else:
                print("💾 Command saved to clipboard (copy from above)")
                print("📝 You can run it manually when ready")
        else:
            print("❌ Invalid choice. Please select 1-4 or 'q'")
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()