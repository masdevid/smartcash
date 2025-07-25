#!/usr/bin/env python3
"""
System memory diagnosis to understand why training processes are being killed
Provides detailed memory analysis and recommendations
"""

import sys
import os
import gc
import psutil
import torch
from pathlib import Path

def check_system_memory():
    """Check detailed system memory information"""
    print("🔍 SYSTEM MEMORY DIAGNOSIS")
    print("="*60)
    
    # Get memory info
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"📊 Physical Memory:")
    print(f"   • Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   • Available: {memory.available / (1024**3):.1f} GB")
    print(f"   • Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    print(f"   • Free: {memory.free / (1024**3):.1f} GB")
    
    print(f"\n💽 Swap Memory:")
    print(f"   • Total Swap: {swap.total / (1024**3):.1f} GB")
    print(f"   • Used Swap: {swap.used / (1024**3):.1f} GB ({swap.percent:.1f}%)")
    print(f"   • Free Swap: {swap.free / (1024**3):.1f} GB")
    
    # Memory pressure analysis
    available_gb = memory.available / (1024**3)
    if available_gb < 2:
        print(f"\n🚨 CRITICAL: Only {available_gb:.1f}GB available!")
        print("   System is under severe memory pressure")
    elif available_gb < 4:
        print(f"\n⚠️  WARNING: Only {available_gb:.1f}GB available")
        print("   Training may be killed by system")
    else:
        print(f"\n✅ OK: {available_gb:.1f}GB available for training")

def check_running_processes():
    """Check memory-hungry processes"""
    print("\n🔍 TOP MEMORY CONSUMING PROCESSES:")
    print("="*60)
    
    # Get processes sorted by memory usage
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by memory usage
    processes.sort(key=lambda x: x['memory_info'].rss, reverse=True)
    
    print(f"{'PID':<8} {'Memory':<10} {'Process Name':<30}")
    print("-" * 50)
    
    for proc in processes[:10]:  # Top 10
        memory_mb = proc['memory_info'].rss / (1024**2)
        if memory_mb > 100:  # Only show processes using >100MB
            print(f"{proc['pid']:<8} {memory_mb:<10.0f}MB {proc['name']:<30}")

def test_pytorch_memory():
    """Test PyTorch memory allocation"""
    print("\n🔍 PYTORCH MEMORY TEST:")
    print("="*60)
    
    try:
        # Test small tensor allocation
        print("   Testing small tensor (1MB)...")
        small_tensor = torch.zeros(256, 256)
        print("   ✅ Small tensor OK")
        del small_tensor
        gc.collect()
        
        # Test medium tensor allocation
        print("   Testing medium tensor (100MB)...")
        medium_tensor = torch.zeros(2560, 2560)
        print("   ✅ Medium tensor OK")
        del medium_tensor
        gc.collect()
        
        # Test large tensor allocation
        print("   Testing large tensor (1GB)...")
        try:
            large_tensor = torch.zeros(8192, 8192)
            print("   ✅ Large tensor OK")
            del large_tensor
            gc.collect()
        except RuntimeError as e:
            print(f"   ❌ Large tensor failed: {e}")
            
    except Exception as e:
        print(f"   ❌ PyTorch memory test failed: {e}")

def check_disk_space():
    """Check available disk space for checkpoints"""
    print("\n🔍 DISK SPACE CHECK:")
    print("="*60)
    
    # Check current directory
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    total_gb = disk_usage.total / (1024**3)
    used_percent = (disk_usage.used / disk_usage.total) * 100
    
    print(f"   • Total Disk: {total_gb:.1f} GB")
    print(f"   • Free Space: {free_gb:.1f} GB")
    print(f"   • Used: {used_percent:.1f}%")
    
    if free_gb < 1:
        print("   🚨 CRITICAL: Less than 1GB free!")
    elif free_gb < 5:
        print("   ⚠️  WARNING: Low disk space")
    else:
        print("   ✅ Sufficient disk space")

def memory_recommendations():
    """Provide memory optimization recommendations"""
    print("\n💡 MEMORY OPTIMIZATION RECOMMENDATIONS:")
    print("="*60)
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 2:
        print("🚨 CRITICAL RECOMMENDATIONS:")
        print("   1. Restart your Mac immediately")
        print("   2. Close ALL applications (Safari, Chrome, etc.)")
        print("   3. Disable browser tabs")
        print("   4. Quit Xcode, Docker, VMs if running")
        print("   5. Use Activity Monitor to kill memory hogs")
        print("   6. Consider cloud training (Google Colab)")
        
    elif available_gb < 4:
        print("⚠️  WARNING RECOMMENDATIONS:")
        print("   1. Close non-essential applications")
        print("   2. Use only one browser with minimal tabs")
        print("   3. Try training during system idle time")
        print("   4. Consider restarting before training")
        
    else:
        print("✅ SYSTEM READY FOR TRAINING:")
        print("   • Sufficient memory available")
        print("   • Try conservative training first")
        print("   • Monitor system during training")
    
    print(f"\n🎯 TRAINING STRATEGY:")
    if available_gb >= 4:
        print("   • Try: memory_optimized_efficientnet.py")
        print("   • Use: --batch-size 1")
    elif available_gb >= 2:
        print("   • Try: ultra_low_memory_training.py")
        print("   • Use: --epochs 1")
    else:
        print("   • Try: emergency_micro_training.py")
        print("   • Or consider cloud training")

def main():
    """Run complete system diagnosis"""
    print("🏥 SMARTCASH TRAINING SYSTEM DIAGNOSIS")
    print("="*70)
    print("This tool diagnoses why training processes are being killed")
    print("="*70)
    
    check_system_memory()
    check_running_processes()
    test_pytorch_memory()
    check_disk_space()
    memory_recommendations()
    
    print("\n" + "="*70)
    print("🏁 DIAGNOSIS COMPLETE")
    print("="*70)
    print("Use the recommendations above to optimize your system")
    print("If all else fails, cloud training is recommended")

if __name__ == "__main__":
    main()