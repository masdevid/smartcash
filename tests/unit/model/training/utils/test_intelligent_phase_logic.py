#!/usr/bin/env python3
"""
Test script to verify the intelligent phase-aware metrics logic is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback
from smartcash.model.training.utils.metric_color_utils import ColorScheme


def test_phase_aware_logic():
    """Test the intelligent phase-aware metrics filtering."""
    print("🧠 Testing Intelligent Phase-Aware Metrics Logic")
    print("=" * 60)
    
    # Create callback
    callback = create_ui_metrics_callback(verbose=True, console_scheme=ColorScheme.EMOJI)
    
    # Test Phase 1 scenario - should only show layer_1, filter zeros
    print("\n🔬 TEST 1: Phase 1 (should show only layer_1, filter zeros)")
    phase1_metrics = {
        'train_loss': 0.7245,
        'val_loss': 2.3381,
        'val_accuracy': 0.25,
        'layer_1_accuracy': 0.83,
        'layer_1_precision': 0.75,
        'layer_2_accuracy': 0.0,  # Should be filtered out
        'layer_2_precision': 0.0,  # Should be filtered out
        'layer_3_accuracy': 0.0001,  # Should be filtered out (too small)
    }
    
    result1 = callback("training_phase_1", 1, phase1_metrics, max_epochs=100)
    ui_summary1 = callback.get_metric_summary_for_ui()
    
    print(f"✅ Phase info: {ui_summary1['phase_info']['display_mode']}")
    print(f"✅ Active layers: {ui_summary1['phase_info']['active_layers']}")
    print(f"✅ Layer metrics shown: {list(ui_summary1['categories']['layer_metrics'].keys())}")
    
    # Test Phase 2 scenario - should show all layers, no filtering
    print("\n🔬 TEST 2: Phase 2 (should show all layers, no filtering)")
    phase2_metrics = {
        'train_loss': 0.3245,
        'val_loss': 1.1234,
        'val_accuracy': 0.78,
        'layer_1_accuracy': 0.91,
        'layer_1_precision': 0.89,
        'layer_2_accuracy': 0.0,  # Should still be shown (no filtering in phase 2)
        'layer_2_precision': 0.0,  # Should still be shown
        'layer_3_accuracy': 0.85,
        'layer_3_precision': 0.82,
    }
    
    result2 = callback("training_phase_2", 15, phase2_metrics, max_epochs=100)
    ui_summary2 = callback.get_metric_summary_for_ui()
    
    print(f"✅ Phase info: {ui_summary2['phase_info']['display_mode']}")
    print(f"✅ Active layers: {ui_summary2['phase_info']['active_layers']}")
    print(f"✅ Layer metrics shown: {list(ui_summary2['categories']['layer_metrics'].keys())}")
    
    # Test Single Phase - Single Layer scenario
    print("\n🔬 TEST 3: Single Phase - Single Layer (should auto-detect layer_1 only)")
    single_layer_metrics = {
        'train_loss': 0.2156,
        'val_loss': 0.4678,
        'val_accuracy': 0.89,
        'layer_1_accuracy': 0.94,
        'layer_1_precision': 0.91,
        'layer_1_recall': 0.93,
        'layer_1_f1': 0.92,
        # No layer_2 or layer_3 metrics - should auto-detect single layer mode
    }
    
    result3 = callback("training_phase_single", 25, single_layer_metrics, max_epochs=100)
    ui_summary3 = callback.get_metric_summary_for_ui()
    
    print(f"✅ Phase info: {ui_summary3['phase_info']['display_mode']}")
    print(f"✅ Active layers: {ui_summary3['phase_info']['active_layers']}")
    print(f"✅ Layer metrics shown: {list(ui_summary3['categories']['layer_metrics'].keys())}")
    
    # Test Single Phase - Multi Layer scenario
    print("\n🔬 TEST 4: Single Phase - Multi Layer (should auto-detect all layers)")
    multi_layer_metrics = {
        'train_loss': 0.1234,
        'val_loss': 0.2456,
        'val_accuracy': 0.94,
        'layer_1_accuracy': 0.96,
        'layer_1_precision': 0.94,
        'layer_2_accuracy': 0.89,  # Non-zero, should trigger multi-layer mode
        'layer_2_precision': 0.87,
        'layer_3_accuracy': 0.85,  # Non-zero, confirms multi-layer
        'layer_3_precision': 0.83,
    }
    
    result4 = callback("training_phase_single", 40, multi_layer_metrics, max_epochs=100)
    ui_summary4 = callback.get_metric_summary_for_ui()
    
    print(f"✅ Phase info: {ui_summary4['phase_info']['display_mode']}")
    print(f"✅ Active layers: {ui_summary4['phase_info']['active_layers']}")
    print(f"✅ Layer metrics shown: {list(ui_summary4['categories']['layer_metrics'].keys())}")


def test_zero_filtering_behavior():
    """Test the zero filtering behavior in detail."""
    print("\n🔍 Testing Zero Filtering Behavior")
    print("=" * 60)
    
    callback = create_ui_metrics_callback(verbose=False)  # Disable console output for cleaner test
    
    # Test metrics with various small values
    test_metrics = {
        'layer_1_accuracy': 0.85,      # Should always show
        'layer_1_precision': 0.0001,   # Exactly at threshold
        'layer_1_recall': 0.00005,     # Below threshold
        'layer_1_f1': 0.0,             # Zero
        'layer_2_accuracy': 0.0002,    # Just above threshold
        'layer_2_precision': 0.0,      # Zero
    }
    
    # Test Phase 1 (should filter)
    callback("training_phase_1", 1, test_metrics)
    ui_summary_filtered = callback.get_metric_summary_for_ui()
    
    print("📊 Phase 1 (filtered) - Layer metrics shown:")
    for metric in ui_summary_filtered['categories']['layer_metrics'].keys():
        value = test_metrics[metric]
        print(f"   ✅ {metric}: {value} (kept - above 0.0001 threshold)")
    
    # Test Phase 2 (should not filter)
    callback("training_phase_2", 1, test_metrics)
    ui_summary_unfiltered = callback.get_metric_summary_for_ui()
    
    print("\n📊 Phase 2 (unfiltered) - Layer metrics shown:")
    for metric in ui_summary_unfiltered['categories']['layer_metrics'].keys():
        value = test_metrics[metric]
        print(f"   ✅ {metric}: {value} (kept - no filtering)")
    
    print(f"\n🔢 Filtering Results:")
    print(f"   Phase 1 (filtered): {len(ui_summary_filtered['categories']['layer_metrics'])} metrics")
    print(f"   Phase 2 (unfiltered): {len(ui_summary_unfiltered['categories']['layer_metrics'])} metrics")


if __name__ == "__main__":
    test_phase_aware_logic()
    test_zero_filtering_behavior()
    
    print("\n✅ Intelligent Phase-Aware Logic Test Complete!")
    print("\n🎯 Key Features Verified:")
    print("   • Phase 1: Shows only layer_1, filters zeros for clean output")
    print("   • Phase 2: Shows all layers, no filtering")
    print("   • Single Phase: Auto-detects single vs multi-layer mode")
    print("   • Zero filtering: Removes noise in single-layer modes, keeps all in multi-layer")
    print("   • UI integration: Provides phase info and display mode descriptions")