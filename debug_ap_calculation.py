#!/usr/bin/env python3
"""
Debug script to trace AP calculation and identify why each class returns 0.0.
"""

import torch
import numpy as np
from smartcash.model.training.metrics_tracker import APCalculator
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def debug_ap_calculation():
    """Debug AP calculation to identify why all classes return 0.0."""
    print("🔍 DEBUG: Tracing AP calculation issue")
    print("=" * 60)
    
    # Create test AP calculator
    ap_calc = APCalculator()
    
    # Create simple test data to ensure IoU matches should work
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create very simple test case with guaranteed matches
    # 1 prediction and 1 ground truth for class 0, with identical boxes
    pred_boxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32, device=device)  # [1, 1, 4]
    pred_scores = torch.tensor([[0.9]], dtype=torch.float32, device=device)  # [1, 1]
    pred_classes = torch.tensor([[0]], dtype=torch.long, device=device)  # [1, 1] - class 0
    
    true_boxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32, device=device)  # [1, 1, 4] - identical box
    true_classes = torch.tensor([[0]], dtype=torch.long, device=device)  # [1, 1] - class 0
    
    image_ids = [0]
    
    print("Test data:")
    print(f"  Pred boxes: {pred_boxes}")
    print(f"  Pred scores: {pred_scores}")
    print(f"  Pred classes: {pred_classes}")
    print(f"  True boxes: {true_boxes}")
    print(f"  True classes: {true_classes}")
    print(f"  Image IDs: {image_ids}")
    
    # Add to AP calculator
    print("\n🔄 Adding data to AP calculator...")
    ap_calc.add_batch(pred_boxes, pred_scores, pred_classes, true_boxes, true_classes, image_ids)
    
    print(f"AP calculator state:")
    print(f"  Predictions: {len(ap_calc.predictions)}")
    print(f"  Targets: {len(ap_calc.targets)}")
    
    if ap_calc.predictions:
        print(f"  First prediction: {ap_calc.predictions[0]}")
    if ap_calc.targets:
        print(f"  First target: {ap_calc.targets[0]}")
    
    # Test IoU calculation manually
    print("\n🧮 Testing IoU calculation...")
    if ap_calc.predictions and ap_calc.targets:
        pred_box = ap_calc.predictions[0][2:6]  # [x1, y1, x2, y2]
        true_box = ap_calc.targets[0][1:5]      # [x1, y1, x2, y2]
        
        print(f"  Pred box: {pred_box}")
        print(f"  True box: {true_box}")
        
        iou = ap_calc._calculate_iou(pred_box, true_box)
        print(f"  IoU: {iou}")
        
        if iou < 0.5:
            print("❌ ISSUE: IoU below threshold!")
        else:
            print("✅ IoU above threshold")
    
    # Test AP computation for class 0
    print("\n📊 Testing AP computation for class 0...")
    try:
        ap_class_0 = ap_calc.compute_ap(class_id=0, iou_threshold=0.5)
        print(f"AP for class 0: {ap_class_0}")
        
        if ap_class_0 == 0.0:
            print("❌ ISSUE: AP for class 0 is 0.0")
            
            # Debug the AP calculation step by step
            print("\n🔍 Debugging AP calculation step by step...")
            
            # Filter predictions and targets for class 0
            class_preds = [p for p in ap_calc.predictions if p[1] == 0]
            class_targets = [t for t in ap_calc.targets if t[0] == 0]
            
            print(f"  Class 0 predictions: {len(class_preds)}")
            print(f"  Class 0 targets: {len(class_targets)}")
            
            if class_preds:
                print(f"  First class pred: {class_preds[0]}")
            if class_targets:
                print(f"  First class target: {class_targets[0]}")
            
            if len(class_targets) == 0:
                print("❌ No targets for class 0!")
            elif len(class_preds) == 0:
                print("❌ No predictions for class 0!")
            else:
                # Sort predictions by confidence
                class_preds.sort(key=lambda x: x[0], reverse=True)
                
                # Manual matching process
                num_targets = len(class_targets)
                matched = [False] * num_targets
                tp = []
                fp = []
                
                print(f"  Processing {len(class_preds)} predictions against {num_targets} targets...")
                
                for i, pred in enumerate(class_preds):
                    pred_box = pred[2:6]
                    img_id = pred[6]
                    
                    print(f"    Pred {i}: conf={pred[0]:.3f}, box={pred_box}, img_id={img_id}")
                    
                    # Find best matching target
                    best_iou = 0.0
                    best_idx = -1
                    
                    for j, target in enumerate(class_targets):
                        if target[5] != img_id or matched[j]:
                            print(f"      Target {j}: skipped (img_id={target[5]}, matched={matched[j]})")
                            continue
                        
                        target_box = target[1:5]
                        iou = ap_calc._calculate_iou(pred_box, target_box)
                        print(f"      Target {j}: box={target_box}, IoU={iou:.3f}")
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = j
                    
                    print(f"    Best match: idx={best_idx}, IoU={best_iou:.3f}")
                    
                    # Check if match meets IoU threshold
                    if best_iou >= 0.5 and best_idx >= 0:
                        tp.append(1)
                        fp.append(0)
                        matched[best_idx] = True
                        print(f"    Result: TP (matched target {best_idx})")
                    else:
                        tp.append(0)
                        fp.append(1)
                        print(f"    Result: FP (no match)")
                
                print(f"  Final TP: {tp}")
                print(f"  Final FP: {fp}")
                
                # Calculate precision/recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                recalls = tp_cumsum / num_targets
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                print(f"  Recalls: {recalls}")
                print(f"  Precisions: {precisions}")
                
                # Calculate AP using 11-point interpolation
                ap = 0.0
                for recall_threshold in np.linspace(0, 1, 11):
                    precision_at_recall = 0.0
                    for k in range(len(recalls)):
                        if recalls[k] >= recall_threshold:
                            precision_at_recall = max(precision_at_recall, precisions[k])
                    ap += precision_at_recall / 11
                    print(f"  Recall threshold {recall_threshold:.1f}: precision = {precision_at_recall:.3f}")
                
                print(f"  Final AP: {ap}")
        else:
            print("✅ AP calculation working correctly")
    except Exception as e:
        print(f"❌ Error in AP calculation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test full mAP computation
    print("\n📊 Testing full mAP computation...")
    try:
        map_score, class_aps = ap_calc.compute_map(iou_threshold=0.5)
        print(f"mAP@0.5: {map_score}")
        print(f"Class APs: {class_aps}")
        
        valid_aps = [ap for ap in class_aps.values() if ap > 0]
        print(f"Valid APs (>0): {valid_aps}")
        
        if map_score == 0.0:
            print("❌ ISSUE CONFIRMED: mAP is 0.0")
            if not valid_aps:
                print("  Root cause: No class has AP > 0")
            else:
                print("  Unexpected: Some classes have AP > 0 but mAP is still 0")
        
    except Exception as e:
        print(f"❌ Error in mAP calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ap_calculation()