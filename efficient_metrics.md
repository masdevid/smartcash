============================================================
2025-08-03 23:56:08 | INFO | 🔍 HIERARCHICAL PROCESSING BATCH
2025-08-03 23:56:08 | INFO | ============================================================
2025-08-03 23:56:08 | INFO | Input tensor shapes:
2025-08-03 23:56:08 | INFO |   • predictions: torch.Size([16, 36, 6])
2025-08-03 23:56:08 | INFO |   • targets: torch.Size([16, 6])
2025-08-03 23:56:08 | INFO | 🔹 PHASE 2 DETECTED: Hierarchical multi-layer processing
2025-08-03 23:56:08 | INFO |   • Applying hierarchical filtering and confidence modulation
2025-08-03 23:56:08 | INFO | 
📊 PHASE 2 PREDICTION ANALYSIS:
2025-08-03 23:56:08 | INFO |   • Original prediction classes detected: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16]
2025-08-03 23:56:08 | INFO |   • Total predictions: 576
2025-08-03 23:56:08 | INFO |   • Layer breakdown by class ranges:
2025-08-03 23:56:09 | INFO |     - Layer 1 (classes 0-6): 411 predictions
2025-08-03 23:56:09 | INFO |     - Layer 2 (classes 7-13): 139 predictions
2025-08-03 23:56:09 | INFO |     - Layer 3 (classes 14-16): 26 predictions
2025-08-03 23:56:12 | INFO | 
🔧 CONFIDENCE MODULATION DETAILED ANALYSIS:
2025-08-03 23:56:12 | INFO | ============================================================
2025-08-03 23:56:12 | INFO | 📊 MULTI-LAYER PREDICTION AVAILABILITY:
2025-08-03 23:56:12 | INFO |   • Layer 2 predictions available: Yes (139 predictions)
2025-08-03 23:56:12 | INFO |   • Layer 3 predictions available: Yes (26 predictions)
2025-08-03 23:56:12 | INFO |   • Has multi-layer predictions: Yes
2025-08-03 23:56:13 | INFO | 
🎚️  CONFIDENCE MODULATION SUMMARY:
2025-08-03 23:56:13 | INFO |   • Total Layer 1 predictions: 411
2025-08-03 23:56:13 | INFO |   • Confidence BOOSTED: 0 predictions (0.0%)
2025-08-03 23:56:13 | INFO |   • Confidence REDUCED: 0 predictions (0.0%)
2025-08-03 23:56:13 | INFO |   • Confidence UNCHANGED: 411 predictions (100.0%)
2025-08-03 23:56:13 | INFO | 
📈 CONFIDENCE STATISTICS:
2025-08-03 23:56:13 | INFO |   • Average original confidence: 0.0001
2025-08-03 23:56:13 | INFO |   • Average hierarchical confidence: 0.0001
2025-08-03 23:56:13 | INFO |   • Average confidence change: +0.0000
2025-08-03 23:56:13 | INFO | 
🎯 LAYER CONTRIBUTION ANALYSIS:
2025-08-03 23:56:13 | INFO |   • Layer 2 active contributions: 0/411 (0.0%)
2025-08-03 23:56:13 | INFO |   • Layer 3 active contributions: 0/411 (0.0%)
2025-08-03 23:56:13 | INFO | 
🔍 PER-CLASS CONFIDENCE MODULATION:
2025-08-03 23:56:13 | INFO |   Class 0: 282 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 1: 90 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 2: 2 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 3: 4 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 4: 17 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 5: 14 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO |   Class 6: 2 predictions
2025-08-03 23:56:13 | INFO |     • Confidence: 0.000 → 0.000 (+0.000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contrib: 0.000, Layer 3 contrib: 0.000
2025-08-03 23:56:13 | INFO | 
🔬 SAMPLE PREDICTION ANALYSIS (first 5):
2025-08-03 23:56:13 | INFO |   Prediction 1: Class 1
2025-08-03 23:56:13 | INFO |     • Original confidence: 0.0002
2025-08-03 23:56:13 | INFO |     • Hierarchical confidence: 0.0002 (+0.0000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • Layer 3 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • ❌ Money validation failed (L3≤0.1) - confidence reduced by 90%
2025-08-03 23:56:13 | INFO |   Prediction 2: Class 1
2025-08-03 23:56:13 | INFO |     • Original confidence: 0.0002
2025-08-03 23:56:13 | INFO |     • Hierarchical confidence: 0.0002 (+0.0000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • Layer 3 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • ❌ Money validation failed (L3≤0.1) - confidence reduced by 90%
2025-08-03 23:56:13 | INFO |   Prediction 3: Class 5
2025-08-03 23:56:13 | INFO |     • Original confidence: 0.0001
2025-08-03 23:56:13 | INFO |     • Hierarchical confidence: 0.0001 (+0.0000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • Layer 3 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • ❌ Money validation failed (L3≤0.1) - confidence reduced by 90%
2025-08-03 23:56:13 | INFO |   Prediction 4: Class 0
2025-08-03 23:56:13 | INFO |     • Original confidence: 0.0000
2025-08-03 23:56:13 | INFO |     • Hierarchical confidence: 0.0000 (+0.0000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • Layer 3 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • ❌ Money validation failed (L3≤0.1) - confidence reduced by 90%
2025-08-03 23:56:13 | INFO |   Prediction 5: Class 0
2025-08-03 23:56:13 | INFO |     • Original confidence: 0.0000
2025-08-03 23:56:13 | INFO |     • Hierarchical confidence: 0.0000 (+0.0000)
2025-08-03 23:56:13 | INFO |     • Layer 2 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • Layer 3 contribution: 0.0000
2025-08-03 23:56:13 | INFO |     • ❌ Money validation failed (L3≤0.1) - confidence reduced by 90%
2025-08-03 23:56:13 | INFO | ============================================================
2025-08-03 23:56:13 | INFO | 
🎯 HIERARCHICAL PROCESSING RESULTS:
2025-08-03 23:56:13 | INFO |   • Layer 1 predictions after filtering: 411
2025-08-03 23:56:13 | INFO |   • Layer 1 targets after filtering: 16
2025-08-03 23:56:13 | INFO |   • Confidence statistics:
2025-08-03 23:56:13 | INFO |     - Average: 0.0001
2025-08-03 23:56:13 | INFO |     - Min: 0.0000
2025-08-03 23:56:13 | INFO |     - Max: 0.0008
2025-08-03 23:56:13 | INFO |     - Total confidence sum: 0.0286
2025-08-03 23:56:13 | INFO |   • Final prediction classes: [0, 1, 2, 3, 4, 5, 6]
2025-08-03 23:56:13 | INFO |     - Class 0: 282 predictions, avg_conf=0.0000
2025-08-03 23:56:13 | INFO |     - Class 1: 90 predictions, avg_conf=0.0003
2025-08-03 23:56:13 | INFO |     - Class 2: 2 predictions, avg_conf=0.0001
2025-08-03 23:56:13 | INFO |     - Class 3: 4 predictions, avg_conf=0.0001
2025-08-03 23:56:13 | INFO |     - Class 4: 17 predictions, avg_conf=0.0001
2025-08-03 23:56:13 | INFO |     - Class 5: 14 predictions, avg_conf=0.0001
2025-08-03 23:56:13 | INFO |     - Class 6: 2 predictions, avg_conf=0.0001
2025-08-03 23:56:14 | INFO | 



🔍 STARTING mAP COMPUTATION ANALYSIS (Epoch validation)
2025-08-04 05:33:39 | INFO | 📊 Total batches processed: 22
2025-08-04 05:33:39 | INFO | 📊 Total stat entries: 22
2025-08-04 05:33:39 | INFO | 📊 Total detection samples: 25631
2025-08-04 05:33:39 | INFO | 
================================================================================
2025-08-04 05:33:39 | INFO | 🔍 COMPREHENSIVE mAP DEBUG ANALYSIS
2025-08-04 05:33:39 | INFO | ================================================================================
2025-08-04 05:33:39 | INFO | 📊 OVERALL STATISTICS:
2025-08-04 05:33:39 | INFO |   • Total predictions: 25,631
2025-08-04 05:33:39 | INFO |   • Total targets: 343
2025-08-04 05:33:39 | INFO |   • Total True Positives (TP): 2,685
2025-08-04 05:33:39 | INFO |   • Total False Positives (FP): 22,946
2025-08-04 05:33:39 | INFO |   • Overall Precision: 0.1048
2025-08-04 05:33:39 | INFO |   • Confidence range: 0.005000 - 0.283879
2025-08-04 05:33:39 | INFO | 
📋 CLASS DISTRIBUTION:
2025-08-04 05:33:39 | INFO |   • Predicted classes: [0 1 2 3 4 5 6]
2025-08-04 05:33:39 | INFO |   • Target classes: [        0         1         2         3         4         5         6 223692160]
2025-08-04 05:33:39 | INFO |   • Classes in both pred & target: [0 1 2 3 4 5 6]
2025-08-04 05:33:39 | INFO |   • Classes only in predictions: []
2025-08-04 05:33:39 | INFO |   • Classes only in targets: [223692160]
2025-08-04 05:33:39 | INFO | 
🎯 PER-CLASS DETAILED ANALYSIS:
2025-08-04 05:33:39 | INFO | 
  CLASS 0 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 1,950 predictions, 71 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 38
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 1,912
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.0195, Recall=0.5352, F1=0.0376
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0100, min=0.0050, max=0.1035
2025-08-04 05:33:39 | INFO | 
  CLASS 1 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 1,231 predictions, 19 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 67
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 1,164
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.0544, Recall=3.5263, F1=0.1072
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0066, min=0.0050, max=0.0210
2025-08-04 05:33:39 | INFO | 
  CLASS 2 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 4,408 predictions, 59 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 594
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 3,814
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.1348, Recall=10.0678, F1=0.2660
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0174, min=0.0050, max=0.1617
2025-08-04 05:33:39 | INFO | 
  CLASS 3 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 3,836 predictions, 45 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 633
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 3,203
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.1650, Recall=14.0667, F1=0.3262
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0148, min=0.0050, max=0.1988
2025-08-04 05:33:39 | INFO | 
  CLASS 4 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 4,937 predictions, 36 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 53
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 4,884
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.0107, Recall=1.4722, F1=0.0213
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0163, min=0.0050, max=0.1179
2025-08-04 05:33:39 | INFO | 
  CLASS 5 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 4,635 predictions, 55 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 384
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 4,251
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.0828, Recall=6.9818, F1=0.1638
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0148, min=0.0050, max=0.2839
2025-08-04 05:33:39 | INFO | 
  CLASS 6 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 4,634 predictions, 56 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 916
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 3,718
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.1977, Recall=16.3571, F1=0.3906
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0114, min=0.0050, max=0.1177
2025-08-04 05:33:39 | INFO | 
  CLASS 223692160 ANALYSIS:
2025-08-04 05:33:39 | INFO |     📊 Counts: 0 predictions, 2 targets
2025-08-04 05:33:39 | INFO |     ✅ True Positives: 0
2025-08-04 05:33:39 | INFO |     ❌ False Positives: 0
2025-08-04 05:33:39 | INFO |     📈 Metrics: Precision=0.0000, Recall=0.0000, F1=0.0000
2025-08-04 05:33:39 | INFO |     🎯 Confidence: avg=0.0000, min=0.0000, max=0.0000
2025-08-04 05:33:39 | INFO |     ⚠️  ISSUE: NO PREDICTIONS for class 223692160 - missing all 2 targets!
2025-08-04 05:33:39 | INFO | 
🎚️  CONFIDENCE THRESHOLD ANALYSIS:
2025-08-04 05:33:39 | INFO |     Threshold | Predictions | True Positives | Precision
2025-08-04 05:33:39 | INFO |     ---------|-------------|----------------|----------
2025-08-04 05:33:39 | INFO |          0.1 |          91 |             25 |   0.2747
2025-08-04 05:33:39 | INFO |          0.2 |           8 |              2 |   0.2500
2025-08-04 05:33:39 | INFO |          0.3 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.4 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.5 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.6 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.7 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.8 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO |          0.9 |           0 |              0 |   0.0000
2025-08-04 05:33:39 | INFO | 
📋 SUMMARY INSIGHTS:
2025-08-04 05:33:39 | INFO |    • Overall precision: 0.1048
2025-08-04 05:33:39 | INFO |    • Average targets per class: 42.9
2025-08-04 05:33:39 | INFO |    • Low precision - model needs more training or threshold tuning
2025-08-04 05:33:39 | INFO | ================================================================================
2025-08-04 05:33:39 | INFO | ⚡ Starting ap_per_class computation with 25631 predictions and 343 targets
2025-08-04 05:33:39 | INFO | ✅ ap_per_class computation completed
2025-08-04 05:33:40 | INFO | 📊 AP RESULTS:
2025-08-04 05:33:40 | INFO |   • AP shape: (8, 1)
2025-08-04 05:33:40 | INFO |   • AP classes: [        0         1         2         3         4         5         6 223692160]
2025-08-04 05:33:40 | INFO |   • AP values: [   0.015017    0.067067     0.13479     0.45073    0.010816     0.35844     0.20228           0]
2025-08-04 05:33:40 | INFO |   • Precision: [   0.019487    0.054427     0.13475     0.16502    0.010735    0.082848     0.19767           0]
2025-08-04 05:33:40 | INFO |   • Recall: [    0.53521      3.5263      10.068      14.067      1.4722      6.9818      16.357           0]
2025-08-04 05:33:40 | INFO | Hierarchical mAP: 0.1549, precision: 0.0831, recall: 6.6259
