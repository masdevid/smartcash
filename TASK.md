Updated 04 August 2025, 03:10:
# NOTES:
- Raw Data: `/content/data/{valid, train, test}/{images, labels}` or `/data/{valid, train, test}/{images, labels} (Symlink|Local)`
- Preprocessed Data: `/content/data/preprocessed/{valid, train, test}/{images, labels}` or `/data/preprocessed/{valid, train, test}/{images, labels} (Symlink|Local)`
- Augmented Data: `/content/data/augmented/{valid, train, test}/{images, labels}` or `/data/augmented/{valid, train, test}/{images, labels} (Symlink|Local)`
- Pretrained Data: `/content/datata/pretrained` or `/data/pretrained (Symlink|Local)`
- Checkpoint Data: `/content/data/checkpoints` or `/data/checkpoints (Symlink|Local)`
- Backbone Data: `/content/data/models` or `/data/models (Symlink|Local)`

## Data Organization:
1. Raw Data (/data/{train,valid,test}/{images, labels}/):
  - Files with prefix rp_* in regular image formats (.jpg, .png, etc.)
  - Labels as .txt files
2. Augmented Data (/data/augmented/{train,valid,test}/{images, labels}/):
  - Files with prefix aug_* in regular image formats (before normalization)
  - Labels as .txt files
3. Preprocessed Data (/data/preprocessed/{train,valid,test}/{images, labels}/):
  - Files with prefix pre_* as .npy files (preprocessed)
  - Files with prefix aug_* as .npy files (augmented after normalization)
  - Labels as .txt files

# CURRENT ISSUES:

# RESOLVED ISSUES:
- [✅] **high priority** Scenario augmentation system producing identical results - Fixed critical bug where both position_variation and lighting_variation scenarios used the same copied test files without augmentation. Implemented built-in PIL-based augmentation with position transforms (rotation, scaling, flipping) and lighting transforms (brightness, contrast, gamma). Now shows different results: position_variation achieves 21.9% mAP (harder due to geometric changes) vs lighting_variation 30.7% mAP (similar to original), properly testing model robustness (08/04/2025)
- [✅] **high priority** Denomination classification metrics data leakage - Fixed critical bug where perfect 100% accuracy was caused by excluding missed detections from metrics calculation. Now includes all 761 ground truth samples showing true performance: 0.3% accuracy (2 detected out of 761 samples), 100% precision on detected samples, revealing model's extremely conservative detection behavior (08/04/2025)
- [✅] **low priority** fix best model checkpoint saving on Phase 2 on Training Pipeline. Early stopping shows improvement but i don't see succesfull best model saved: (08/04/2025)
- [✅] **high priority** Validation bug in hierarchical metrics - Fixed critical bug where hierarchical validation was using corrupted class IDs (e.g., 223692160) and calculating impossible recall values (>1.0) due to incorrect True Positive counting. Implemented robust class ID validation and corrected the TP/FP logic to ensure metrics are accurate. (08/04/2025)
- [✅] **high priority** Contradictory logging in hierarchical processing - Fixed bug where the summary log incorrectly reported that no confidence scores were changed, while detailed logs showed reductions. Adjusted logging thresholds to accurately reflect all confidence modifications. (08/04/2025)
- [✅] **high priority** Scenario augmentation system producing identical results - Fixed critical bug where both position_variation and lighting_variation scenarios used the same copied test files without augmentation. Implemented built-in PIL-based augmentation with position transforms (rotation, scaling, flipping) and lighting transforms (brightness, contrast, gamma). Now shows different results: position_variation achieves 21.9% mAP (harder due to geometric changes) vs lighting_variation 30.7% mAP (similar to original), properly testing model robustness (08/04/2025)
- [✅] **high priority** Denomination classification metrics data leakage - Fixed critical bug where perfect 100% accuracy was caused by excluding missed detections from metrics calculation. Now includes all 761 ground truth samples showing true performance: 0.3% accuracy (2 detected out of 761 samples), 100% precision on detected samples, revealing model's extremely conservative detection behavior (08/04/2025)
- [✅] **high priority** Evaluation Checkpoint auto selector (all-scenario) - Fixed checkpoint selector to prioritize unified checkpoints over phase-specific variants. Now correctly selects best_cspdarknet_two_phase_multi_unfrozen_pretrained_20250803.pt (mAP: 0.466) and best_efficientnet_b4_two_phase_multi_unfrozen_pretrained_20250804.pt (mAP: 0.414) while skipping _phase1.pt and _phase2.pt variants (08/04/2025)
- [✅] **high priority** Evaluation metrics summary - Implemented comprehensive metrics with clear distinction: mAP-based metrics (map50, map50_precision, map50_recall, map50_f1) from YOLOv5 training module for object detection, and denomination classification metrics (accuracy, precision, recall, f1) for 7-class classification, plus performance metrics (inference_time, fps) (08/04/2025)
- [✅] **high priority** Confusion matrix for 7 class denomination classification - Created 8×8 confusion matrix including "no detection" class for denomination classification. Shows true detection performance: 759 missed detections vs 2 successful detections with perfect classification on detected samples (08/04/2025)
- [✅] Model output format compatibility with mAP@0.5 computation - Validated that SmartCash model outputs are properly converted by ValidationMapProcessor to YOLOv5 format for accurate mAP evaluation (08/03/2025)
- [✅] Phase 2 prediction format mismatch - Fixed training executor to use full processing for Phase 2 multi-layer predictions, ensuring Dict format for both predictions and targets (08/03/2025)
- [✅] Phase transition model rebuilding - Added proper model rebuilding during Phase 1→Phase 2 transition with unfrozen backbone configuration (08/03/2025)
- [✅] Phase 2 jumping backup model management - Implemented comprehensive backup model system with _phase1 and _phase2 suffixes and proper override logic (08/03/2025)
- [✅] Zero mAP issue despite good validation loss - Root cause identified: Zero mAP is normal in early training when no predictions overlap with ground truth (no true positives). The "No valid statistics" warning was misleading - the pipeline works correctly (08/03/2025)
- [✅] Layer metrics not improving after validation refactoring - Root cause identified: Layer metrics appear as zeros because target filtering assigns most SmartCash targets (classes 0-6) to layer_1, leaving layer_2 and layer_3 with few/no targets. This is expected behavior for the hierarchical class structure (08/03/2025)
- [✅] Phase 2 jumping state propagation issue - Fixed missing phase propagation after model rebuilding in Phase 2 jumping scenarios. Added proper phase state propagation to all model components in both new training and resume modes (08/03/2025)
- [✅] Large class matrix warning in hierarchical_processor - Root cause identified: Memory optimization for class 7 predictions when matrix size exceeds 100K pairs. Enhanced logging to prevent spam and provide context. This is normal behavior for popular classes (08/03/2025)
- [✅] Model history reset when starting fresh training - Fixed MetricsHistoryRecorder to properly clear existing history files when resume_mode=False. Fresh training now removes existing metrics, phase summary, and latest files (08/03/2025)
- [✅] Model history epoch handling for resume mode - Fixed duplicate epoch handling by treating (phase, epoch) as composite primary key. Resume mode now updates existing epoch records instead of appending duplicates (08/03/2025)
- [✅] **high priority** Evaluation metrics summary - Implemented comprehensive metrics with clear distinction: mAP-based metrics (map50, map50_precision, map50_recall, map50_f1) from YOLOv5 training module for object detection, and denomination classification metrics (accuracy, precision, recall, f1) for 7-class classification, plus performance metrics (inference_time, fps) (08/04/2025)
- [✅] **high priority** Confusion matrix for 7 class denomination classification - Created 8×8 confusion matrix including "no detection" class for denomination classification. Shows true detection performance: 759 missed detections vs 2 successful detections with perfect classification on detected samples (08/04/2025)
- [✅] Model output format compatibility with mAP@0.5 computation - Validated that SmartCash model outputs are properly converted by ValidationMapProcessor to YOLOv5 format for accurate mAP evaluation (08/03/2025)
- [✅] Phase 2 prediction format mismatch - Fixed training executor to use full processing for Phase 2 multi-layer predictions, ensuring Dict format for both predictions and targets (08/03/2025)
- [✅] Phase transition model rebuilding - Added proper model rebuilding during Phase 1→Phase 2 transition with unfrozen backbone configuration (08/03/2025)
- [✅] Phase 2 jumping backup model management - Implemented comprehensive backup model system with _phase1 and _phase2 suffixes and proper override logic (08/03/2025)
- [✅] Zero mAP issue despite good validation loss - Root cause identified: Zero mAP is normal in early training when no predictions overlap with ground truth (no true positives). The "No valid statistics" warning was misleading - the pipeline works correctly (08/03/2025)
- [✅] Layer metrics not improving after validation refactoring - Root cause identified: Layer metrics appear as zeros because target filtering assigns most SmartCash targets (classes 0-6) to layer_1, leaving layer_2 and layer_3 with few/no targets. This is expected behavior for the hierarchical class structure (08/03/2025)
- [✅] Phase 2 jumping state propagation issue - Fixed missing phase propagation after model rebuilding in Phase 2 jumping scenarios. Added proper phase state propagation to all model components in both new training and resume modes (08/03/2025)
- [✅] Large class matrix warning in hierarchical_processor - Root cause identified: Memory optimization for class 7 predictions when matrix size exceeds 100K pairs. Enhanced logging to prevent spam and provide context. This is normal behavior for popular classes (08/03/2025)
- [✅] Model history reset when starting fresh training - Fixed MetricsHistoryRecorder to properly clear existing history files when resume_mode=False. Fresh training now removes existing metrics, phase summary, and latest files (08/03/2025)
- [✅] Model history epoch handling for resume mode - Fixed duplicate epoch handling by treating (phase, epoch) as composite primary key. Resume mode now updates existing epoch records instead of appending duplicates (08/03/2025) 
- [✅] **high priority** Efficient metrics discrepancy in Phase 1 - Fixed discrepancy between val_accuracy and layer_1_accuracy in Phase 1 by ensuring proper metric standardization in research_metrics.py (08/04/2025)
