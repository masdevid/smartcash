
# Low mAP Investigation Summary

## 1. Problem Description

The model is exhibiting a very low mAP score during Phase 1 of training, even at epoch 11. This investigation aims to determine the root cause of the low mAP by analyzing the debug logs.

## 2. Investigation Summary

The investigation involved analyzing the following log files for epoch 11:

- `logs/validation_metrics/efficientnet_b4/hierarchical_debug_phase1_epoch11.log`
- `logs/validation_metrics/efficientnet_b4/map_debug_phase1_epoch11.log`

The analysis of these logs revealed several critical issues that are contributing to the low mAP score.

### 2.1. Key Findings

1.  **Extremely Low Precision**: The overall precision is a mere **0.0308**, indicating that the vast majority of the model's predictions are incorrect. Out of 33,119 predictions, only 1,019 were true positives.

2.  **Low Prediction Confidence**: The model's confidence in its predictions is extremely low. The average confidence score hovers around 0.01-0.02, and at a confidence threshold of 0.2, there are no true positives. This suggests the model has not yet learned to distinguish the target objects effectively.

3.  **High Number of False Positives**: The per-class analysis shows a massive number of false positives. For instance, Class 0 has over 5,000 false positives for only 32 ground truth targets. This indicates the model is making a large number of incorrect detections.

4.  **Bug in Recall Calculation**: The recall metric is being calculated incorrectly, resulting in values greater than 1.0, which is mathematically impossible. This bug is present in the mAP calculation script and is skewing the F1 score and other related metrics.

## 3. Conclusion

The low mAP is a direct result of the model's poor performance at this stage of training, characterized by a high volume of false positives and low prediction confidence. The issue is compounded by a bug in the recall calculation, which prevents an accurate assessment of the model's performance.

## 4. Recommended Actions

1.  **Fix the Recall Calculation Bug**: The most immediate action is to debug and fix the recall calculation in the mAP script. This is essential for accurate model evaluation.

2.  **Hyperparameter Review**: The `conf_thres` and `iou_thres` hyperparameters should be reviewed and tuned. The current values are likely too low and are contributing to the high number of false positives.

3.  **Continue Training**: Since the model is still in the early stages of Phase 1 training, it is expected that the performance will be low. It is recommended to continue training for more epochs to allow the model to learn more robust features.

4.  **Review Data Augmentation**: The data augmentation pipeline should be reviewed to ensure that it is not creating overly complex or unrealistic training examples that are hindering the model's ability to learn.
