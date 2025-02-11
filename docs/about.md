# SmartCash 💵

## Objective 🎯
- Enhance the detection accuracy and efficiency of YOLOv5 by utilizing EfficientNet-B4's superior feature extraction capabilities.
- Address challenges related to detecting small details on banknotes, such as nominal values, watermarks, and security symbols under varying conditions like lighting, folds, and damage.

## Proposed Hybrid Approach 🔄
- **EfficientNet-B4 as Backbone**: EfficientNet-B4 replaces YOLOv5's default backbone (CSPDarknet) to enhance feature extraction, particularly for small objects and complex patterns.
- **YOLOv5 as Detection Head**: Maintains the responsibility of object localization by predicting bounding boxes and classifying currency denominations based on the refined feature maps from EfficientNet-B4.

## Workflow 🔧
1. **Input**: Preprocessed banknote images (resized, normalized).
2. **Feature Extraction**: EfficientNet-B4 extracts detailed features.
3. **Neck**: Feature maps are passed to the Feature Pyramid Network (FPN) and PANet for multi-scale feature aggregation.
4. **Head**: YOLOv5's head performs final object detection and classification.
5. **Output**: Predicted bounding boxes, confidence scores, and currency denomination labels.

## Advantages of the Proposed System 🌟
- Improved detection of small features on banknotes.
- Enhanced robustness to varying environmental conditions.
- Increased accuracy and efficiency compared to YOLOv5 alone.
- Potential for real-time application in devices like ATMs, vending machines, and assistive tools for visually impaired users.

## Challenges Addressed 🚧
- False positives and false negatives in detecting small objects.
- Variations in illumination and orientation.
- Complex backgrounds and overlapping objects.

## Implementation Considerations 🛠️
- The dataset used consists of official Rupiah banknotes from Bank Indonesia, with images sourced from RoboFlow.
- EfficientNet-B4 offers compound scaling, optimizing depth, width, and resolution to balance accuracy and computational efficiency.
- The model is designed to operate on resource-limited devices such as mobile applications.

## Conclusion 🏁
The proposed system leverages the strengths of both YOLOv5 and EfficientNet-B4 to create a high-performance currency detection system with enhanced accuracy and robustness across diverse conditions.