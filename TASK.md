# Refactoring Tasks

## 🚀 Development Priorities

### ✅ Recently Completed (July 19, 2025)
1. **Enhanced YOLOv5 Model Building Implementation**
   - ✅ **EfficientNet-B4 Backbone Builder**: Enhanced with multi-layer detection support and model building capabilities
   - ✅ **CSPDarknet Backbone Builder**: Enhanced with multi-layer detection support and YOLO integration
   - ✅ **Multi-Layer Detection Heads**: Implemented 3-layer detection system (layer_1, layer_2, layer_3) with attention mechanisms
   - ✅ **Uncertainty-based Multi-Task Loss**: Implemented based on Kendall et al. (Google DeepMind) with dynamic weighting
   - ✅ **Complete Model Builder Service**: Integrated backbone, neck, and heads with comprehensive build pipeline
   - ✅ **Backbone UI Module Enhancement**: Updated to support new model building functionality with enhanced summaries
   - ✅ **Comprehensive Unit Tests**: Created with dummy data testing for all new components (17/20 tests passing)

### Architecture Implementation Details
**Multi-Layer Detection System**:
- **Layer 1**: Full banknote detection (7 classes: 001, 002, 005, 010, 020, 050, 100)
- **Layer 2**: Nominal-defining features (7 classes: l2_001, l2_002, l2_005, l2_010, l2_020, l2_050, l2_100)  
- **Layer 3**: Common features (3 classes: l3_sign, l3_text, l3_thread)

**Training Strategy**:
- **Phase 1**: Freeze backbone, train detection heads only (Learning Rate: 1e-3)
- **Phase 2**: Unfreeze entire model for fine-tuning (Learning Rate: 1e-5)
- **Loss Function**: Uncertainty-based Multi-Task Loss dengan dynamic weighting

**Model Builder Features**:
- Support for both EfficientNet-B4 and CSPDarknet backbones
- Automated FPN-PAN neck integration
- Multi-layer head configuration
- Phase-based training preparation
- Comprehensive model information and statistics

### Immediate Next Steps (High Priority)
1. **Apply BaseUIModule Pattern to Remaining Modules**
   - Migrate remaining 19 modules to use BaseUIModule pattern (backbone ✅ completed)
   - Follow standardized refactoring checklist in `UI_MODULE_REFACTORING.md`
   - **Next Targets**: downloader, preprocess, augment, pretrained, training, evaluation, visualization

2. **Model Evaluation Module Refactoring**  
   - Complete the model management trilogy after backbone ✅ and training
   - Use new BaseUIModule pattern for consistency
   - Integration with backend evaluation services and new multi-layer models

3. **Data Pipeline Enhancement**
   - Apply BaseUIModule pattern to preprocess and augment modules
   - Implement consistent container architecture
   - Focus on essential functionality with standardized patterns

### Medium Priority
1. **Forms Refactoring**
   - Fix overlapping forms
   - Rearrange form groups and layout to save vertical space
   - Cleanup Obsolete Legacy Codes

### Long-term Goals
1. **Complete Migration to BaseUIModule Pattern**
   - All 22+ remaining modules migrated to BaseUIModule pattern
   - Remove legacy code patterns and deprecated mixins
   - Unified codebase with 90% less duplication
