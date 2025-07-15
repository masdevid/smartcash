# Refactoring Tasks

## 🚀 Development Priorities

### Immediate Next Steps (High Priority)
1. **Apply BaseUIModule Pattern to Remaining Modules**
   - Migrate remaining 20 modules to use BaseUIModule pattern
   - Follow standardized refactoring checklist in `UI_MODULE_REFACTORING.md`
   - **Next Targets**: downloader, preprocess, augment, pretrained, backbone, training, evaluation, visualization

2. **Model Evaluation Module Refactoring**
   - Complete the model management trilogy after backbone and training
   - Use new BaseUIModule pattern for consistency
   - Integration with backend evaluation services

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
