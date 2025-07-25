Updated 22 Juli 2025, 20:
# NOTES:
- Raw Data: `/content/data/{valid, train, test}/{images, labels}` or `/data/{valid, train, test}/{images, labels} (Symlink|Local)`
- Preprocessed Data: `/content/data/preprocessed/{valid, train, test}/{images, labels}` or `/data/preprocessed/{valid, train, test}/{images, labels} (Symlink|Local)`
- Augmented Data: `/content/data/augmented/{valid, train, test}/{images, labels}` or `/data/augmented/{valid, train, test}/{images, labels} (Symlink|Local)`
- Pretrained Data: `/content/datata/pretrained` or `/data/pretrained (Symlink|Local)`
- Checkpoint Data: `/content/data/checkpoints` or `/data/checkpoints (Symlink|Local)`
- Model Data: `/content/data/models` or `/data/models (Symlink|Local)`
## Data Organization:
1. Raw Data (/data/{train,valid,test}/images/):
  - Files with prefix rp_* in regular image formats (.jpg, .png, etc.)
  - Labels as .txt files
2. Augmented Data (/data/augmented/{train,valid,test}/images/):
  - Files with prefix aug_* in regular image formats (before normalization)
  - Labels as .txt files
3. Preprocessed Data (/data/preprocessed/{train,valid,test}/images/):
  - Files with prefix pre_* as .npy files (preprocessed)
  - Files with prefix aug_* as .npy files (augmented after normalization)
  - Labels as .txt files
# GENERAL
- [✅] Create Consistent Summary Format for Downloader, Preprocessing and Augmentation UI Module - All three modules now use MarkdownHTMLFormatter for consistent summary styling and formatting

# COLAB MODULE
- ✅ All Works

# DEPENDENCY MODULE
- [✅] Progress tracker not appearing - Fixed with proper timing delays
- [✅] Install, Uninstall and updates process use synchronous - Already implemented correctly
- [✅] Check missing packages and updates should use threadpool for faster response - Implemented ThreadPoolExecutor for check operations
- [✅] Ensure no overlapping operations - Added operation lock mechanism with proper validation
- [✅] Button states should be disabled during operations - Fixed BaseUIModule to disable all buttons during operations and re-enable after completion
- [✅] Config manager logs leaking outside operation container - Fixed by suppressing config manager console logs in base operation
- [✅] Installation should only install missing packages instead of all packages - Added package checking to only install packages that are not already installed 

# DOWNLOAD MODULE
- [✅] Each operation done should update summary result - Implemented summary container integration with formatted markdown output for all operations (download, check, cleanup) 
- [✅] I've installed markdown package, so use it to generate summary html - Updated all download operations (download, check, cleanup) to use MarkdownHTMLFormatter
- [✅] Need confirmation dialog working on download/cleanup dataset operations on existing data in `/data/{train, valid, test}` - Implemented comprehensive confirmation dialogs for both download and cleanup operations with detailed existing data detection, file count, size information, and proper danger mode warnings

# PREPROCESSING MODULE
- [✅] Create quick backend service readiness check. Add new api on `smartcash/dataset/preprocessor/api` that do quick check (empty or not) on `/dataset/preprocessed/{train, valid, test}/{labels, images}` - Implemented readiness_api.py with comprehensive checks
- [✅] Add post init hook using quick check api and set service ready flag in preprocessing ui module - Added _post_init_tasks method with service readiness detection
- [✅] Do not run download and cleanup operation if service not ready - Updated validation methods to check service readiness
- [✅] If service ready, open confirmation dialog if existing data found - Enhanced confirmation dialogs with existing data warnings
- [✅] Operation summary missing new line formatting - Fixed with core HTML formatter utility 
- [✅] `Service belum siap - tidak dapat melakukan cleanup. Pastikan direktori data telah dibuat dengan benar.` -> false negative. I'm running in colab and data in `/data/{train, valid, test}` is exist. - Fixed by implementing prefix-aware data detection that correctly identifies all data stages
- [✅] Audit quick check update api: - Enhanced readiness API with comprehensive prefix-aware checks:
    - Raw data: `/data/{train, valid, test}/{labels, images}` - detects `rp_*` prefixed regular images
    - Augmented data: `/data/augmented/{train, valid, test}/{labels, images}` - detects `aug_*` prefixed regular images
    - Preprocessed data: `/data/preprocessed/{train, valid, test}/{labels, images}` - detects `pre_*` and `aug_*` prefixed .npy files
    - Service is ready if ANY of the three data stages exist (completely fixing Colab false negative)
- [✅] I've installed markdown package, so use it to generate summary html - Updated module to use MarkdownHTMLFormatter for consistent summary formatting

# SPLIT MODULE
- ✅ All Works

# AUGMENTATION MODULE
- [✅] Summary Container on UI dictionary but not yet updated when operations done. Should be like preprocessing operation summary - Updated to use core HTML formatter utility 
- [✅] I've installed markdown package, so use it to generate summary html - Updated module to use MarkdownHTMLFormatter for consistent summary formatting

# PRETRAINED MODULE
- ✅ All Works

# BACKBONE MODULE
- [✅] Ensure each operation clear operation log first before logging new operation log - Added clear_operation_logs() method to base operation class
- [✅] Ensure Dynamic Button Registration only run once on init and not poluting log during operations - Added _buttons_registered flag to prevent duplicate registration 
- [✅] I've installed markdown package, so use it to generate summary html - Updated module to use MarkdownHTMLFormatter for consistent summary formatting from backend
- [✅] Rescan build models seems has looping bug. Optimize it to discover in paralel using threadpool - Implemented ThreadPoolExecutor with concurrent.futures for parallel model discovery with proper timeouts and error handling
- [✅] It discover zero model while model exist on `/data/models/backbone_smartcash_efficientnet_b4_20250723_1209.pt (122.0MB)` - Enhanced metadata extraction to properly handle both EfficientNet and CSPDarkNet naming patterns
- [✅] Progress tracker stucks on 25% while the whole operations successfull - Fixed dual progress calculation to properly combine completed steps with current step progress
- [✅] Bug: ⚠️ No clear method available on operation container - Added robust clear method handling with multiple fallback approaches (clear_logs, clear_operations, clear, reset, clear_output)
- [✅] Ensure use proper inheritance with base_ui_module and existing mixin - Fixed inheritance order to BaseUIModule, ModelDiscoveryMixin, ModelConfigSyncMixin, BackendServiceMixin with proper MRO and added example method demonstrating mixin usage
- [ ] Do comprehensive operations tests with more edge cases

# TRAINING MODULE
- [✅] Ensure each operation clear operation log first before logging new operation log - Added clear_operation_logs() method to training base operation class
- [✅] Ensure Dynamic Button Registration only run once on init and not poluting log during operations - Added _buttons_registered flag with duplicate prevention 
- [✅] I've installed markdown package, so use it to generate summary html - Updated module to use MarkdownHTMLFormatter for consistent summary formatting from backend
- [✅] Triple progress tracker well integrated with backend progress callback - Enhanced training start operation to use consistent triple progress tracking, added complete_triple_progress and error_triple_progress methods to OperationMixin, and improved backend callback integration
- [✅] Why get build model error when it supposed to be not rebuild the model that done before in bacbone module? - Fixed by implementing proper backbone model discovery with existence verification and manual fallback discovery when backbone module unavailable
    - ✅ Enhanced _get_backbone_configuration() to check for built models
    - ✅ Added _discover_backbone_models_manually() for fallback model discovery
    - ✅ Updated _select_model_from_backbone() to validate model availability before proceeding
- [✅] Prerequisete check seem has performance issue (too slow for checking thins). Optimize it - Replaced synchronous package checking with parallel ThreadPoolExecutor-based checking for significant performance improvement
- [✅] Test optimized training module configuration handler overlap removal - Verified successful config handler overlap removal with proper ConfigurationMixin delegation, removed duplicate methods (save_current_config, reset_to_defaults, _update_module_config), fixed method calls, and ensured all configuration management works through proper inheritance hierarchy



# EVALUATION MODULE
- [✅] Ensure each operation clear operation log first before logging new operation log - Added clear_operation_logs() method to evaluation base operation class and all concrete operations
- [✅] Ensure Dynamic Button Registration only run once on init and not poluting log during operations - Added _buttons_registered flag with duplicate prevention
- [✅] Remove this handler to not triggering unnecesary log from dynamic buttons registration - Commented out missing button handlers for 'stop_evaluation' and 'export_results'
- [✅] I've installed markdown package, so use it to generate summary html - Updated module to use MarkdownHTMLFormatter for consistent summary formatting from backend
- [NEW] TypeError: get_samples() got an unexpected keyword argument 'sample_type' - [NEW] Dashboard Charts still not rendered. Try refactor ui module and ensure  proper inheritance and mixin usage. Reduce duplication to keep it DRY.

# VISUALIZATION MODULE
- [✅] Dashboard cards still not appearing. Only blank container - Fixed container layout issues, improved placeholder data initialization, and simplified dashboard card creation process