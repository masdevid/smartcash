Updated 22 Juli 2025, 20:
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

# GENERAL
- [COMPLETED] Data Statistics count on Backend used by Downloader, Preprocessing and Augmentation UI Module is incorrect. 
- [COMPLETED] Ensure it calculate the right file explained in `Data Organization`.
- [COMPLETED] Ensure preprocessing count differentiate between raw_preprocessed (pre_*) and augmented_preprocessed (aug_*)
- [COMPLETED] Move Summary Container in Downloader, Preprocessing and Augmentation UI Module
- [COMPLETED] Fix issue when discover/refresh model on backbone, training, and evaluation module. When event triggered, it polutes operation logs, sometimes it also triggering dynamic button registration logs during discovery/refresh process. 
- [COMPLETED] Fix progress bar issues in Operation Mixin -> Operation Container -> Progress Tracker. In single mode, progress bar not appear. In dual mode, only primary bar updated. Ensure progress tracking integration using correct bar level. There is possibility level correct but progress tracking showing incorrect bar or the opposite. 
- [NEW] Clear summary content each time operation executed on Downloader, Preprocessing and Augmentation UI Module.
- [NEW] Summary Result between Downloader, Preprocessing and Augmentation Module do not match. I suspect it's because incomplete information context on summary returned by backend. Use consistent summary format accross modules, returning total and per split stats. Include data path too.
  - Downloader:
    Total Files: 📁 4,568 file
    Total Images: 🖼️ 2,284
    Total Labels: 🏷️ 2,284
  - Preprocessing:
    Total Raw Files: 📁 1599 file (looks like returning train split only)
    Total Preprocessed Images: 🖼️ 1599
  - Augmentation:
    TRAIN: Raw=1599(available), Aug=2274(available), Prep=6607(available)
    VALID: Raw=343(available), Aug=0(not_found), Prep=343(available)
    TEST: Raw=342(available), Aug=0(not_found), Prep=342(available)
- [COMPLETED] enhance logs in `smarcash/common/**`, `smartcash/dataset/**`, `smartcash/model/**` to detect environment. If colab detected, use log to file and disable console logs 
599
# COLAB MODULE
- N/A

# DEPENDENCY MODULE
- [COMPLETED] Optimize check missing packages and updates should use threadpool for faster response
- [COMPLETED] The progress tracker message appear but not the progress bar. Might use wrong method to update the progress bar.
- [COMPLETED] Optimize uimodule to properly inherit from base ui module with mixin. Ensure no overlap with config handler. Reduce duplication accross operations by put shared consoslidated methode on parent class or mixin.

# DOWNLOAD MODULE
- [COMPLETED] Dataset cleanup still not triggering confirmation dialog
- [NEW] Dataset cleanup on existing data still has issue, it's triggering "✅ Pembersihan dataset berhasil diselesaikan 2" when it should show confirmation dialog.

# PREPROCESSING MODULE
- [COMPLETED] Still getting error `⚠️ Service belum siap - tidak dapat melakukan cleanup. Pastikan direktori data telah dibuat dengan benar.` when the data is actually exist in `/data/preprocessed/{train, valid, test}/{labels, images}`
- [NEW] On Cleanup, got `Service belum siap - tidak dapat melakukan cleanup. Pastikan direktori data telah dibuat dengan benar.` when I already run check event with service ready returned from backend. Ensure post init execution is correct so i don't need to run check anytime to make service status ready. 
- [NEW] Failed to set error progress: 'OperationContainer' object has no attribute 'log_debug'
# SPLIT MODULE
- N/A

# AUGMENTATION MODULE
- [COMPLETED] Summary Container still not showing after operations
- [NEW] Failed to set error progress: 'OperationContainer' object has no attribute 'log_debug'

# PRETRAINED MODULE
- N/A

# BACKBONE MODULE
- [NEW]  Model rescan completed - found 0 models, when the actual backbone is exist on  `/data/models/backbone_smartcash_efficientnet_b4_20250723_1209.pt (122.0MB)`. It should use manual search like training module use:
```
Manual search results: data/models/*backbone*smartcash*.pt -> 1 files; data/models/*backbone*smartcash*.pth -> 0 files; data/models/backbone_*.pt -> 1 files; data/models/backbone_*.pth -> 0 files; data/checkpoints/*backbone*.pt -> 0 files; data/checkpoints/*backbone*.pth -> 0 files
```

# TRAINING MODULE
- [COMPLETED] `Cannot start training: No built backbone models found. Please build models in the backbone module first.` when the actual backbone is exist on  `/data/models/backbone_smartcash_efficientnet_b4_20250723_1209.pt (122.0MB)`
- [COMPLETED] Ensure backbone model discovery match backbone module
- [NEW] ERROR - Missing required section: training on Refresh Config
- [NEW] ⚠️ No clear method available on training operation container
- [NEw] Training cannot start. Missing required packages: scikit-learn. Please install them using the Dependency module. -> This is weird, on dependency module reported it's installed already. 
- [NEW] Failed to save configuration: 'TrainingConfigHandler' object has no attribute 'extract_config_from_ui'
- [NEW] Make batch step = 1, currently it prevent me to set batch to 32 because of it. 
# EVALUATION MODULE
- N/A 

# VISUALIZATION MODULE

- [COMPLETED] Dashboard Charts still not rendered. Try refactor ui module and ensure  proper inheritance and mixin usage. Reduce duplication to keep it DRY.
- [COMPLETED] TypeError: get_samples() got an unexpected keyword argument 'sample_type' 
- [NEW] `File "/content/smartcash/smartcash/ui/dataset/visualization/visualization_uimodule.py", line 438, in _render_preprocessed_samples raise ValueError("Visualization container not found") ValueError: Visualization container not found` -> container is in `smartcash/ui/components/visualization_container.py`
- [NEW] `File "/content/smartcash/smartcash/ui/dataset/visualization/visualization_uimodule.py", line 390, in update_visualization self.update_operation_status(error_msg, "error"). AttributeError: 'VisualizationUIModule' object has no attribute 'update_operation_status'` -> still referenceing old method
- [NEW] Optimize uimodule, reduce duplication. Do comprehensive unit and integration test. 

