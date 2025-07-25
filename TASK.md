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

# COLAB MODULE
- N/A

# DEPENDENCY MODULE

# DOWNLOAD MODULE
- [NEW] Dataset cleanup on existing data still has issue, it's triggering "✅ Pembersihan dataset berhasil diselesaikan 2" when it should show confirmation dialog.

# PREPROCESSING MODULE
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
- N/A

# TRAINING MODULE
- N/A

# EVALUATION MODULE
- N/A 

# VISUALIZATION MODULE

- [NEW] `File "/content/smartcash/smartcash/ui/dataset/visualization/visualization_uimodule.py", line 438, in _render_preprocessed_samples raise ValueError("Visualization container not found") ValueError: Visualization container not found` -> container is in `smartcash/ui/components/visualization_container.py`
- [NEW] `File "/content/smartcash/smartcash/ui/dataset/visualization/visualization_uimodule.py", line 390, in update_visualization self.update_operation_status(error_msg, "error"). AttributeError: 'VisualizationUIModule' object has no attribute 'update_operation_status'` -> still referenceing old method
- [NEW] Optimize uimodule, reduce duplication. Do comprehensive unit and integration test. 
