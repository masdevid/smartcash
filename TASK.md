Updated 22 Juli 2025, 20:
# NOTES:
- Raw Data: `/content/data/{valid, train, test}/{images, labels}` or `/data/{valid, train, test}/{images, labels} (Symlink|Local)`
- Preprocessed Data: `/content/data/preprocessed/{valid, train, test}/{images, labels}` or `/data/preprocessed/{valid, train, test}/{images, labels} (Symlink|Local)`
- Augmented Data: `/content/data/augmented/{valid, train, test}/{images, labels}` or `/data/augmented/{valid, train, test}/{images, labels} (Symlink|Local)`
- Pretrained Data: `/content/datata/pretrained` or `/data/pretrained (Symlink|Local)`
- Checkpoint Data: `/content/data/checkpoints` or `/data/checkpoints (Symlink|Local)`
- Model Data: `/content/data/models` or `/data/models (Symlink|Local)`

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
- [NEW] Fix progress bar issues in Operation Mixin -> Operation Container -> Progress Tracker. In single mode, progress bar not appear. In dual mode, only primary bar updated. Ensure progress tracking integration using correct bar level. There is possibility level correct but progress tracking showing incorrect bar or the opposite. 
- [NEW] enhance logs in `smarcash/common/**`, `smartcash/dataset/**`, `smartcash/model/**` to detect environment. If colab detected, use log to file and disable log to console to prevent logs appear outside operation container in UI module

# COLAB MODULE
- N/A

# DEPENDENCY MODULE
- [COMPLETED] Optimize check missing packages and updates should use threadpool for faster response
- [COMPLETED] The progress tracker message appear but not the progress bar. Might use wrong method to update the progress bar.
- [COMPLETED] Optimize uimodule to properly inherit from base ui module with mixin. Ensure no overlap with config handler. Reduce duplication accross operations by put shared consoslidated methode on parent class or mixin.

# DOWNLOAD MODULE
- [COMPLETED] Dataset cleanup still not triggering confirmation dialog

# PREPROCESSING MODULE
- [COMPLETED] Still getting error `⚠️ Service belum siap - tidak dapat melakukan cleanup. Pastikan direktori data telah dibuat dengan benar.` when the data is actually exist in `/data/preprocessed/{train, valid, test}/{labels, images}`

# SPLIT MODULE
- N/A

# AUGMENTATION MODULE
- [COMPLETED] Summary Container still not showing after operations

# PRETRAINED MODULE
- N/A

# BACKBONE MODULE
- N/A

# TRAINING MODULE
- [COMPLETED] `Cannot start training: No built backbone models found. Please build models in the backbone module first.` when the actual backbone is exist on  `/data/models/backbone_smartcash_efficientnet_b4_20250723_1209.pt (122.0MB)`
- [COMPLETED] Ensure backbone model discovery match backbone module


# EVALUATION MODULE
- N/A 

# VISUALIZATION MODULE

- [COMPLETED] Dashboard Charts still not rendered. Try refactor ui module and ensure  proper inheritance and mixin usage. Reduce duplication to keep it DRY.
- [COMPLETED] TypeError: get_samples() got an unexpected keyword argument 'sample_type' 
