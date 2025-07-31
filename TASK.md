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
- N/A
# COLAB MODULE
- N/A
# DEPENDENCY MODULE
- N/A
# DOWNLOAD MODULE
- N/A
# PREPROCESSING MODULE
- N/A
# AUGMENTATION MODULE
- N/A
# TRAINING MODULE
- Fixed validation metrics stagnation issue (val_precision, val_recall, val_f1, val_accuracy) - July 31, 2025
  - Root cause: Model's current_phase was not being propagated to child modules (especially MultiLayerHead)
  - Solution: Added _propagate_phase_to_children() method in PhaseOrchestrator to ensure all modules receive phase info
  - Updated research metrics system to force zero values instead of static fallbacks when layer metrics missing
# EVALUATION MODULE
- N/A
# VISUALIZATION MODULE
- N/A