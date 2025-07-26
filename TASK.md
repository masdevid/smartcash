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

# COLAB MODULE
- N/A

# DEPENDENCY MODULE

# DOWNLOAD MODULE


# PREPROCESSING MODULE

# SPLIT MODULE
- N/A

# AUGMENTATION MODULE


# PRETRAINED MODULE
- N/A

# BACKBONE MODULE
- N/A

# TRAINING MODULE
- N/A

# EVALUATION MODULE
- N/A 

# VISUALIZATION MODULE

