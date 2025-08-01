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
- Fixed TypeError in core.py - August 1, 2025
  - Root cause: The `run_full_training_pipeline` function in `core.py` was not passing the `patience` argument to the `TrainingPipeline`'s `run_full_training_pipeline` method.
  - Solution: Modified the function to extract the `patience` argument from the `kwargs` and pass it to the method.
- Fixed SyntaxError in training_pipeline.py - August 1, 2025
  - Root cause: A parameter without a default value (`patience`) was following parameters with default values in the `run_full_training_pipeline` function.
  - Solution: Moved the `patience` parameter to be the first argument after `self`.
- Fixed issue where `--patience` argument was being overridden by a default value - August 1, 2025
  - Root cause: The `run_full_training_pipeline` function in `training_pipeline.py` had a default value for the `patience` argument, which was overriding the value passed from the command line.
  - Solution: Removed the default value from the function definition.
- Fixed issue where `--patience` argument was being overridden by a default value - August 1, 2025
  - Root cause: The `apply_configuration_overrides` function in `setup_utils.py` was not correctly handling the `patience` argument, causing it to be overridden by a default value.
  - Solution: Modified the `apply_configuration_overrides` function to correctly handle the `patience` argument.
- Fixed issue where `--patience` argument was being overridden by a default value - August 1, 2025
  - Root cause: The training pipeline was prioritizing a default value over the command-line argument.
  - Solution: Modified the code to prioritize the `--patience` argument.
- Fixed AttributeError in progress_manager.py when early_stopping is None - August 1, 2025
  - Root cause: handle_early_stopping function did not check if early_stopping object was None before accessing its attributes. The create_early_stopping factory function in early_stopping.py could also return None.
  - Solution: Modified create_early_stopping to always return a valid (dummy) object. Added a check in handle_early_stopping to return False if early_stopping is None.
- Fixed validation metrics stagnation issue (val_precision, val_recall, val_f1, val_accuracy) - July 31, 2025
  - Root cause: Model's current_phase was not being propagated to child modules (especially MultiLayerHead)
  - Solution: Added _propagate_phase_to_children() method in PhaseOrchestrator to ensure all modules receive phase info
  - Updated research metrics system to force zero values instead of static fallbacks when layer metrics missing
- Fixed ERROR - smartcash.model.training.core.validation_executor - Error computing loss in validation batch 19: Unexpected prediction shape: torch.Size([25200, 22]) - July 31, 2025
# EVALUATION MODULE
- N/A
# VISUALIZATION MODULE
- N/A