"""
File: smartcash/ui/cells/cell_3_3_train.py
Deskripsi: Entry point untuk training model
NOTE: Cell Code should remain minimal (import and run initializer only).
      Initializer should handle all the logic.
"""

from smartcash.ui.model.training import TrainingUIFactory

# Create and display the training UI
# Note: auto_display=False is used to prevent double display since the module handles its own display
TrainingUIFactory.create_and_display_training(auto_display=False)