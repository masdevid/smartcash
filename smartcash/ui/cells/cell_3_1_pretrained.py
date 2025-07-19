"""
File: smartcash/ui/cells/cell_3_1_pretrained.py
Deskripsi: Entry point untuk model pretrained
NOTE: Cell Code should remain minimal (import and run factory only). 
      Factory handles all the UI creation and display logic.
"""

from smartcash.ui.model.pretrained.pretrained_ui_factory import PretrainedUIFactory

# Create and display the pretrained UI using the factory
# This maintains the same interface as before but uses the new factory pattern
PretrainedUIFactory.create_and_display_pretrained()
