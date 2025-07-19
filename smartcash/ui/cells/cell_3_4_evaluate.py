"""
File: smartcash/ui/cells/cell_3_4_evaluate.py
Deskripsi: Entry point untuk evaluasi model
NOTE: Cell Code should remain minimal (import and run initializer only). 
      Initializer should handle all the logic.
"""

from smartcash.ui.model.evaluation import EvaluationUIFactory

# Create and display the evaluation UI
# Note: auto_display=False is used to prevent double display since the module handles its own display
EvaluationUIFactory.create_and_display_evaluation(auto_display=False)
