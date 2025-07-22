"""
File: smartcash/ui/cells/cell_1_2_colab.py
Deskripsi: Entry point untuk Google Colab environment setup
NOTE: Cell Code should remain minimal (import and run initializer only). 
      Initializer should handle all the logic.
"""

from smartcash.ui.setup.colab import create_colab_display
colab = create_colab_display()
colab()