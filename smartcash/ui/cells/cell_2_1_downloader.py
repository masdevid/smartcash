"""
File: smartcash/ui/cells/cell_2_1_downloader.py
Deskripsi: Entry point untuk download dataset
NOTE: Cell Code should remain minimal (import and run initializer only). 
      Initializer should handle all the logic.
"""

from smartcash.ui.dataset.downloader import create_downloader_display
downloader = create_downloader_display()
downloader()
