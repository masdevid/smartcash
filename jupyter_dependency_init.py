"""
File: jupyter_dependency_init.py
Deskripsi: Script untuk menjalankan inisialisasi dependency UI di Jupyter.
"""

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from smartcash.ui.setup.dependency.dependency_initializer import initialize_dependency_ui

# Run initialization
result = initialize_dependency_ui()
print("Hasil inisialisasi dependency UI:")
print(f"Success: {result['success']}")
print(f"UI Components: {list(result['ui_components'].keys()) if result['ui_components'] else 'None'}")
print(f"Module Handler: {result['module_handler'] is not None}")
print(f"Config Available: {'config' in result}")
print(f"Operation Handlers: {list(result['operation_handlers'].keys()) if result['operation_handlers'] else 'None'}")
