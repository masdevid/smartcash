# File: run.py
# Author: Alfrida Sabar
# Deskripsi: Legacy entry point script dengan warning penggunaan

import warnings
from smartcash.__main__ import main

if __name__ == '__main__':
    warnings.warn(
        "run.py akan dihapus di versi mendatang. "
        "Gunakan 'python -m smartcash' sebagai gantinya.",
        DeprecationWarning,
        stacklevel=2
    )
    main()