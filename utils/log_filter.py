# File: smartcash/utils/log_filter.py
# Author: Alfrida Sabar
# Deskripsi: Filter log untuk meminimalisir output yang tidak perlu dari library PyTorch

import logging
from typing import List, Optional

class PyTorchLogFilter:
    """Filter logging untuk PyTorch dan library lainnya."""
    
    # Daftar module yang akan difilter
    FILTERED_MODULES = [
        'torch.distributed.nn.jit.instantiator',
        'torch._dynamo.utils',
        'torch._inductor',
        'torch.distributed',
        'torch._dynamo',
        'torch._C',
        'torch.cuda',
        'torch._subclasses',
        'torch._functorch'
    ]
    
    @staticmethod
    def setup(
        log_level: str = 'INFO', 
        log_file: Optional[str] = 'logs/smartcash.log',
        filtered_modules: Optional[List[str]] = None
    ) -> None:
        """
        Setup konfigurasi logging untuk menyaring log yang tidak diinginkan.
        
        Args:
            log_level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path file log (None untuk disable file logging)
            filtered_modules: Module tambahan yang akan difilter
        """
        # Setup level numerik
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Konfigurasi logging dasar
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Tambahkan file handler jika log_file dispecify
        if log_file:
            # Pastikan direktori log ada
            import os
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Setup file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Tambahkan ke root logger
            logging.getLogger().addHandler(file_handler)
        
        # Gabungkan modul yang difilter
        modules_to_filter = PyTorchLogFilter.FILTERED_MODULES.copy()
        if filtered_modules:
            modules_to_filter.extend(filtered_modules)
            
        # Set level WARNING untuk modul yang difilter
        for module in modules_to_filter:
            logging.getLogger(module).setLevel(logging.WARNING)