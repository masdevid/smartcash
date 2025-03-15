"""
File: smartcash/dataset/services/loader/batch_generator.py
Deskripsi: Generator batch data untuk dataset dengan dukungan prefetching dan paralelisasi
"""

import threading
import queue
import time
from typing import List, Dict, Any, Callable, Optional, Tuple, Iterator

import torch
from torch.utils.data import Dataset, DataLoader
from smartcash.common.logger import get_logger


class BatchGenerator:
    """
    Generator batch data yang dioptimalkan dengan prefetching dan multi-threading
    untuk meningkatkan throughput data pipeline training.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        timeout: float = 120.0,
        collate_fn: Optional[Callable] = None,
        logger=None
    ):
        """
        Inisialisasi BatchGenerator.
        
        Args:
            dataset: Dataset yang akan digunakan
            batch_size: Ukuran batch
            shuffle: Apakah mengacak data
            num_workers: Jumlah worker untuk proses paralel
            pin_memory: Apakah men-pin memory untuk transfer GPU yang lebih cepat
            drop_last: Apakah membuang batch terakhir jika tidak lengkap
            prefetch_factor: Faktor prefetch (jumlah batch yang di-prefetch per worker)
            timeout: Batas waktu tunggu dalam detik
            collate_fn: Fungsi kustom untuk menggabungkan item menjadi batch
            logger: Logger kustom (opsional)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.collate_fn = collate_fn
        self.logger = logger or get_logger("batch_generator")
        
        # Buat dataloader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
        
        # Setup prefetch queue
        self.queue_size = max(2, prefetch_factor * num_workers)
        self.prefetch_queue = queue.Queue(maxsize=self.queue_size)
        
        # Flag untuk kontrol
        self._stop_event = threading.Event()
        self._prefetch_thread = None
        self._current_epoch = 0
        self._iterations = 0
        
        self.logger.debug(
            f"🔄 BatchGenerator diinisialisasi:\n"
            f"   • Batch size: {batch_size}\n"
            f"   • Shuffle: {shuffle}\n"
            f"   • Num workers: {num_workers}\n"
            f"   • Prefetch factor: {prefetch_factor}\n"
            f"   • Queue size: {self.queue_size}"
        )
    
    def __len__(self) -> int:
        """Mendapatkan jumlah batch dalam satu epoch."""
        return len(self.dataloader)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Memulai iterasi batch.
        
        Returns:
            Iterator untuk batch data
        """
        # Reset flag
        self._stop_event.clear()
        
        # Tambahkan epoch counter
        self._current_epoch += 1
        
        # Mulai thread prefetch jika belum berjalan
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._start_prefetch_thread()
        
        # Yield batch dari queue
        try:
            num_batches = len(self.dataloader)
            for i in range(num_batches):
                try:
                    # Tunggu dan ambil batch dari queue
                    batch = self.prefetch_queue.get(timeout=self.timeout)
                    
                    if batch is None:
                        # Sentinel value, berarti prefetch selesai
                        break
                    
                    self._iterations += 1
                    yield batch
                    
                except queue.Empty:
                    self.logger.warning(f"⚠️ Timeout menunggu batch (> {self.timeout}s)")
                    break
                    
        except Exception as e:
            self.logger.error(f"❌ Error saat iterasi batch: {str(e)}")
            self._stop_event.set()  # Signal prefetch thread untuk berhenti
            
        finally:
            pass