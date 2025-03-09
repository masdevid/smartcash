# File: smartcash/handlers/dataset/core/download_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager untuk mengelola proses download file dengan dukungan parallel, resume, dan retry

import os
import json
import shutil
import requests
import zipfile
import threading
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger

class DownloadManager:
    """
    Manager untuk mengelola proses download dataset.
    Mendukung parallel download, resume, dan retry otomatis.
    """
    
    def __init__(
        self,
        output_dir: Path,
        num_workers: int = 4,
        chunk_size: int = 8192,
        retry_limit: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        logger=None
    ):
        """
        Inisialisasi DownloadManager.
        
        Args:
            output_dir: Direktori output
            num_workers: Jumlah worker untuk download paralel
            chunk_size: Ukuran chunk untuk streaming download
            retry_limit: Jumlah percobaan ulang jika gagal
            retry_delay: Delay antar percobaan ulang (detik)
            timeout: Timeout koneksi (detik)
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logger or get_logger("download_manager")
        
        # Buat direktori output jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Direktori untuk metadata dan file sementara
        self.temp_dir = output_dir / ".temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Lock untuk akses ke file
        self.file_locks = {}
        
        self.logger.info(
            f"🔽 DownloadManager diinisialisasi:\n"
            f"   • Output dir: {output_dir}\n"
            f"   • Workers: {num_workers}\n"
            f"   • Retry limit: {retry_limit}"
        )
    
    def download_file(
        self,
        url: str,
        output_path: Union[str, Path],
        resume: bool = True,
        progress_bar: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Download file tunggal dengan dukungan resume.
        
        Args:
            url: URL file yang akan didownload
            output_path: Path tujuan file
            resume: Jika True, coba resume download jika file sudah ada
            progress_bar: Tampilkan progress bar
            metadata: Metadata tambahan untuk disimpan
            
        Returns:
            Path file yang didownload
        """
        output_path = Path(output_path)
        
        # Buat direktori tujuan jika belum ada
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Generate file ID untuk lock dan resume info
        file_id = self._generate_file_id(url, output_path)
        
        # Check dan buat lock untuk file ini jika belum ada
        if file_id not in self.file_locks:
            self.file_locks[file_id] = threading.Lock()
        
        lock = self.file_locks[file_id]
        
        # Metadata untuk resume
        resume_info_path = self.temp_dir / f"{file_id}.json"
        
        with lock:
            # Cek apakah file sudah ada dan lengkap
            if output_path.exists() and resume:
                # Cek apakah file sudah lengkap dengan membaca info resume
                if resume_info_path.exists():
                    try:
                        with open(resume_info_path, 'r') as f:
                            info = json.load(f)
                        
                        if info.get('completed', False) and info.get('size', 0) == output_path.stat().st_size:
                            self.logger.info(f"✅ File sudah ada dan lengkap: {output_path}")
                            return output_path
                    except (json.JSONDecodeError, FileNotFoundError):
                        # Info file rusak atau tidak ada, lanjutkan download
                        pass
            
            # Posisi awal untuk resume
            start_byte = 0
            
            # Cek apakah file sudah ada sebagian untuk resume
            temp_file_path = self.temp_dir / f"{file_id}.part"
            
            if temp_file_path.exists() and resume:
                # Cek info resume
                if resume_info_path.exists():
                    try:
                        with open(resume_info_path, 'r') as f:
                            info = json.load(f)
                        
                        # Cek validitas ukuran file dan waktu download
                        if (
                            time.time() - info.get('timestamp', 0) < 24 * 60 * 60 and  # Tidak lebih dari 24 jam
                            temp_file_path.stat().st_size == info.get('downloaded', 0)
                        ):
                            start_byte = temp_file_path.stat().st_size
                            self.logger.info(f"🔄 Melanjutkan download dari byte {start_byte}: {output_path}")
                    except (json.JSONDecodeError, FileNotFoundError):
                        # Info file rusak atau tidak ada, mulai dari awal
                        start_byte = 0
            
            # Buat atau update info resume
            resume_info = {
                'url': url,
                'output_path': str(output_path),
                'timestamp': time.time(),
                'downloaded': start_byte,
                'completed': False
            }
            
            if metadata:
                resume_info['metadata'] = metadata
            
            # Simpan info resume
            with open(resume_info_path, 'w') as f:
                json.dump(resume_info, f)
            
            # Download file
            try:
                # Setup header untuk resume
                headers = {}
                if start_byte > 0:
                    headers['Range'] = f'bytes={start_byte}-'
                
                # Buat request dengan retry
                for attempt in range(self.retry_limit):
                    try:
                        with requests.get(
                            url, 
                            stream=True, 
                            headers=headers, 
                            timeout=self.timeout
                        ) as response:
                            # Cek response
                            response.raise_for_status()
                            
                            # Dapatkan ukuran total dan ukuran konten saat ini
                            total_size = int(response.headers.get('content-length', 0))
                            
                            if start_byte > 0 and response.status_code == 206:
                                # Jika resume berhasil (status 206 Partial Content)
                                content_range = response.headers.get('content-range', '')
                                if content_range:
                                    try:
                                        total_size = int(content_range.split('/')[-1])
                                    except (ValueError, IndexError):
                                        total_size = 0
                            
                            # Persiapkan progress bar
                            progress = None
                            if progress_bar:
                                progress = tqdm(
                                    total=total_size + start_byte if total_size > 0 else None,
                                    initial=start_byte,
                                    unit='B',
                                    unit_scale=True,
                                    desc=f"Downloading {output_path.name}"
                                )
                            
                            # Buka file untuk append jika resume, atau write jika baru
                            mode = 'ab' if start_byte > 0 else 'wb'
                            with open(temp_file_path, mode) as f:
                                downloaded = start_byte
                                
                                # Download dengan chunks
                                for chunk in response.iter_content(chunk_size=self.chunk_size):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        
                                        # Update progress bar
                                        if progress:
                                            progress.update(len(chunk))
                                        
                                        # Update resume info secara periodik
                                        if downloaded % (self.chunk_size * 100) == 0:
                                            resume_info['downloaded'] = downloaded
                                            resume_info['timestamp'] = time.time()
                                            
                                            with open(resume_info_path, 'w') as info_file:
                                                json.dump(resume_info, info_file)
                            
                            # Selesaikan progress bar
                            if progress:
                                progress.close()
                            
                            # Download selesai, pindahkan file ke lokasi akhir
                            shutil.move(temp_file_path, output_path)
                            
                            # Update dan simpan resume info
                            resume_info['completed'] = True
                            resume_info['size'] = output_path.stat().st_size
                            resume_info['timestamp'] = time.time()
                            
                            with open(resume_info_path, 'w') as info_file:
                                json.dump(resume_info, info_file)
                            
                            # Log sukses
                            self.logger.info(f"✅ Download selesai: {output_path}")
                            
                            return output_path
                            
                    except (requests.RequestException, IOError) as e:
                        if attempt < self.retry_limit - 1:
                            delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                            self.logger.warning(f"⚠️ Download gagal, mencoba lagi dalam {delay:.1f}s: {str(e)}")
                            time.sleep(delay)
                        else:
                            raise
                
                # Jika sampai di sini, semua percobaan gagal
                raise RuntimeError(f"Gagal mendownload setelah {self.retry_limit} percobaan")
                
            except Exception as e:
                self.logger.error(f"❌ Download gagal: {str(e)}")
                
                # Simpan error info
                resume_info['error'] = str(e)
                resume_info['timestamp'] = time.time()
                
                with open(resume_info_path, 'w') as info_file:
                    json.dump(resume_info, info_file)
                
                raise
    
    def download_files(
        self,
        files: List[Dict[str, Any]],
        show_progress: bool = True,
        resume: bool = True
    ) -> List[Path]:
        """
        Download beberapa file secara paralel.
        
        Args:
            files: List file yang akan didownload
                [{'url': url, 'output_path': path, 'metadata': {optional metadata}}]
            show_progress: Tampilkan progress bar
            resume: Jika True, coba resume download jika file sudah ada
            
        Returns:
            List path file yang berhasil didownload
        """
        if not files:
            return []
        
        self.logger.info(f"🔽 Memulai download {len(files)} file dengan {self.num_workers} workers")
        
        # Setup progress bar keseluruhan
        overall_progress = None
        if show_progress:
            overall_progress = tqdm(
                total=len(files),
                desc="Total progress",
                unit="file"
            )
        
        results = []
        failures = []
        
        # Download dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit jobs
            future_to_file = {}
            for file_info in files:
                url = file_info['url']
                output_path = file_info['output_path']
                metadata = file_info.get('metadata')
                
                future = executor.submit(
                    self._download_with_retry,
                    url,
                    output_path,
                    resume,
                    show_progress,
                    metadata
                )
                future_to_file[future] = file_info
            
            # Proses hasil saat selesai
            for future in future_to_file:
                file_info = future_to_file[future]
                try:
                    path = future.result()
                    results.append(path)
                    
                    # Update progress
                    if overall_progress:
                        overall_progress.update(1)
                        
                except Exception as e:
                    failures.append({
                        'file': file_info,
                        'error': str(e)
                    })
                    
                    # Update progress
                    if overall_progress:
                        overall_progress.update(1)
        
        # Tutup progress bar
        if overall_progress:
            overall_progress.close()
        
        # Log hasil
        if failures:
            self.logger.warning(
                f"⚠️ {len(failures)}/{len(files)} file gagal didownload. "
                f"Gunakan resume=True untuk mencoba lagi."
            )
        
        if results:
            self.logger.success(f"✅ Berhasil mendownload {len(results)}/{len(files)} file")
        
        return results
    
    def _download_with_retry(
        self, 
        url: str, 
        output_path: Union[str, Path],
        resume: bool,
        progress_bar: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Fungsi helper untuk download dengan retry untuk ThreadPoolExecutor.
        
        Args:
            url: URL file
            output_path: Path output
            resume: Resume download
            progress_bar: Tampilkan progress bar
            metadata: Metadata tambahan
            
        Returns:
            Path file yang didownload
        """
        try:
            return self.download_file(
                url=url,
                output_path=output_path,
                resume=resume,
                progress_bar=progress_bar,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"❌ Gagal mendownload {url}: {str(e)}")
            raise
    
    def _generate_file_id(self, url: str, output_path: Path) -> str:
        """
        Generate ID unik untuk file berdasarkan URL dan path.
        
        Args:
            url: URL file
            output_path: Path output
            
        Returns:
            String ID file
        """
        # Gunakan hash dari kombinasi URL dan path
        hash_input = f"{url}_{str(output_path)}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def get_download_status(self) -> Dict[str, Any]:
        """
        Dapatkan status download untuk semua file.
        
        Returns:
            Dict berisi status download
        """
        status = {
            'completed': [],
            'in_progress': [],
            'failed': []
        }
        
        # Cek semua file info di temp dir
        for info_file in self.temp_dir.glob('*.json'):
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                file_status = {
                    'url': info.get('url', ''),
                    'output_path': info.get('output_path', ''),
                    'downloaded': info.get('downloaded', 0),
                    'timestamp': info.get('timestamp', 0),
                    'metadata': info.get('metadata', {})
                }
                
                if info.get('completed', False):
                    status['completed'].append(file_status)
                elif 'error' in info:
                    file_status['error'] = info['error']
                    status['failed'].append(file_status)
                else:
                    status['in_progress'].append(file_status)
                    
            except (json.JSONDecodeError, IOError):
                # Skip file yang rusak
                continue
        
        return status
    
    def resume_all_failed(self, show_progress: bool = True) -> List[Path]:
        """
        Coba resume semua download yang gagal.
        
        Args:
            show_progress: Tampilkan progress bar
            
        Returns:
            List path file yang berhasil didownload
        """
        status = self.get_download_status()
        failed_downloads = status['failed'] + status['in_progress']
        
        if not failed_downloads:
            self.logger.info("✅ Tidak ada download yang gagal untuk di-resume")
            return []
        
        self.logger.info(f"🔄 Memulai resume {len(failed_downloads)} download yang gagal")
        
        # Siapkan daftar file untuk resume
        files_to_resume = [
            {
                'url': download['url'],
                'output_path': download['output_path'],
                'metadata': download.get('metadata', {})
            }
            for download in failed_downloads
        ]
        
        # Resume download
        return self.download_files(
            files=files_to_resume,
            show_progress=show_progress,
            resume=True
        )
    
    def extract_zip(
        self, 
        zip_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        remove_zip: bool = True,
        show_progress: bool = True
    ) -> Path:
        """
        Ekstrak file zip dengan progress bar.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output (opsional, jika None gunakan parent dari zip)
            remove_zip: Hapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Path direktori hasil ekstraksi
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"File zip tidak ditemukan: {zip_path}")
        
        # Tentukan direktori output
        if output_dir is None:
            output_dir = zip_path.parent
        else:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"📦 Mengekstrak {zip_path} ke {output_dir}")
        
        try:
            # Baca informasi zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Dapatkan ukuran total file yang akan diekstrak
                total_size = sum(info.file_size for info in zip_ref.infolist())
                extracted_size = 0
                
                # Setup progress bar
                progress = None
                if show_progress:
                    progress = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Extracting {zip_path.name}"
                    )
                
                # Ekstrak semua file
                for info in zip_ref.infolist():
                    zip_ref.extract(info, output_dir)
                    extracted_size += info.file_size
                    
                    # Update progress
                    if progress:
                        progress.update(info.file_size)
                
                # Tutup progress bar
                if progress:
                    progress.close()
            
            # Hapus file zip jika diminta
            if remove_zip:
                zip_path.unlink()
                self.logger.info(f"🗑️ File zip dihapus: {zip_path}")
            
            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
            return output_dir
            
        except zipfile.BadZipFile:
            self.logger.error(f"❌ File zip rusak: {zip_path}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekstrak zip: {str(e)}")
            raise