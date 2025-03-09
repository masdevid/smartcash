"""
File: smartcash/handlers/preprocessing/integration/colab_drive_adapter.py
Author: Alfrida Sabar
Deskripsi: Adapter untuk integrasi Google Drive di Google Colab.
           Menyediakan fungsi mount/unmount, konversi path, dan pembuatan symlink
           dengan penanganan error yang robust dan dukungan operasi file yang mulus.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import traceback
from datetime import datetime

from smartcash.utils.logger import get_logger, SmartCashLogger


class ColabDriveAdapter:
    """
    Adapter untuk integrasi Google Drive di Google Colab.
    Menyediakan fungsi mount/unmount, konversi path, dan pembuatan symlink.
    """
    
    def __init__(
        self, 
        project_dir: str = "/content/SmartCash",
        drive_mount_point: str = "/content/drive",
        drive_project_path: str = "MyDrive/SmartCash",
        logger: Optional[SmartCashLogger] = None,
        auto_mount: bool = False
    ):
        """
        Inisialisasi adapter Google Drive untuk Colab.
        
        Args:
            project_dir: Direktori lokal project di Colab
            drive_mount_point: Titik mount Google Drive
            drive_project_path: Jalur project di Google Drive (relatif terhadap mount_point)
            logger: Logger kustom (opsional)
            auto_mount: Mount Google Drive secara otomatis saat inisialisasi
        """
        self.project_dir = Path(project_dir)
        self.drive_mount_point = Path(drive_mount_point)
        self.drive_project_path = drive_project_path
        self.logger = logger or get_logger("ColabDriveAdapter")
        self.is_mounted = False
        self.symlinks = {}
        
        # Deteksi apakah berjalan di Google Colab
        self.is_colab = self._is_running_in_colab()
        
        if not self.is_colab:
            self.logger.warning("⚠️ Bukan lingkungan Google Colab, adapter akan beroperasi dalam mode terbatas")
        elif auto_mount:
            self.mount_drive()
    
    def mount_drive(self, force_remount: bool = False) -> bool:
        """
        Mount Google Drive ke Colab.
        
        Args:
            force_remount: Paksa remount meskipun sudah ter-mount
            
        Returns:
            bool: True jika berhasil
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Tidak dapat mount Google Drive: Bukan lingkungan Google Colab")
            return False
        
        # Cek apakah sudah ter-mount
        if self.is_mounted and not force_remount:
            self.logger.info("🔄 Google Drive sudah ter-mount")
            return True
        
        try:
            self.logger.start("🔌 Memulai proses mount Google Drive")
            
            # Import hanya jika di Colab
            from google.colab import drive
            
            # Mount drive
            drive.mount(str(self.drive_mount_point), force_remount=force_remount)
            
            # Verifikasi mount
            if self.drive_mount_point.exists():
                self.is_mounted = True
                self.logger.success(f"✅ Google Drive berhasil di-mount di {self.drive_mount_point}")
                
                # Pastikan direktori project ada
                drive_project_full_path = self.drive_mount_point / self.drive_project_path
                if not drive_project_full_path.exists():
                    drive_project_full_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"📁 Membuat direktori project di Drive: {drive_project_full_path}")
                
                return True
            else:
                self.logger.error(f"❌ Gagal memverifikasi mount point: {self.drive_mount_point}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal mount Google Drive: {str(e)}")
            self.is_mounted = False
            return False
    
    def unmount_drive(self) -> bool:
        """
        Unmount Google Drive dari Colab.
        
        Returns:
            bool: True jika berhasil
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Tidak dapat unmount Google Drive: Bukan lingkungan Google Colab")
            return False
        
        if not self.is_mounted:
            self.logger.info("🔄 Google Drive tidak ter-mount")
            return True
        
        try:
            # Hapus semua symlink terlebih dahulu
            for symlink_path in self.symlinks.keys():
                self._remove_symlink(symlink_path)
            
            # Unmount dengan perintah terminal
            result = subprocess.run(
                ["fusermount", "-u", str(self.drive_mount_point)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.is_mounted = False
                self.logger.success(f"✅ Google Drive berhasil di-unmount dari {self.drive_mount_point}")
                return True
            else:
                self.logger.error(f"❌ Gagal unmount Google Drive: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal unmount Google Drive: {str(e)}")
            return False
    
    def to_drive_path(self, local_path: Union[str, Path]) -> Path:
        """
        Konversi path lokal menjadi path di Google Drive.
        
        Args:
            local_path: Path lokal yang akan dikonversi
            
        Returns:
            Path: Path di Google Drive
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, path tidak dapat dikonversi")
            return Path(local_path)
        
        local_path = Path(local_path)
        
        try:
            # Jika path adalah subpath dari project_dir
            if str(local_path).startswith(str(self.project_dir)):
                # Dapatkan path relatif dari project_dir
                rel_path = local_path.relative_to(self.project_dir)
                
                # Gabungkan dengan path project di Drive
                drive_path = self.drive_mount_point / self.drive_project_path / rel_path
                return drive_path
            
            # Jika path absolut yang tidak di project_dir
            elif local_path.is_absolute():
                # Gunakan path lokal apa adanya
                return local_path
            
            # Jika path relatif
            else:
                # Anggap relatif terhadap project_dir
                drive_path = self.drive_mount_point / self.drive_project_path / local_path
                return drive_path
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mengonversi path lokal ke Drive: {str(e)}")
            return Path(local_path)
    
    def to_local_path(self, drive_path: Union[str, Path]) -> Path:
        """
        Konversi path Google Drive menjadi path lokal.
        
        Args:
            drive_path: Path Google Drive yang akan dikonversi
            
        Returns:
            Path: Path lokal
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, path tidak dapat dikonversi")
            return Path(drive_path)
        
        drive_path = Path(drive_path)
        
        try:
            # Jika path di drive_mount_point dan di bawah drive_project_path
            drive_project_full_path = self.drive_mount_point / self.drive_project_path
            
            if str(drive_path).startswith(str(drive_project_full_path)):
                # Dapatkan path relatif dari project di Drive
                rel_path = drive_path.relative_to(drive_project_full_path)
                
                # Gabungkan dengan project_dir lokal
                local_path = self.project_dir / rel_path
                return local_path
            
            # Jika path di drive_mount_point tapi tidak di bawah drive_project_path
            elif str(drive_path).startswith(str(self.drive_mount_point)):
                # Kembalikan path apa adanya
                return drive_path
            
            # Jika path relatif terhadap drive_project_path
            elif not drive_path.is_absolute():
                # Anggap path relatif terhadap project_dir
                local_path = self.project_dir / drive_path
                return local_path
            
            # Jika path absolut tapi bukan di drive_mount_point
            else:
                # Kembalikan path apa adanya
                return drive_path
                
        except Exception as e:
            self.logger.error(f"❌ Gagal mengonversi path Drive ke lokal: {str(e)}")
            return Path(drive_path)
    
    def setup_symlinks(self, dir_list: List[str], force: bool = False) -> Dict[str, bool]:
        """
        Setup symlink dari direktori di Google Drive ke direktori lokal.
        
        Args:
            dir_list: List direktori yang akan di-symlink
            force: Paksa recreate symlink meskipun sudah ada
            
        Returns:
            Dict[str, bool]: Status symlink per direktori
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat membuat symlink")
            return {dir_name: False for dir_name in dir_list}
        
        result = {}
        
        for dir_name in dir_list:
            # Path di Google Drive
            drive_dir = self.drive_mount_point / self.drive_project_path / dir_name
            
            # Path lokal
            local_dir = self.project_dir / dir_name
            
            # Pastikan direktori di Drive ada
            if not drive_dir.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Membuat direktori di Drive: {drive_dir}")
            
            # Buat symlink jika belum ada atau force
            symlink_status = self._create_symlink(
                src_path=drive_dir,
                dest_path=local_dir,
                force=force
            )
            
            result[dir_name] = symlink_status
            self.symlinks[str(local_dir)] = {"src": str(drive_dir), "status": symlink_status}
        
        return result
    
    def _create_symlink(self, src_path: Path, dest_path: Path, force: bool = False) -> bool:
        """
        Buat symlink dengan penanganan error yang robust.
        
        Args:
            src_path: Path sumber (asli)
            dest_path: Path tujuan (symlink)
            force: Paksa recreate symlink meskipun sudah ada
            
        Returns:
            bool: True jika berhasil
        """
        try:
            # Hapus symlink jika sudah ada dan force=True
            if dest_path.exists():
                if force:
                    self._remove_symlink(dest_path)
                else:
                    if dest_path.is_symlink():
                        self.logger.info(f"🔄 Symlink sudah ada: {dest_path} -> {os.readlink(dest_path)}")
                        return True
                    else:
                        self.logger.warning(
                            f"⚠️ Tujuan symlink ada tapi bukan symlink: {dest_path}. "
                            f"Gunakan force=True untuk menimpa."
                        )
                        return False
            
            # Pastikan direktori induk ada
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Buat symlink
            os.symlink(str(src_path), str(dest_path), target_is_directory=True)
            
            # Verifikasi symlink
            if dest_path.is_symlink() and dest_path.exists():
                self.logger.success(f"✅ Berhasil membuat symlink: {dest_path} -> {src_path}")
                return True
            else:
                self.logger.error(f"❌ Gagal memverifikasi symlink: {dest_path}")
                return False
                
        except PermissionError as e:
            self.logger.error(f"❌ Error permission saat membuat symlink {dest_path}: {str(e)}")
            self.logger.info("💡 Coba jalankan 'chmod +x /content' atau restart runtime")
            return False
            
        except FileExistsError as e:
            self.logger.warning(f"⚠️ Symlink sudah ada, gunakan force=True untuk menimpa: {str(e)}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat symlink {dest_path}: {str(e)}")
            self.logger.debug(f"🐞 Traceback: {traceback.format_exc()}")
            return False
    
    def _remove_symlink(self, symlink_path: Union[str, Path]) -> bool:
        """
        Hapus symlink dengan penanganan error yang robust.
        
        Args:
            symlink_path: Path symlink yang akan dihapus
            
        Returns:
            bool: True jika berhasil
        """
        symlink_path = Path(symlink_path)
        
        try:
            if not symlink_path.exists():
                return True
                
            if symlink_path.is_symlink():
                symlink_path.unlink()
                self.logger.info(f"🗑️ Menghapus symlink: {symlink_path}")
                return True
            else:
                self.logger.warning(f"⚠️ Path bukan symlink: {symlink_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menghapus symlink {symlink_path}: {str(e)}")
            return False
    
    def sync_to_drive(self, src_dir: Union[str, Path], dest_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Sinkronisasi direktori lokal ke Google Drive.
        
        Args:
            src_dir: Direktori sumber (lokal)
            dest_dir: Direktori tujuan di Drive (opsional)
            
        Returns:
            bool: True jika berhasil
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat sinkronisasi")
            return False
        
        src_dir = Path(src_dir)
        
        # Jika dest_dir tidak diberikan, gunakan path relatif terhadap project_dir
        if dest_dir is None:
            # Cek apakah src_dir adalah subpath dari project_dir
            try:
                rel_path = src_dir.relative_to(self.project_dir)
                dest_dir = self.drive_mount_point / self.drive_project_path / rel_path
            except ValueError:
                # Jika bukan subpath, gunakan nama direktori saja
                dest_dir = self.drive_mount_point / self.drive_project_path / src_dir.name
        else:
            dest_dir = Path(dest_dir)
            # Konversi ke path Drive jika perlu
            if not str(dest_dir).startswith(str(self.drive_mount_point)):
                dest_dir = self.to_drive_path(dest_dir)
        
        try:
            # Pastikan direktori tujuan ada
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.start(f"🔄 Memulai sinkronisasi: {src_dir} → {dest_dir}")
            
            # Gunakan rsync untuk sinkronisasi
            result = subprocess.run(
                ["rsync", "-av", f"{src_dir}/", f"{dest_dir}/"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.success(f"✅ Sinkronisasi selesai: {src_dir} → {dest_dir}")
                return True
            else:
                self.logger.error(f"❌ Gagal sinkronisasi: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal sinkronisasi {src_dir} ke {dest_dir}: {str(e)}")
            return False
    
    def sync_from_drive(self, src_dir: Union[str, Path], dest_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Sinkronisasi direktori dari Google Drive ke lokal.
        
        Args:
            src_dir: Direktori sumber di Drive
            dest_dir: Direktori tujuan lokal (opsional)
            
        Returns:
            bool: True jika berhasil
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat sinkronisasi")
            return False
        
        src_dir = Path(src_dir)
        
        # Konversi ke path Drive jika perlu
        if not str(src_dir).startswith(str(self.drive_mount_point)):
            src_dir = self.to_drive_path(src_dir)
        
        # Jika dest_dir tidak diberikan, gunakan path relatif terhadap drive_project_path
        if dest_dir is None:
            try:
                drive_project_full_path = self.drive_mount_point / self.drive_project_path
                rel_path = src_dir.relative_to(drive_project_full_path)
                dest_dir = self.project_dir / rel_path
            except ValueError:
                # Jika bukan subpath, gunakan nama direktori saja
                dest_dir = self.project_dir / src_dir.name
        else:
            dest_dir = Path(dest_dir)
        
        try:
            # Pastikan direktori tujuan ada
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.start(f"🔄 Memulai sinkronisasi: {src_dir} → {dest_dir}")
            
            # Gunakan rsync untuk sinkronisasi
            result = subprocess.run(
                ["rsync", "-av", f"{src_dir}/", f"{dest_dir}/"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.success(f"✅ Sinkronisasi selesai: {src_dir} → {dest_dir}")
                return True
            else:
                self.logger.error(f"❌ Gagal sinkronisasi: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal sinkronisasi {src_dir} ke {dest_dir}: {str(e)}")
            return False
    
    def get_available_space(self) -> Dict[str, Any]:
        """
        Dapatkan ruang tersedia di Google Drive.
        
        Returns:
            Dict[str, Any]: Informasi ruang tersedia
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat mendapatkan ruang tersedia")
            return {"available_gb": 0, "total_gb": 0, "used_gb": 0, "used_percent": 0}
        
        try:
            # Gunakan df untuk mendapatkan informasi disk
            result = subprocess.run(
                ["df", "-h", str(self.drive_mount_point)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse output df
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        total = parts[1]
                        used = parts[2]
                        avail = parts[3]
                        used_percent = parts[4]
                        
                        # Konversi ke bytes
                        total_bytes = self._parse_size_to_bytes(total)
                        used_bytes = self._parse_size_to_bytes(used)
                        avail_bytes = self._parse_size_to_bytes(avail)
                        
                        # Konversi ke GB untuk output
                        total_gb = total_bytes / (1024**3)
                        used_gb = used_bytes / (1024**3)
                        avail_gb = avail_bytes / (1024**3)
                        
                        self.logger.info(
                            f"💾 Ruang Drive: {avail_gb:.1f}GB tersedia dari {total_gb:.1f}GB "
                            f"({used_percent} terpakai)"
                        )
                        
                        return {
                            "available_gb": avail_gb,
                            "total_gb": total_gb,
                            "used_gb": used_gb,
                            "used_percent": used_percent.rstrip('%'),
                            "available_bytes": avail_bytes,
                            "total_bytes": total_bytes,
                            "used_bytes": used_bytes
                        }
            
            self.logger.warning(f"⚠️ Gagal mendapatkan informasi ruang disk: {result.stderr}")
            return {"available_gb": 0, "total_gb": 0, "used_gb": 0, "used_percent": 0}
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan informasi ruang disk: {str(e)}")
            return {"available_gb": 0, "total_gb": 0, "used_gb": 0, "used_percent": 0}
    
    def check_permissions(self) -> Dict[str, bool]:
        """
        Periksa apakah memiliki permission yang diperlukan.
        
        Returns:
            Dict[str, bool]: Status permission
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Bukan lingkungan Google Colab, tidak dapat memeriksa permission")
            return {"colab": False, "drive_mounted": False, "write_permission": False}
        
        result = {
            "colab": True,
            "drive_mounted": self.is_mounted,
            "write_permission": False,
            "symlink_permission": False,
            "project_dir_exists": self.project_dir.exists()
        }
        
        try:
            # Cek permission tulis
            if self.is_mounted:
                test_file = self.drive_mount_point / self.drive_project_path / f"test_perm_{int(time.time())}.txt"
                
                try:
                    with open(test_file, 'w') as f:
                        f.write("Permission test")
                    
                    if test_file.exists():
                        result["write_permission"] = True
                        test_file.unlink()  # Hapus file test
                except Exception:
                    result["write_permission"] = False
            
            # Cek permission symlink
            test_symlink = self.project_dir / f"test_symlink_{int(time.time())}"
            test_target = self.project_dir / "test_target"
            
            try:
                # Buat direktori target
                if not test_target.exists():
                    test_target.mkdir(parents=True, exist_ok=True)
                
                # Buat symlink
                os.symlink(str(test_target), str(test_symlink), target_is_directory=True)
                
                if test_symlink.is_symlink() and test_symlink.exists():
                    result["symlink_permission"] = True
                    
                # Hapus test
                if test_symlink.exists():
                    test_symlink.unlink()
                if test_target.exists():
                    test_target.rmdir()
            except Exception:
                result["symlink_permission"] = False
            
            # Log hasil
            if result["drive_mounted"] and result["write_permission"] and result["symlink_permission"]:
                self.logger.success("✅ Semua permission terpenuhi")
            else:
                missing = []
                if not result["drive_mounted"]:
                    missing.append("Google Drive tidak ter-mount")
                if not result["write_permission"]:
                    missing.append("Tidak ada permission tulis di Drive")
                if not result["symlink_permission"]:
                    missing.append("Tidak ada permission symlink")
                
                self.logger.warning(f"⚠️ Beberapa permission tidak terpenuhi: {', '.join(missing)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memeriksa permission: {str(e)}")
            return {
                "colab": True,
                "drive_mounted": self.is_mounted,
                "write_permission": False,
                "symlink_permission": False
            }
    
    def create_backup(
        self, 
        dir_path: Union[str, Path],
        backup_name: Optional[str] = None,
        include_timestamp: bool = True,
        compress: bool = True
    ) -> Optional[Path]:
        """
        Buat backup folder penting ke Google Drive.
        
        Args:
            dir_path: Path direktori yang akan di-backup
            backup_name: Nama backup (opsional)
            include_timestamp: Sertakan timestamp dalam nama backup
            compress: Kompres backup ke dalam format tar.gz
            
        Returns:
            Optional[Path]: Path backup jika berhasil, None jika gagal
        """
        if not self.is_mounted:
            self.logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat membuat backup")
            return None
        
        dir_path = Path(dir_path)
        
        # Verifikasi direktori ada
        if not dir_path.exists():
            self.logger.error(f"❌ Direktori tidak ditemukan: {dir_path}")
            return None
        
        try:
            # Tentukan nama backup
            if backup_name is None:
                backup_name = dir_path.name
            
            timestamp = ""
            if include_timestamp:
                timestamp = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir_name = f"{backup_name}{timestamp}"
            
            # Lokasi backup di Drive
            backups_dir = self.drive_mount_point / self.drive_project_path / "backups"
            backups_dir.mkdir(parents=True, exist_ok=True)
            
            if compress:
                # Nama file backup dengan ekstensi
                backup_filename = f"{backup_dir_name}.tar.gz"
                backup_path = backups_dir / backup_filename
                
                self.logger.start(f"🔄 Membuat backup terkompresi: {dir_path} → {backup_path}")
                
                # Buat tar.gz
                result = subprocess.run(
                    ["tar", "-czf", str(backup_path), "-C", str(dir_path.parent), dir_path.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    backup_size = backup_path.stat().st_size / (1024 * 1024)  # Size in MB
                    self.logger.success(
                        f"✅ Backup selesai: {backup_path} ({backup_size:.2f} MB)"
                    )
                    return backup_path
                else:
                    self.logger.error(f"❌ Gagal membuat backup terkompresi: {result.stderr}")
                    return None
            else:
                # Backup tanpa kompresi (salinan folder)
                backup_path = backups_dir / backup_dir_name
                
                self.logger.start(f"🔄 Membuat backup: {dir_path} → {backup_path}")
                
                # Gunakan rsync untuk menyalin folder
                result = subprocess.run(
                    ["rsync", "-av", f"{dir_path}/", f"{backup_path}/"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.logger.success(f"✅ Backup selesai: {backup_path}")
                    return backup_path
                else:
                    self.logger.error(f"❌ Gagal membuat backup: {result.stderr}")
                    return None
                
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup {dir_path}: {str(e)}")
            self.logger.debug(f"🐞 Traceback: {traceback.format_exc()}")
            return None
    
    def _is_running_in_colab(self) -> bool:
        """
        Deteksi apakah kode berjalan dalam Google Colab.
        
        Returns:
            bool: True jika berjalan di Google Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
            
    def _parse_size_to_bytes(self, size_str: str) -> int:
        """
        Parse string ukuran (mis. '15G', '100M') ke bytes.
        
        Args:
            size_str: String ukuran dengan unit
            
        Returns:
            int: Ukuran dalam bytes
        """
        try:
            # Hapus semua spasi
            size_str = size_str.strip()
            
            # Pisahkan angka dan unit
            if size_str[-1].isalpha():
                number = float(size_str[:-1])
                unit = size_str[-1].upper()
            else:
                return float(size_str)
            
            # Konversi berdasarkan unit
            if unit == 'K':
                return int(number * 1024)
            elif unit == 'M':
                return int(number * 1024**2)
            elif unit == 'G':
                return int(number * 1024**3)
            elif unit == 'T':
                return int(number * 1024**4)
            else:
                return int(number)
        except Exception:
            # Fallback jika parsing gagal
            return 0