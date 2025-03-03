# File: smartcash/cli/configuration_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manajemen konfigurasi yang cerdas dengan validasi konfigurasi, backup otomatis, dan kemampuan migrasi

import os
import yaml
import json
import shutil
from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime, timedelta
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.exceptions.base import ConfigError, ValidationError

class ConfigurationManager:
    """
    Manajemen konfigurasi yang cerdas untuk proyek SmartCash dengan fitur:
    - Validasi konfigurasi otomatis
    - Backup konfigurasi periodik
    - Migrasi dari format lama
    - Riwayat perubahan
    """
    
    # Skema konfigurasi default
    DEFAULT_CONFIG: Dict[str, Any] = {
        'data_source': None,
        'detection_mode': None,
        'backbone': None,
        'layers': ['banknote'],
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10
        },
        'roboflow': {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3'
        },
        'model': {
            'img_size': [640, 640],
            'batch_size': 16,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'workers': 8,
            'memory_limit': 0.6
        },
        'dataset': {
            'classes': [
                '001', '002', '005', '010', '020', '050', '100',
                'l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100',
                'l3_sign', 'l3_text', 'l3_thread'
            ]
        },
        'output_dir': 'runs/train'
    }
    
    # Validator untuk nilai konfigurasi
    VALIDATORS: Dict[str, Dict] = {
        'training.batch_size': {
            'type': int,
            'min': 1,
            'max': 256
        },
        'training.learning_rate': {
            'type': float,
            'min': 0.00001,
            'max': 0.1
        },
        'training.epochs': {
            'type': int,
            'min': 1,
            'max': 1000
        },
        'model.batch_size': {
            'type': int,
            'min': 1,
            'max': 256
        },
        'model.conf_thres': {
            'type': float,
            'min': 0.01,
            'max': 1.0
        },
        'model.iou_thres': {
            'type': float,
            'min': 0.01,
            'max': 1.0
        }
    }
    
    # Nilai legal untuk enum fields
    VALID_VALUES: Dict[str, List] = {
        'data_source': ['local', 'roboflow'],
        'detection_mode': ['single', 'multi'],
        'backbone': ['cspdarknet', 'efficientnet']
    }
    
    def __init__(
        self, 
        base_config_path: str, 
        config_dir: Optional[str] = None,
        max_config_age_days: int = 30,
        max_config_files: int = 10,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi Configuration Manager dengan error handling yang lebih baik.
        
        Args:
            base_config_path: Path ke file konfigurasi dasar
            config_dir: Direktori konfigurasi (default: folder yang sama dengan base_config)
            max_config_age_days: Usia maksimal file konfigurasi sebelum dibersihkan
            max_config_files: Jumlah maksimal file konfigurasi yang disimpan
            logger: Logger kustom (opsional)
        """
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi atribut konfigurasi
        self.base_config_path: Path = Path(base_config_path)
        
        # Verifikasi direktori konfigurasi dengan error handling yang lebih baik
        if config_dir:
            self.config_dir: Path = Path(config_dir)
        else:
            # Default ke folder yang sama dengan base_config
            self.config_dir: Path = self.base_config_path.parent
        
        # Pastikan config_dir ada dan dapat ditulis
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = self.config_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            error_msg = f"Tidak memiliki izin untuk menulis ke direktori konfigurasi: {self.config_dir}"
            self.logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Gagal akses direktori konfigurasi: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigError(error_msg)
        
        self.max_config_age: timedelta = timedelta(days=max_config_age_days)
        self.max_config_files: int = max_config_files
        
        # Inisialisasi atribut konfigurasi dengan error handling terpisah
        try:
            self.base_config: Dict[str, Any] = self._load_base_config()
        except Exception as e:
            error_msg = f"Gagal memuat konfigurasi dasar: {str(e)}"
            self.logger.error(error_msg)
            # Fall back to default config
            self.base_config = self.DEFAULT_CONFIG.copy()
            self.logger.warning("⚠️ Menggunakan konfigurasi default karena error")
            
        try:
            self.current_config: Dict[str, Any] = self._load_latest_config() or self.base_config.copy()
        except Exception as e:
            error_msg = f"Gagal memuat konfigurasi terbaru: {str(e)}"
            self.logger.error(error_msg)
            # Fall back to base config
            self.current_config = self.base_config.copy()
            self.logger.warning("⚠️ Menggunakan konfigurasi dasar karena error")
            
        self.config_history: List[Dict[str, Any]] = []
        
        # Lakukan validasi konfigurasi saat ini dengan error handling yang lebih baik
        try:
            self._validate_config(self.current_config)
        except ValidationError as ve:
            self.logger.warning(f"⚠️ Validasi konfigurasi gagal: {str(ve)}")
            # Tidak raise exception, hanya log warning
        
        # Lakukan pembersihan saat inisialisasi
        try:
            self._cleanup_old_configs()
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membersihkan konfigurasi lama: {str(e)}")
        
        self.logger.info(f"🔧 Konfigurasi dimuat dari {self.base_config_path}")
        
    def _load_base_config(self) -> Dict[str, Any]:
        """
        Muat konfigurasi dasar dengan penanganan kesalahan.
        
        Returns:
            Dictionary konfigurasi dasar
            
        Raises:
            ConfigError: Jika gagal memuat konfigurasi
        """
        try:
            # Coba memuat konfigurasi dari file
            if self.base_config_path.exists():
                with open(self.base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f) or {}
                
                # Gabungkan default dengan konfigurasi yang dimuat
                return self._deep_merge(self.DEFAULT_CONFIG, base_config)
            
            self.logger.warning(
                f"⚠️ File konfigurasi tidak ditemukan: {self.base_config_path}, "
                "menggunakan default"
            )
            return self.DEFAULT_CONFIG.copy()
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi dasar: {e}")
            raise ConfigError(f"Gagal memuat konfigurasi dasar: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Melakukan merge rekursif untuk konfigurasi.
        
        Args:
            base: Dictionary dasar
            update: Dictionary update
            
        Returns:
            Dictionary hasil merge
        """
        merged = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Jika kunci sudah ada dan bernilai dict, lakukan merge rekursif
                merged[key] = self._deep_merge(merged.get(key, {}), value)
            else:
                # Jika bukan dict, langsung timpa atau tambahkan
                merged[key] = value
        return merged
    
    def _load_latest_config(self) -> Optional[Dict[str, Any]]:
        """
        Muat konfigurasi training terbaru.
        
        Returns:
            Dictionary konfigurasi terbaru atau None jika tidak ada
        """
        try:
            # Temukan file konfigurasi training terbaru
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Kembalikan konfigurasi terbaru jika ada
            if config_files:
                with open(config_files[0], 'r') as f:
                    latest_config = yaml.safe_load(f) or {}
                
                self.logger.info(f"📄 Memuat konfigurasi terbaru: {config_files[0].name}")
                
                # Merge dengan base config untuk memastikan semua kunci ada
                return self._deep_merge(self.base_config, latest_config)
            
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi terbaru: {str(e)}")
            return None
    
    def _cleanup_old_configs(self) -> None:
        """
        Bersihkan file konfigurasi lama dengan logging detail.
        """
        try:
            # Temukan semua file konfigurasi
            config_files = sorted(
                self.config_dir.glob('train_config_*.yaml'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not config_files:
                return
                
            self.logger.info(f"🧹 Membersihkan konfigurasi lama, ditemukan {len(config_files)} file")
            
            # Calculate cutoff date
            now = datetime.now()
            cutoff_date = now - self.max_config_age
            
            # Batasi jumlah file yang disimpan
            files_to_keep = config_files[:self.max_config_files]
            files_to_remove = []
            
            # Tambahkan file yang terlalu tua
            for file in config_files:
                mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                if mod_time < cutoff_date and file not in files_to_keep:
                    files_to_remove.append(file)
            
            # Hapus file yang berlebihan
            for file in files_to_remove:
                try:
                    file.unlink()
                    self.logger.info(f"🗑️ Menghapus konfigurasi lama: {file.name}")
                except Exception as remove_error:
                    self.logger.warning(f"⚠️ Gagal menghapus {file.name}: {remove_error}")
            
            if files_to_remove:
                self.logger.success(f"✅ Berhasil membersihkan {len(files_to_remove)} file konfigurasi lama")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan konfigurasi: {str(e)}")

    # Metode untuk update nilai tidak berubah, tapi akan ditingkatkan error handlingnya
    def update(self, key: str, value: Any) -> None:
        """
        Perbarui konfigurasi dengan perbaikan penanganan error struktur.
        
        Args:
            key: Kunci yang akan diperbarui (mendukung dot notation)
            value: Nilai baru
            
        Raises:
            ValidationError: Jika nilai tidak valid
        """
        # Koreksi kunci jika diperlukan
        if key == 'mode.batch_size':
            self.logger.warning(f"⚠️ Kunci 'mode.batch_size' ditemukan, kemungkinan typo. Dikoreksi ke 'model.batch_size'")
            key = 'model.batch_size'
            
        # Validasi nilai
        try:
            self._validate_value(key, value)
        except ValidationError as ve:
            self.logger.error(f"❌ Validasi gagal untuk {key}: {str(ve)}")
            raise ValidationError(str(ve))
        
        # Simpan untuk riwayat
        old_config = self.current_config.copy()
        
        # Update konfigurasi
        keys = key.split('.')
        
        # Cek struktur untuk memastikan path ada
        config = self.current_config
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # Jika bukan dict, konversi ke dict
                self.logger.warning(
                    f"⚠️ Mengkonversi {'.'.join(keys[:i+1])} dari {type(config[k]).__name__} ke dict"
                )
                config[k] = {}
            config = config[k]
        
        # Set nilai pada level terakhir
        last_key = keys[-1]
        config[last_key] = value
        
        # Catat riwayat
        self.config_history.append(old_config)
        
        # Log update
        self.logger.info(f"📝 Update konfigurasi: {key} = {value}")
        
    def _validate_value(self, key: str, value: Any) -> None:
        """
        Validasi nilai konfigurasi dengan perbaikan untuk error kunci mode.batch_size.
        
        Args:
            key: Kunci konfigurasi
            value: Nilai yang akan divalidasi
            
        Raises:
            ValidationError: Jika nilai tidak valid
        """
        # Perbaikan untuk typo pada kunci 'mode.batch_size'
        if key == 'mode.batch_size':
            self.logger.warning(f"⚠️ Kunci 'mode.batch_size' ditemukan, kemungkinan typo. Seharusnya 'model.batch_size'")
            key = 'model.batch_size'  # Koreksi ke kunci yang benar
        
        # Skip validasi jika nilai None atau key tidak dalam validator
        if value is None:
            return
            
        # Cek enum values
        if key in self.VALID_VALUES:
            if value not in self.VALID_VALUES[key]:
                valid_values = ", ".join(self.VALID_VALUES[key])
                raise ValidationError(
                    f"Nilai tidak valid untuk {key}: {value}. "
                    f"Nilai yang valid: {valid_values}"
                )
                
        # Validasi tipe dan range
        if key in self.VALIDATORS:
            validator = self.VALIDATORS[key]
            
            # Validasi tipe
            expected_type = validator.get('type')
            if expected_type and not isinstance(value, expected_type):
                # Log tipe aktual untuk debugging
                actual_type = type(value).__name__
                raise ValidationError(
                    f"Tipe tidak valid untuk {key}: {actual_type}. "
                    f"Tipe yang diharapkan: {expected_type.__name__}"
                )
                
            # Validasi minimum
            min_val = validator.get('min')
            if min_val is not None and value < min_val:
                raise ValidationError(
                    f"Nilai terlalu kecil untuk {key}: {value}. "
                    f"Minimum: {min_val}"
                )
                
            # Validasi maximum
            max_val = validator.get('max')
            if max_val is not None and value > max_val:
                raise ValidationError(
                    f"Nilai terlalu besar untuk {key}: {value}. "
                    f"Maximum: {max_val}"
                )
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validasi seluruh konfigurasi.
        
        Args:
            config: Konfigurasi yang akan divalidasi
            
        Raises:
            ValidationError: Jika konfigurasi tidak valid
        """
        # Cek nilai-nilai kritis
        for key in self.VALIDATORS:
            # Navigate nested keys
            keys = key.split('.')
            value = config
            for k in keys:
                value = value.get(k, {}) if isinstance(value, dict) else None
                
            # Skip if not set
            if value is None:
                continue
                
            # Validate
            self._validate_value(key, value)
            
        # Cek konfigurasi yang enum
        for key, valid_values in self.VALID_VALUES.items():
            value = config.get(key)
            
            # Skip if not set
            if value is None:
                continue
                
            # Validate
            if value not in valid_values:
                valid_str = ", ".join(valid_values)
                raise ValidationError(
                    f"Nilai tidak valid untuk {key}: {value}. "
                    f"Nilai yang valid: {valid_str}"
                )
    
    # Metode untuk menyimpan konfigurasi yang menjadi sumber error
    def save(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Simpan konfigurasi saat ini dengan error handling yang lebih baik.
        
        Args:
            config: Konfigurasi kustom untuk disimpan (opsional)
            
        Returns:
            Path ke file konfigurasi yang disimpan
            
        Raises:
            ConfigError: Jika gagal menyimpan konfigurasi
        """
        # Bersihkan file konfigurasi sebelum menyimpan
        try:
            self._cleanup_old_configs()
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membersihkan konfigurasi lama: {str(e)}")
        
        config = config or self.current_config
        
        try:
            # Validasi konfigurasi sebelum disimpan
            self._validate_config(config)
            
            # Generate nama file dengan timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_file = self.config_dir / f"train_config_{timestamp}.yaml"
            
            # Simpan ke file sementara dulu untuk mencegah file rusak jika error
            temp_file = self.config_dir / f"temp_config_{timestamp}.yaml"
            try:
                with open(temp_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                # Jika berhasil ditulis, rename ke nama file final
                if temp_file.exists():
                    temp_file.rename(config_file)
                else:
                    raise ConfigError(f"Gagal menulis ke file sementara: {temp_file}")
            except Exception as write_error:
                # Clean up temp file jika ada
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                raise write_error
                
            self.logger.success(f"💾 Konfigurasi disimpan ke {config_file}")
            
            return config_file
            
        except ValidationError as ve:
            error_msg = f"Validasi konfigurasi gagal: {str(ve)}"
            self.logger.error(error_msg)
            raise ConfigError(error_msg)
        except PermissionError as pe:
            error_msg = f"Tidak memiliki izin untuk menulis ke {self.config_dir}: {str(pe)}"
            self.logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Gagal menyimpan konfigurasi: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise ConfigError(error_msg)

    def backup(self, dest_dir: Optional[str] = None) -> Path:
        """
        Buat backup seluruh konfigurasi.
        
        Args:
            dest_dir: Direktori tujuan (opsional)
            
        Returns:
            Path ke file backup
            
        Raises:
            ConfigError: Jika gagal membuat backup
        """
        try:
            # Setup direktori backup
            backup_dir = Path(dest_dir or self.config_dir / 'backups')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate nama file dengan timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"config_backup_{timestamp}.yaml"
            
            # Simpan backup current config
            with open(backup_file, 'w') as f:
                yaml.dump(self.current_config, f, default_flow_style=False)
                
            self.logger.success(f"📦 Backup konfigurasi dibuat: {backup_file}")
            
            return backup_file
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup: {str(e)}")
            raise ConfigError(f"Gagal membuat backup: {str(e)}")
    
    def reset(self, sections: Optional[List[str]] = None) -> None:
        """
        Reset konfigurasi ke nilai default.
        
        Args:
            sections: List section yang akan direset (None untuk reset semua)
        """
        # Simpan konfigurasi saat ini untuk history
        old_config = self.current_config.copy()
        
        if sections is None:
            # Reset seluruh konfigurasi
            self.current_config = self.DEFAULT_CONFIG.copy()
            self.logger.info("🔄 Reset seluruh konfigurasi ke default")
        else:
            # Reset section tertentu
            for section in sections:
                if section in self.DEFAULT_CONFIG:
                    self.current_config[section] = self.DEFAULT_CONFIG[section].copy()
                    self.logger.info(f"🔄 Reset section '{section}' ke default")
                else:
                    self.logger.warning(f"⚠️ Section tidak ditemukan: {section}")
        
        # Catat riwayat
        self.config_history.append(old_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dapatkan nilai konfigurasi dengan dukungan nested key.
        
        Args:
            key: Kunci konfigurasi (mendukung dot notation)
            default: Nilai default jika kunci tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        keys = key.split('.')
        value = self.current_config
        
        # Navigate to final key
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def find_class_names(self) -> List[str]:
        """
        Temukan nama kelas untuk dataset.
        
        Returns:
            List nama kelas
        """
        # Cari dari berbagai lokasi yang mungkin
        locations = [
            'dataset.classes',
            'data.classes'
        ]
        
        for loc in locations:
            classes = self.get(loc)
            if classes:
                return classes
        
        # Cek mode deteksi 
        if self.get('detection_mode') == 'multi':
            # Multi-layer: semua 17 kelas
            return [
                '001', '002', '005', '010', '020', '050', '100',  # Layer 1 (banknote)
                'l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100',  # Layer 2 (nominal)
                'l3_sign', 'l3_text', 'l3_thread'  # Layer 3 (security)
            ]
        else:
            # Default: 7 kelas mata uang layer 1
            return ['001', '002', '005', '010', '020', '050', '100']
    
    def get_required_config_status(self) -> Dict[str, bool]:
        """
        Dapatkan status konfigurasi yang dibutuhkan untuk training.
        
        Returns:
            Dictionary berisi status tiap konfigurasi yang dibutuhkan
        """
        required = {
            'data_source': self.get('data_source') is not None,
            'backbone': self.get('backbone') is not None,
            'detection_mode': self.get('detection_mode') is not None,
            'batch_size': self.get('training.batch_size') is not None,
            'learning_rate': self.get('training.learning_rate') is not None,
            'epochs': self.get('training.epochs') is not None
        }
        
        return required
    
    def export(
        self, 
        format: str = 'yaml',
        path: Optional[str] = None
    ) -> str:
        """
        Export konfigurasi ke format yang ditentukan.
        
        Args:
            format: Format export ('yaml' atau 'json')
            path: Path untuk menyimpan hasil export (opsional)
            
        Returns:
            Path ke file export atau string konfigurasi jika path None
        """
        try:
            if format.lower() == 'yaml':
                if path:
                    with open(path, 'w') as f:
                        yaml.dump(self.current_config, f, default_flow_style=False)
                    return path
                else:
                    return yaml.dump(self.current_config, default_flow_style=False)
            elif format.lower() == 'json':
                if path:
                    with open(path, 'w') as f:
                        json.dump(self.current_config, f, indent=2)
                    return path
                else:
                    return json.dumps(self.current_config, indent=2)
            else:
                raise ValueError(f"Format tidak didukung: {format}")
        except Exception as e:
            self.logger.error(f"❌ Gagal export konfigurasi: {str(e)}")
            raise ConfigError(f"Gagal export konfigurasi: {str(e)}")
    
    def import_config(self, path: str) -> None:
        """
        Import konfigurasi dari file.
        
        Args:
            path: Path ke file konfigurasi
            
        Raises:
            ConfigError: Jika gagal import konfigurasi
        """
        try:
            config_path = Path(path)
            if not config_path.exists():
                raise ConfigError(f"File tidak ditemukan: {path}")
                
            # Detect format dari extension
            suffix = config_path.suffix.lower()
            
            if suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    new_config = yaml.safe_load(f)
            elif suffix == '.json':
                with open(config_path, 'r') as f:
                    new_config = json.load(f)
            else:
                raise ConfigError(f"Format file tidak didukung: {suffix}")
                
            # Validate
            self._validate_config(new_config)
            
            # Backup current config
            self.config_history.append(self.current_config.copy())
            
            # Update current config
            self.current_config = self._deep_merge(self.DEFAULT_CONFIG, new_config)
            
            self.logger.success(f"📥 Konfigurasi berhasil diimport dari {path}")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal import konfigurasi: {str(e)}")
            raise ConfigError(f"Gagal import konfigurasi: {str(e)}")

    def debug_config(self) -> Dict[str, Any]:
        """
        Dapatkan informasi debug untuk konfigurasi saat ini.
        
        Returns:
            Dict berisi informasi debug
        """
        debug_info = {
            'config_dir': str(self.config_dir),
            'base_config_path': str(self.base_config_path),
            'current_config': self.current_config,
            'dir_writeable': os.access(self.config_dir, os.W_OK),
            'config_keys': list(self.current_config.keys()),
            'validation_issues': self._find_validation_issues()
        }
        
        return debug_info
        
    def _find_validation_issues(self) -> Dict[str, str]:
        """
        Temukan masalah validasi dalam konfigurasi saat ini.
        
        Returns:
            Dict berisi masalah validasi
        """
        issues = {}
        
        # Cek nilai-nilai kritis
        for key in self.VALIDATORS:
            # Navigate nested keys
            keys = key.split('.')
            value = self.current_config
            for k in keys:
                value = value.get(k, {}) if isinstance(value, dict) else None
                
            # Skip if not set
            if value is None:
                continue
                
            # Validate type
            validator = self.VALIDATORS[key]
            expected_type = validator.get('type')
            if expected_type and not isinstance(value, expected_type):
                issues[key] = f"Tipe tidak valid: {type(value).__name__}, diharapkan {expected_type.__name__}"
                continue
                
            # Validate min
            min_val = validator.get('min')
            if min_val is not None and value < min_val:
                issues[key] = f"Nilai terlalu kecil: {value}, minimum {min_val}"
                continue
                
            # Validate max
            max_val = validator.get('max')
            if max_val is not None and value > max_val:
                issues[key] = f"Nilai terlalu besar: {value}, maksimum {max_val}"
                continue
                
        # Cek enum values
        for key, valid_values in self.VALID_VALUES.items():
            value = self.current_config.get(key)
            
            # Skip if not set
            if value is None:
                continue
                
            # Validate enum value
            if value not in valid_values:
                issues[key] = f"Nilai tidak valid: {value}, nilai yang valid: {', '.join(valid_values)}"
                
        return issues