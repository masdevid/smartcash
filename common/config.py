"""
File: smartcash/common/config.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable

# Type variable untuk dependency injection
T = TypeVar('T')

class ConfigManager:
    """
    Manager untuk konfigurasi aplikasi dengan dukungan untuk:
    - Loading dari file YAML/JSON
    - Environment variable overrides
    - Hierarki konfigurasi
    - Dependency injection
    """
    
    DEFAULT_CONFIG_DIR = 'config'
    
    def __init__(self, 
                base_dir: Optional[str] = None, 
                config_file: Optional[str] = None,
                env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager.
        
        Args:
            base_dir: Direktori root project
            config_file: Path file konfigurasi utama (relatif ke base_dir)
            env_prefix: Prefix untuk environment variables
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_dir = self.base_dir / self.DEFAULT_CONFIG_DIR
        self.env_prefix = env_prefix
        
        # Konfigurasi dasar
        self.config = {}
        
        # Dependency container
        self._dependencies = {}
        self._factory_functions = {}
        
        # Muat konfigurasi dari file jika disediakan
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load konfigurasi dari file.
        
        Args:
            config_file: Nama file konfigurasi atau path lengkap
            
        Returns:
            Dictionary konfigurasi
        """
        # Tentukan path konfigurasi
        config_path = self._resolve_config_path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load berdasarkan ekstensi file
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Format file konfigurasi tidak didukung: {config_path.suffix}")
        
        # Override dengan environment variables
        self._override_with_env_vars()
        
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve path konfigurasi relatif atau absolut."""
        config_path = Path(config_file)
        
        # Jika path absolut, gunakan langsung
        if config_path.is_absolute():
            return config_path
            
        # Jika config ada di config_dir, gunakan
        if (self.config_dir / config_path).exists():
            return self.config_dir / config_path
            
        # Jika config ada di direktori kerja, gunakan
        if (Path.cwd() / config_path).exists():
            return Path.cwd() / config_path
            
        # Default ke config_dir
        return self.config_dir / config_path
    
    def _override_with_env_vars(self) -> None:
        """Override konfigurasi dengan environment variables."""
        for env_name, env_value in os.environ.items():
            # Hanya proses var dengan prefix yang sesuai
            if not env_name.startswith(self.env_prefix):
                continue
                
            # Convert format ENV_VAR ke nested dict
            # contoh: SMARTCASH_MODEL_IMG_SIZE_WIDTH=640 -> config['model']['img_size']['width']=640
            config_path = env_name[len(self.env_prefix):].lower().split('_')
            
            # Traverse & update config dict
            current = self.config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set nilai (dengan auto type conversion)
            key = config_path[-1]
            current[key] = self._parse_env_value(env_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse nilai environment variable ke tipe yang sesuai."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
            
        # Numbers
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        # Lists (comma-separated values)
        if ',' in value:
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
            
        # Default: string
        return value
    
    def get(self, key: str, default=None) -> Any:
        """
        Ambil nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            value: Nilai yang akan di-set
        """
        parts = key.split('.')
        current = self.config
        
        # Traverse sampai level terakhir
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set nilai
        current[parts[-1]] = value
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """
        Merge konfigurasi dari dict atau file.
        
        Args:
            config: Dictionary konfigurasi atau path file
            
        Returns:
            Konfigurasi setelah merge
        """
        # Load dari file jika string
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Format file tidak didukung: {config_path.suffix}")
        
        # Deep merge config
        self._deep_merge(self.config, config)
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """
        Deep merge dua dictionary.
        
        Args:
            target: Dictionary target
            source: Dictionary source
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Rekursif untuk nested dict
                self._deep_merge(target[key], value)
            else:
                # Override atau tambahkan key baru
                target[key] = value
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> None:
        """
        Simpan konfigurasi ke file.
        
        Args:
            config_file: Path file untuk menyimpan konfigurasi
            create_dirs: Flag untuk create direktori jika belum ada
        """
        config_path = Path(config_file)
        
        # Buat direktori jika diperlukan
        if create_dirs and not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan berdasarkan ekstensi
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        else:
            # Default ke YAML
            with open(f"{config_path}.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    # ===== Dependency Injection Methods =====
    
    def register(self, interface_type: Type[T], implementation: Type[T]) -> None:
        """
        Daftarkan implementasi untuk interface tertentu.
        
        Args:
            interface_type: Tipe interface yang akan diregister
            implementation: Implementasi dari interface tersebut
        """
        self._dependencies[interface_type] = implementation
    
    def register_instance(self, interface_type: Type[T], instance: T) -> None:
        """
        Daftarkan instance untuk interface tertentu (singleton).
        
        Args:
            interface_type: Tipe interface yang akan diregister
            instance: Instance objek implementasi
        """
        self._dependencies[interface_type] = instance
    
    def register_factory(self, interface_type: Type[T], factory: Callable[..., T]) -> None:
        """
        Daftarkan factory function untuk membuat implementasi.
        
        Args:
            interface_type: Tipe interface yang akan diregister
            factory: Factory function untuk membuat implementasi
        """
        self._factory_functions[interface_type] = factory
    
    def resolve(self, interface_type: Type[T], *args, **kwargs) -> T:
        """
        Resolve dependency untuk interface tertentu.
        
        Args:
            interface_type: Tipe interface yang akan diresolve
            *args, **kwargs: Parameter tambahan untuk konstruktor/factory
            
        Returns:
            Implementasi dari interface yang diminta
            
        Raises:
            KeyError: Jika interface tidak terdaftar
        """
        # Cek apakah ada factory function
        if interface_type in self._factory_functions:
            return self._factory_functions[interface_type](*args, **kwargs)
        
        # Cek dependencies
        if interface_type in self._dependencies:
            implementation = self._dependencies[interface_type]
            
            # Jika implementation sudah berupa instance, langsung kembalikan
            if not isinstance(implementation, type):
                return implementation
                
            # Jika implementation adalah kelas, buat instance baru
            return implementation(*args, **kwargs)
        
        # Jika tidak ditemukan implementasi, raise exception
        raise KeyError(f"Tidak ada implementasi terdaftar untuk {interface_type.__name__}")

    def sync_with_drive(self, config_file, sync_strategy='merge', drive_path=None):
        """
        Sinkronisasi file konfigurasi dengan Google Drive dengan penanganan konflik.
        
        Args:
            config_file: Nama file konfigurasi
            sync_strategy: Strategi sinkronisasi - 'merge' (gabungkan), 'repo_priority' (repo lebih penting),
                        'drive_priority' (drive lebih penting), atau 'newest' (yang terbaru menang)
            drive_path: Path di Drive (default: 'configs/{config_file}')
        
        Returns:
            Tuple (success, message)
        """
        from pathlib import Path
        import os
        import shutil
        import time
        
        env_manager = get_environment_manager(logger=self._logger)
        
        if not env_manager.is_drive_mounted:
            return False, "⚠️ Google Drive tidak ter-mount"
        
        drive_config_path = drive_path or env_manager.drive_path / 'configs' / config_file
        local_config_path = env_manager.get_path(f'configs/{config_file}')
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(str(drive_config_path)), exist_ok=True)
        os.makedirs(os.path.dirname(str(local_config_path)), exist_ok=True)
        
        try:
            local_exists = os.path.exists(str(local_config_path))
            drive_exists = os.path.exists(str(drive_config_path))
            
            # Kedua file tidak ada - tidak ada yang perlu dilakukan
            if not local_exists and not drive_exists:
                return False, f"⚠️ File konfigurasi tidak ditemukan: {config_file}"
                
            # Hanya file lokal ada - salin ke Drive
            if local_exists and not drive_exists:
                shutil.copy2(str(local_config_path), str(drive_config_path))
                if self._logger:
                    self._logger.info(f"⬆️ Upload: Lokal → Drive ({config_file})")
                return True, f"✅ File lokal disalin ke Drive: {config_file}"
                
            # Hanya file Drive ada - salin ke lokal
            if not local_exists and drive_exists:
                shutil.copy2(str(drive_config_path), str(local_config_path))
                if self._logger:
                    self._logger.info(f"⬇️ Download: Drive → Lokal ({config_file})")
                return True, f"✅ File Drive disalin ke lokal: {config_file}"
            
            # Kedua file ada - terapkan strategi konflik
            # Dapatkan timestamp modifikasi
            local_mtime = os.path.getmtime(str(local_config_path))
            drive_mtime = os.path.getmtime(str(drive_config_path))
            
            # Bandingkan konten untuk melihat apakah berbeda
            with open(str(local_config_path), 'rb') as f_local:
                local_content = f_local.read()
            with open(str(drive_config_path), 'rb') as f_drive:
                drive_content = f_drive.read()
                
            if local_content == drive_content:
                if self._logger:
                    self._logger.info(f"✅ Konfigurasi sudah tersinkronisasi: {config_file}")
                return True, f"✅ Konfigurasi sudah tersinkronisasi: {config_file}"
            
            # Buat backup sebelum melakukan perubahan
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if drive_exists:
                drive_backup = f"{str(drive_config_path)}.{timestamp}.bak"
                shutil.copy2(str(drive_config_path), drive_backup)
            if local_exists:
                local_backup = f"{str(local_config_path)}.{timestamp}.bak"
                shutil.copy2(str(local_config_path), local_backup)
            
            # Terapkan strategi konflik yang dipilih
            if sync_strategy == 'repo_priority':
                # Repo lebih penting - selalu gunakan file lokal
                shutil.copy2(str(local_config_path), str(drive_config_path))
                if self._logger:
                    self._logger.info(f"🔄 Sinkronisasi (repo_priority): Lokal → Drive ({config_file})")
                return True, f"✅ Konfigurasi repo diterapkan ke Drive: {config_file}"
                
            elif sync_strategy == 'drive_priority':
                # Drive lebih penting - selalu gunakan file drive
                shutil.copy2(str(drive_config_path), str(local_config_path))
                if self._logger:
                    self._logger.info(f"🔄 Sinkronisasi (drive_priority): Drive → Lokal ({config_file})")
                return True, f"✅ Konfigurasi Drive diterapkan ke lokal: {config_file}"
                
            elif sync_strategy == 'newest':
                # Yang terbaru yang menang
                if local_mtime > drive_mtime:
                    shutil.copy2(str(local_config_path), str(drive_config_path))
                    if self._logger:
                        self._logger.info(f"🔄 Sinkronisasi (newest): Lokal → Drive ({config_file})")
                    return True, f"✅ File lokal lebih baru, disalin ke Drive: {config_file}"
                else:
                    shutil.copy2(str(drive_config_path), str(local_config_path))
                    if self._logger:
                        self._logger.info(f"🔄 Sinkronisasi (newest): Drive → Lokal ({config_file})")
                    return True, f"✅ File Drive lebih baru, disalin ke lokal: {config_file}"
                    
            elif sync_strategy == 'merge':
                # Gabungkan konfigurasi
                # Load kedua konfigurasi
                local_config = self.load_config(str(local_config_path))
                drive_config = self.load_config(str(drive_config_path))
                
                # Gabungkan (merge) konfigurasi, prioritaskan nilai yang tidak None/empty
                merged_config = self._merge_configs_smart(local_config, drive_config)
                
                # Simpan hasil merge ke kedua lokasi
                self.save_config(merged_config, str(local_config_path))
                self.save_config(merged_config, str(drive_config_path))
                
                if self._logger:
                    self._logger.info(f"🔄 Konfigurasi berhasil digabungkan: {config_file}")
                return True, f"✅ Konfigurasi berhasil digabungkan: {config_file}"
                
            else:
                return False, f"❌ Strategi sinkronisasi tidak valid: {sync_strategy}"
                
        except Exception as e:
            if self._logger:
                self._logger.error(f"❌ Error sinkronisasi konfigurasi: {str(e)}")
            return False, f"❌ Error sinkronisasi: {str(e)}"

    def _merge_configs_smart(self, config1, config2):
        """
        Menggabungkan dua konfigurasi dengan strategi smart.
        - Untuk list, gabungkan dengan unik
        - Untuk dict, gabungkan rekursif
        - Untuk nilai skalar, prioritaskan nilai yang tidak None/empty
        
        Args:
            config1: Konfigurasi pertama (biasanya lokal)
            config2: Konfigurasi kedua (biasanya dari Drive)
            
        Returns:
            Konfigurasi yang digabungkan
        """
        import copy
        # Merge konfigurasi secara rekursif dengan strategi smart
        
        # Jika salah satu None, kembalikan yang lain
        if config1 is None:
            return copy.deepcopy(config2)
        if config2 is None:
            return copy.deepcopy(config1)
        
        # Jika keduanya adalah dict, proses secara rekursif
        if isinstance(config1, dict) and isinstance(config2, dict):
            result = copy.deepcopy(config1)
            for key, value in config2.items():
                if key in result:
                    result[key] = self._merge_configs_smart(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result
        
        # Jika keduanya adalah list, gabungkan dan hilangkan duplikat
        if isinstance(config1, list) and isinstance(config2, list):
            # Khusus untuk list sederhana, gabungkan dengan unik
            if all(not isinstance(x, (dict, list)) for x in config1 + config2):
                return list(set(config1 + config2))
            # Untuk list kompleks, gabungkan saja
            return copy.deepcopy(config1) + copy.deepcopy(config2)
        
        # Untuk nilai skalar, prioritaskan nilai yang tidak None/empty
        if config1 == "" or config1 is None or config1 == 0:
            return copy.deepcopy(config2)
        return copy.deepcopy(config1)

    def sync_all_configs(self, sync_strategy='merge'):
        """
        Sinkronisasi semua file konfigurasi YAML antara repository dan Drive.
        
        Args:
            sync_strategy: Strategi sinkronisasi untuk semua file
            
        Returns:
            Dictionary hasil sinkronisasi
        """
        env_manager = get_environment_manager(logger=self._logger)
        
        if not env_manager.is_drive_mounted:
            if self._logger:
                self._logger.warning("⚠️ Google Drive tidak ter-mount, tidak dapat sinkronisasi")
            return {"status": "error", "message": "Google Drive tidak ter-mount"}
        
        # Cari semua file YAML di direktori configs lokal dan Drive
        config_dir_local = env_manager.get_path('configs')
        config_dir_drive = env_manager.drive_path / 'configs'
        
        os.makedirs(str(config_dir_local), exist_ok=True)
        os.makedirs(str(config_dir_drive), exist_ok=True)
        
        local_yamls = set()
        drive_yamls = set()
        
        # Dapatkan file YAML lokal
        for ext in ['.yaml', '.yml']:
            local_yamls.update([f.name for f in config_dir_local.glob(f'*{ext}')])
        
        # Dapatkan file YAML di Drive
        for ext in ['.yaml', '.yml']:
            drive_yamls.update([f.name for f in config_dir_drive.glob(f'*{ext}')])
        
        # Gabungkan daftar file
        all_yamls = local_yamls.union(drive_yamls)
        
        results = {
            "synced": [],
            "failed": [],
            "skipped": []
        }
        
        # Sinkronisasi setiap file
        for yaml_file in all_yamls:
            success, message = self.sync_with_drive(yaml_file, sync_strategy)
            if success:
                results["synced"].append({"file": yaml_file, "message": message})
            else:
                results["failed"].append({"file": yaml_file, "message": message})
        
        if self._logger:
            self._logger.info(f"🔄 Sinkronisasi selesai: {len(results['synced'])} berhasil, {len(results['failed'])} gagal")
        
        return results
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton).
    
    Args:
        base_dir: Direktori root project
        config_file: Path file konfigurasi utama
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance ConfigManager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager