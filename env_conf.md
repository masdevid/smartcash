Refaktor Plan ENV CONFIG MODULE:
1. `cell_1_3_dependency_installer.py` adalah entry point minimalis yang tidak perlu diubah. Pastikan dengan kode seperti contoh dibawah ini antarmuka tidak terjadi double UI (Sudah dihandle oleh initializer)
```
"""
File: smartcash/ui/cells/cell_1_2_env_config.py
Deskripsi: Entry point minimalist untuk environment configuration
"""

from smartcash.ui.setup.env_config import initialize_environment_config_ui
initialize_environment_config_ui()
```
2. Layout struktur baru:
```
┌─────────────────────────────────────┐
│ Header                              │
├─────────────────────────────────────┤
│ Setup Button (Centered)             │
├─────────────────────────────────────┤
│ Status Panel                        │
├─────────────────────────────────────┤
│ Progress Tracker                    │
├─────────────────────────────────────┤
│ Log Accordion                       │
├─────────────────────────────────────┤
│ Environment Summary                 │
├─────────────────────────────────────┤
│ Tips & Requirements (2 kolom)       │
└─────────────────────────────────────┘
```
3. Gunakan shared components yang sudah ada dan sesuaikan handlernya
4. `ui_factory.py` dan `ui_components.py` saat ini membingungkan, cukup gunakan `ui_components.py` untuk quick mental mapping.
5. Hanya ada 3 folders utama, `components`, `handlers`, dan `utils`, jangan ada folder lain.
6. `SilentEnvironmentManager` dan `SystemInfoHelper` lebih baik jadikan handlers. 
7. Reorganisir constant dan utilities menjadi helper functions yang kecil-kecil dan sesuai context nama file
8. `smartcash/common/config/manager.py` sudah ada method sync config, jadi pada tahapan persioapan environment ini setelah membuat symlink folders hanya perlu verifikasi apakah ada config baru yang missing di drive. Semua proses sync dilakukan oleh config manager. Env config hanya melakukan verifikasi ulang
9. Root module hanya berisi file inisialisasi
10. Pastikan semua menerapkan DRY principles dengan mengurangi repetisi code menjadi reusable utils