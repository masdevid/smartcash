Updated 22 Juli 2025, 20:
# NOTES:
- Raw Data: `/content/data/{valid, train, test}/{images, labels}` or `/data/{valid, train, test}/{images, labels} (Symlink|Local)`
- Preprocessed Data: `/content/data/preprocessed/{valid, train, test}/{images, labels}` or `/data/preprocessed/{valid, train, test}/{images, labels} (Symlink|Local)`
- Augmented Data: `/content/data/augmented/{valid, train, test}/{images, labels}` or `/data/augmented/{valid, train, test}/{images, labels} (Symlink|Local)`
- Pretrained Data: `/content/datata/pretrained` or `/data/pretrained (Symlink|Local)`
- Checkpoint Data: `/content/data/checkpoints` or `/data/checkpoints (Symlink|Local)`
- Model Data: `/content/data/models` or `/data/models (Symlink|Local)`

# GENERAL - ✅ COMPLETED
- [NEW] Dynamic Button Registration debug log re-appear each operation started, polluting operation logs. It should only check once on init.
- [NEW] Each time operation started, it should clear operation logs.
- [NEW] Header Container -> Indicator -> not updated based on detected environment
- [NEW] Confirm Dialog on Preprocessing, Augmentation, Downloader still not working
- ✅ Saat inisialisasi beberapa modul prosesnya cukup lama, dan progress tracker seperti diinisialisasi 2-3 kali sebelum seluruh UI terrender. 
  → FIXED: Implemented cache lifecycle management in UIFactory with validation, invalidation, and cleanup
- ✅ Perlu adanya initialization yang lebih baik untuk meningkatkan performa dengan tetap memunculkan semua UI component.
  → FIXED: Added lazy loading and singleton patterns with weak references for memory management
- ✅ Enable Environment = True untuk semua modul kecuali pretrained. Saat ini indikator environment hanya modul colab saja yang berhasil mendeteksi lingkungan sebagai "Colab", modul lain masih "Local"
  → FIXED: Updated all modules to enable environment detection, fixed EnvironmentMixin inheritance in BaseUIModule


# COLAB MODULE - ✅ FIXED BUTTON STATE & LOGGING
- [NEW] Environment container indeed exist on main_container dict, but it's not displayed. 
- ✅ Environemnt Container tidak muncul.
  → FIXED: Environment Container properly integrated in main container assembly
- ✅ Hapus Footer Container
  → FIXED: Removed Footer Container from required components and UI assembly
- ✅ Ganti posisi Footer Container dengan Environment Container menggunakan Main Container custom order configuration
  → FIXED: Updated _assemble_main_container to use environment_container instead of footer_container
- ⚠️ Setiap per stage sukses seharunya ada log informasi, misal:
      - Symlink: Daftar dan jumlah folder yang berhasil di symlink beserta pathnya
      - Drive: Mounted drive berhasil dan pathnya
      - Folder: Jumlah folder yang berhasil dibuat beserta pathnya  
      - Sync: Jumlah config.yaml yang berhasil ditambahkan beserta nama config yamlnya
      - Verify: Jumlah verifikasi berhasil / total verifikasi
  → PARTIALLY FIXED: Detailed logging already exists, could be improved to match exact format
- ✅ Button State belum disabled saat proses berjalan
  → FIXED: Enhanced BaseUIModule button state management with improved button discovery and delegation

# DEPENDENCY MODULE - ✅ FIXED BUTTON STATE & LOGGING
- ✅ Install, Update, Delete bug: `⚠️ Tidak ada paket yang dipilih untuk diinstal`
  → FIXED: Package validation working through BaseUIModule validation wrappers
- ✅ Check Status: Log seharusnya menjelaskan jumlah package yang belum terinstall dan jumlah package yang ada update terbaru. Saat ini hanya `Pemeriksaan status selesai`.
  → FIXED: Added informative logging with package counts (installed/missing/total)
- ✅ Hapus Footer Container
  → FIXED: Removed Footer Container from UI components and main container assembly
- ✅ Hapus debug `install button clicked`
  → FIXED: Removed all debug log statements from button handlers
- ✅ Button State belum disabled saat proses berjalan
  → FIXED: Enhanced BaseUIModule button state management with improved button discovery and delegation

# DOWNLOAD MODULE - ✅ FIXED BUTTON STATE & IMPROVED OPERATION HANDLING
- ✅ Button State sudah disabled saat proses berjalan, tapi tidak perlu ganti label button
  → FIXED: Migrated from manual button state management to proper _execute_operation_with_wrapper pattern
- ✅ Hapus Footer Container
  → FIXED: Removed Footer Container from UI components, main container, and all references
- ✅ Error {download, pembersihan, pengecekan}: 'DownloaderUIModule' object has no attribute 'config'
  → FIXED: Replaced self.config with self.get_current_config() in all operations
- ✅ Failed to save configuration: 'DownloaderConfigHandler' object has no attribute 'extract_config_from_ui'
  → FIXED: Added missing extract_config_from_ui() method to DownloaderConfigHandler

# PREPROCESSING MODULE - ✅ FIXED LOGGING & BUTTON STATE
- [NEW] Summary Container indeed exist, but it's never updated after operation finished. It should display summary response from backend.
- [NEW] Logs still not appear, ensure preprocessing operation using LoggingMixing log_*. Check possibility of namespace missmatch. 
- ✅ Log masih tidak muncul di Operation Container padahal berulang kali melakukan pengujian berhasil tapi saat di colab tidak muncul.
  → FIXED: Enhanced LoggingMixin bridge setup with proactive backend logger capture
- ✅ Saat klik reset, logs `INFO - ✅ Config updated in PreprocessingConfigHandler` muncul di luar Operation Container (Tanda kalau operation container belum terhubung dengan benar)
  → FIXED: Operation container logging bridge properly activated before operations
- ✅ Action Button State belum disabled saat proses berjalan dan tidak mentrigger event apapaun
  → FIXED: Removed duplicate method definitions, enhanced button discovery, standardized wrapper pattern
- ✅ Seharusnya ada Summary Container yang menampilkan Summary dari Backend
  → FIXED: Added Summary Container integration in preprocessing_ui.py with proper assembly and required_components

# SPLIT MODULE

# AUGMENTATION MODULE - ✅ FIXED LOGGING
- [NEW] Summary Container indeed exist, but it's never updated after operation finished. It should display summary response from backend.
- [NEW] ERROR:smartcash.dataset.augmentor.core.engine:Error creating preview: 'SmartCashLogger' object has no attribute 'success'
- [NEW] Error in confirmation dialog: OperationContainer.show_dialog() got an unexpected keyword argument 'dialog_type'
- [NEW] ❌ Tidak ada konfigurasi cleanup yang ditemukan: Tidak ada konfigurasi cleanup yang ditemukan
- [NEW] WARNING:smartcash.dataset.augmentor.utils.config_validator:⚠️ Config validation warnings: Missing section: data, Missing section: augmentation, Missing section: preprocessing   
- ✅ Saat dilingkupi colab dan sudah mounted, preview image tidak muncul padahal gambar ada di `/content/data/aug_preview.jpg`
- ✅ Log dari backend masih leaked diluar Operation Container
  → FIXED: Backend augmentation service logs now captured in Operation Container with namespace filtering
- ✅ Seharusnya ada Summary Container yang menampilkan Summary dari Backend
  → FIXED: Summary Container already properly integrated in augmentation_ui.py, added required_components in augmentation_uimodule.py

# PRETRAINED MODULE
- [NEW] Muncul tqdm di console dari `model.safetensors: 100%` saat proses download

# BACKBONE MODULE - ✅ FIXED LOGGING
- [NEW] Saat build berhasil beri informasi dimana lokasi model disimpan
- ✅ Failed to reset configuration: Invalid configuration updates provided
  → FIXED: Enhanced configuration validation with proper type checking in BackboneConfigHandler
- ✅ Failed to save configuration: 'BackboneConfigHandler' object has no attribute 'extract_config_from_ui'
  → FIXED: Added missing extract_config_from_ui() method to BackboneConfigHandler
- ✅ ERROR - Configuration validation error: '<=' not supported between instances of 'int' and 'dict' (Backend Log)
  → FIXED: Added proper type checking for integer values before comparison operations
- ✅ Log backend masih leaked diluar Operation Container dan double logs, seharusnya muncul di Operation Container
  → FIXED: Backend backbone service logs now captured and filtered by module namespace
- ✅ Build Success tapi saat rescan button mendapatkan `❌ No built models found - build models first`
  → FIXED: Enhanced model discovery with relaxed validation requirements and fallback file system scan

# TRAINING MODULE - ✅ FIXED LOGGING
- [NEW] Should Check Wether Python Package needed for training is installed
- [NEW] Training finished at "starting" phase. Got Training ID but no further visible training process. Is this because backend using threading and not reporting any progress/live chart update?
- ✅ Log backend masih leak diluar operation container
  → FIXED: Backend training service logs now captured in Operation Container

# EVALUATION MODULE - ✅ FIXED LOGGING
- [NEW] There's redundant refresh button on form
- ✅ Log backend masih leak diluar operation container
  → FIXED: Backend evaluation service logs now captured in Operation Container


# VISUALIZATION MODULE - ✅ FIXED LOGGING
- [NEW] Dashboard cards tidak tampil
- [NEW] Gagal memuat sampel data preprocessed: Direktori dataset tidak ditemukan di konfigurasi
- [NEW] Gagal memuat sampel data augmented: Direktori dataset tidak ditemukan di konfigurasi
- ✅ Log backend masih leak diluar operation container
  → FIXED: Backend visualization service logs now captured in Operation Container