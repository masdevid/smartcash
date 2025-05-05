"""
File: smartcash/common/constants/log_messages.py
Deskripsi: Konstanta pesan log untuk memastikan konsistensi dan mengurangi duplikasi
"""

# Status messages dengan emoji
STATUS_SUCCESS = "✅ {message}"
STATUS_INFO = "ℹ️ {message}" 
STATUS_WARNING = "⚠️ {message}"
STATUS_ERROR = "❌ {message}"
STATUS_DEBUG = "🔍 {message}"
STATUS_PROGRESS = "🔄 {message}"

# Operation result messages
OPERATION_SUCCESS = "✅ {operation} berhasil"
OPERATION_FAILED = "❌ {operation} gagal: {reason}"
OPERATION_SKIPPED = "⚠️ {operation} dilewati: {reason}"
OPERATION_STARTED = "🚀 Memulai {operation}"
OPERATION_COMPLETED = "✅ {operation} selesai dalam {duration}"

# File operation messages
FILE_NOT_FOUND = "❌ File {path} tidak ditemukan"
DIRECTORY_NOT_FOUND = "❌ Direktori {path} tidak ditemukan"
FILE_COPY_SUCCESS = "✅ File berhasil disalin ke: {destination}"
FILE_MOVE_SUCCESS = "✅ File berhasil dipindahkan ke: {destination}"
FILE_BACKUP_SUCCESS = "✅ Backup berhasil dibuat: {path}"
FILE_BACKUP_ERROR = "❌ Gagal membuat backup {source}: {error}"

# Config operation messages
CONFIG_LOADED = "✅ Konfigurasi dimuat dari: {path}"
CONFIG_SAVED = "✅ Konfigurasi disimpan ke: {path}"
CONFIG_MERGED = "✅ Konfigurasi berhasil digabungkan"
CONFIG_ERROR = "❌ Error saat {operation} konfigurasi: {error}"
CONFIG_IDENTICAL = "ℹ️ Konfigurasi sudah identik: {name}"
CONFIG_SYNC_SUCCESS = "✅ Konfigurasi berhasil disinkronisasi {direction}"
CONFIG_SYNC_ERROR = "❌ Error saat sinkronisasi konfigurasi: {error}"

# Drive operation messages
DRIVE_NOT_MOUNTED = "⚠️ Google Drive tidak terpasang"
DRIVE_SYNC_SUCCESS = "✅ Sinkronisasi Drive berhasil: {count} file disalin"
DRIVE_SYNC_SKIPPED = "ℹ️ Sinkronisasi Drive tidak diaktifkan dalam konfigurasi"
DRIVE_PATH_IDENTICAL = "⚠️ Path lokal sama dengan drive: {path}, gunakan path lain"

# Progress tracking messages
PROGRESS_STARTED = "🚀 {description} dimulai"
PROGRESS_COMPLETED = "✅ {description} selesai"
PROGRESS_UPDATE = "🔄 Progres: {percentage}% - {message}"
PROGRESS_STEP = "🔄 Step {current}/{total}: {step}"

# Component initialization messages
COMPONENT_INITIALIZED = "✅ {component} terinisialisasi"
COMPONENT_ERROR = "❌ Error inisialisasi {component}: {error}"

# Common validation messages
VALIDATION_ERROR = "❌ {field}: {message}"
VALIDATION_REQUIRED = "Field ini wajib diisi"
VALIDATION_FORMAT = "Format {field} tidak valid"
VALIDATION_RANGE = "Nilai harus antara {min} dan {max}"