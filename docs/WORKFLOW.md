# Panduan Komprehensif Alur Kerja Git untuk SmartCash Detector

## 📌 Prasyarat
- Git terinstal di komputer
- Akses repository SmartCash Detector
- Koneksi internet

## 🚀 Langkah-Langkah Awal

### Clone Repository
```bash
# Clone menggunakan HTTPS
# git clone https://github.com/[username]/smartcash-detector.git
git clone https://github.com/masdevid/smartcash.git

# Clone menggunakan SSH (disarankan)
git clone https://github.com/masdevid/smartcash.git
# Masuk ke direktori proyek
cd smartcash
```

### Konfigurasi Remote Repository
```bash
# Tambahkan repository upstream (repository asli)
# git remote add upstream https://github.com/[pemilik-asli]/smartcash-detector.git
git remote add upstream https://github.com/masdevid/smartcash.git

# Verifikasi remote
git remote -v
```

## 🌿 Alur Kerja Cabang (Branch Workflow)

### Jenis Cabang
1. `main` (Cabang Utama)
   - Kode produksi stabil
   - Dilindungi dengan pembatasan merge

2. `develop` (Cabang Pengembangan)
   - Cabang integrasi untuk fitur yang sedang dikembangkan
   - Persiapan untuk rilis berikutnya

3. Cabang Fitur (`feature/`)
   - Untuk pengembangan fitur spesifik
   - Dibuat dari cabang `develop`

### Membuat Cabang Develop
```bash
# Pastikan Anda di cabang main
git checkout main

# Tarik pembaruan terbaru
git pull upstream main

# Buat cabang develop
git checkout -b develop

# Dorong cabang develop ke repository
git push -u origin develop
```

### Membuat Cabang Fitur
```bash
# Pastikan Anda di cabang develop terbaru
git checkout develop
git pull upstream develop

# Buat cabang fitur
git checkout -b feature/nama-fitur-spesifik

# Contoh: Menambah fitur baru di backbone
git checkout -b feature/efficientnet-backbone-optimization
```

## 🔧 Bekerja pada Fitur

### Commit dengan Baik
```bash
# Tambahkan perubahan
git add .

# Commit dengan pesan yang jelas
git commit -m "fitur(backbone): Tambahkan optimasi compound scaling EfficientNet"
```

### Tips Commit yang Baik
- Gunakan awalan: `fitur`, `perbaikan`, `dok`, `gaya`, `refaktor`, `uji`, `tugas`
- Gunakan lingkup opsional dalam tanda kurung
- Tuliskan pesan deskriptif singkat namun jelas
- Gunakan emoji untuk memberikan konteks tambahan (opsional)

## 🔀 Mengirim Perubahan

### Dorong Cabang ke Repository
```bash
# Dorong cabang fitur
git push -u origin feature/nama-fitur-spesifik
```

### Membuat Pull Request (PR)
1. Buka repository GitHub proyek
2. Klik "Compare & pull request"
3. Pilih:
   - Base branch: `develop`
   - Compare branch: `feature/nama-fitur-spesifik`
4. Tulis deskripsi PR dengan detail:
   - Apa yang berubah
   - Alasan perubahan
   - Tangkapan layar/bukti (jika perlu)

## 🔄 Sinkronisasi dengan Upstream

### Mengambil Pembaruan Terbaru
```bash
# Pindah ke cabang develop
git checkout develop

# Ambil pembaruan dari upstream
git fetch upstream

# Gabungkan pembaruan
git merge upstream/develop

# Dorong ke repository
git push origin develop
```

## 🛡️ Praktik Terbaik

### Sebelum Memulai Pengembangan
- Selalu update cabang `develop`
- Buat cabang fitur dari `develop`
- Pastikan tidak ada konflik

### Selama Pengembangan
- Commit sering dan dengan pesan yang jelas
- Uji kode sebelum commit
- Gunakan pre-commit hooks
- Hindari merge konflik dengan sering sinkronisasi

### Setelah Pull Request
- Tunggu tinjauan dari maintainer
- Siap melakukan perubahan berdasarkan komentar
- Gunakan `git commit --amend` untuk perbaikan kecil

## ⚠️ Hindari

- Jangan commit langsung ke `main`
- Jangan unggah data sensitif
- Hindari file dengan ukuran besar
- Jangan commit file generated/temporary

## 🤝 Etika Kontribusi
- Hormati kode yang sudah ada
- Ikuti panduan gaya kode proyek
- Berkomunikasi dengan sopan
- Bersedia menerima umpan balik

## 📚 Sumber Belajar Lanjut
- [Git Pro Book](https://git-scm.com/book/id/v2)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Dokumentasi Resmi Git](https://git-scm.com/docs)