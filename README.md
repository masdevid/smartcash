Perbaikan berikutnya:
- `ROBOFLOW_API_KEY` harus autodetect colab secret diawal
- Opsi "Organisasi dataset" tidak diperlukan karena sudah pasti TRUE.
- Action button tidak mentrigger apapun. Pastikan tiap click, buka log_output, reset/clear lognya dan reset state progress_tracking. 
- Selalu konfirmasi dulu sebelum memulai actions
- Button save dan reset belum menampilkan log dan update status panelnya. 
- Sesuaikan warna header dengan tema warna pada ui_logger_namespace.py
- Form masih muncul horizontal scrollbar, dan input text masih overflow. Pastikan menggunakan flex layout. 
- Buat form menjadi dua kolom. Kolom kiri seksi Dataset Information. Kolom kanan Storage Settings dan Checkbox Options.
- Integrasikan dengan `smartcash/dataset/**` dengan tepat.
- Gunakan one-liner style code
- Keep DRY

Perbaiki urutan layout form pada `smartcash/ui/dataset/downloader/components/**`
- Form 2 kolom
- Save Reset Button
- Area Konfirmasi
- Action Buttons
- Progress Tracker
- Log Output
Gunakan one-liner style code. Sesuaikan handler atau init yang terpengaruh.