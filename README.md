Refaktor `smartcash/ui/training_config/strategy/` ke lokasi baru `smartcash/ui/strategy` menggunakan `training_config.yaml` dan mewarisi `smartcash/ui/initializers/config_cell_initializer.py`
* Struktur file: 
   * components/ui_form
   * components/ui_layout
   * handlers/defaults 
   * handlers/config_handler 
   * strategy_init.py 
   * utils/** (if any consolidated function) 
* Reuse shared components, handlers, dan utils `smartcash/ui/{components, handlers, utils}`
* Setiap save/reset action mengupdate status panel secara informatif dan pastikan hanya ada 1 icon dalam status panel message
* Hanya buat parameter yang pasti dan penting untuk diubah
* Tata layout dengan rapi dan compact menggunakan flexbox dan grid, pastikan tidak menyebabkan overflow yang menyebabkan horizontal scrollbar muncul
* Buat Summary Cards untuk mengetahui config lengkapnya yang tersimpan
* Gunakan one-liner style code
* Hilangkan sync_info
* Terapkan prinsip DRY
* Jangan buat UI fallbacks 
* Jangan buat tests suites dan examples 
* Buat script sh pembuatan folder, struktur file baru beserta `__init__.py` yang straight forward diawal


Refaktor `smartcash/ui/dataset/split/` ke lokasi baru `smartcash/ui/split` menggunakan `split_config.yaml` dan mewarisi `smartcash/ui/initializers/config_cell_initializer.py`
* Struktur file: 
   * components/ui_form
   * components/ui_layout
   * handlers/defaults 
   * handlers/config_handler 
   * split_init.py 
   * utils/** (if any consolidated function) 
* Reuse shared components, handlers, dan utils `smartcash/ui/{components, handlers, utils}`
* Setiap save/reset action mengupdate status panel secara informatif dan pastikan hanya ada 1 icon dalam status panel message
* Hanya buat parameter yang pasti dan penting untuk diubah
* Tata layout dengan rapi dan compact menggunakan flexbox dan grid, pastikan tidak menyebabkan overflow yang menyebabkan horizontal scrollbar muncul
* Buat Summary Cards untuk mengetahui config lengkapnya yang tersimpan
* Hilangkan sync_info
* Gunakan one-liner style code
* Terapkan prinsip DRY
* Jangan buat UI fallbacks 
* Jangan buat tests suites dan examples 
* Buat script sh pembuatan folder, struktur file baru beserta `__init__.py` yang straight forward diawal