# Panduan Restrukturisasi Kode SmartCash

## Latar Belakang

Modul `utils/visualization` pada SmartCash sebelumnya memiliki file `research.py` yang terlalu besar dan kompleks, menangani berbagai fungsi yang tidak terkait erat. Pendekatan ini mempersulit pemeliharaan, pengujian, dan pengembangan lebih lanjut.

Restrukturisasi telah dilakukan untuk mengatasi masalah ini dengan membagi kode menjadi komponen-komponen yang lebih kecil dan fokus mengikuti prinsip Single Responsibility dari SOLID.

## Prinsip Restrukturisasi

Restrukturisasi dilakukan berdasarkan prinsip-prinsip berikut:

1. **Single Responsibility**: Setiap kelas dan file hanya memiliki satu tanggung jawab utama
2. **Open/Closed**: Kode mudah diperluas tanpa perlu memodifikasi kode yang ada
3. **Interface Segregation**: Antarmuka dipecah menjadi bagian khusus untuk kebutuhan spesifik
4. **Dependency Inversion**: Komponen bergantung pada abstraksi, bukan implementasi konkret

## Perubahan Utama

### 1. Pemecahan File Utama

File besar `research.py` dipecah menjadi beberapa modul yang lebih kecil:

- **research_base.py**: Kelas dasar dengan fungsionalitas umum untuk visualisasi penelitian
- **experiment_visualizer.py**: Khusus untuk visualisasi eksperimen
- **scenario_visualizer.py**: Khusus untuk visualisasi skenario penelitian 
- **research_utils.py**: Fungsi pembantu yang tidak terkait dengan kelas tertentu
- **analysis/**: Paket baru untuk fungsi analisis data

### 2. Pemisahan Tanggung Jawab

Tanggung jawab dibagi menjadi beberapa bagian utama:

- **Visualisasi**: Bertanggung jawab untuk membuat dan menampilkan grafik
- **Analisis**: Bertanggung jawab untuk analisis data dan memberikan rekomendasi
- **Utilitas**: Bertanggung jawab untuk fungsi pembantu umum

### 3. Penambahan Paket Analysis

Kelas-kelas analisis dipisahkan ke dalam paket terpisah:

- **experiment_analyzer.py**: Menganalisis hasil eksperimen model
- **scenario_analyzer.py**: Menganalisis hasil skenario penelitian

## Proses Restrukturisasi

### Langkah 1: Identifikasi Tanggung Jawab

Kode `research.py` dianalisis untuk mengidentifikasi berbagai tanggung jawab:

1. Visualisasi hasil eksperimen
2. Visualisasi skenario penelitian
3. Analisis data eksperimen
4. Analisis data skenario
5. Fungsi utilitas umum

### Langkah 2: Pembagian Kode

Kode dibagi menjadi file terpisah berdasarkan tanggung jawab yang teridentifikasi:

```
research.py -> research.py (lebih ramping) + modul-modul baru
```

### Langkah 3: Pembuatan Kelas Dasar

Kelas dasar `BaseResearchVisualizer` dibuat untuk menampung fungsionalitas umum:

```python
class BaseResearchVisualizer:
    """Kelas dasar untuk visualisasi hasil penelitian dengan fungsionalitas umum."""
    
    def __init__(self, output_dir, logger):
        # Inisialisasi dasar
        ...
    
    def _create_styled_dataframe(self, df):
        # Fungsionalitas umum untuk styling DataFrame
        ...
        
    def _add_tradeoff_regions(self, ax):
        # Fungsionalitas umum untuk plotting
        ...
        
    def save_visualization(self, fig, filename):
        # Fungsionalitas umum untuk menyimpan hasil
        ...
```

### Langkah 4: Pembuatan Visualizer Khusus

Visualizer khusus dibuat untuk jenis analisis tertentu:

```python
class ExperimentVisualizer(BaseResearchVisualizer):
    """Visualisasi dan analisis hasil eksperimen untuk perbandingan model."""
    
    def visualize_experiment_comparison(self, results_df, ...):
        # Visualisasi khusus untuk eksperimen
        ...
        
class ScenarioVisualizer(BaseResearchVisualizer):
    """Visualisasi dan analisis hasil skenario penelitian."""
    
    def visualize_scenario_comparison(self, results_df, ...):
        # Visualisasi khusus untuk skenario
        ...
```

### Langkah 5: Pemisahan Analisis

Logika analisis dipisahkan ke dalam kelas terpisah:

```python
class ExperimentAnalyzer:
    """Kelas untuk menganalisis hasil eksperimen dan memberikan rekomendasi."""
    
    def analyze_experiment_results(self, df, metric_cols, time_col):
        # Analisis eksperimen
        ...
        
class ScenarioAnalyzer:
    """Kelas untuk menganalisis hasil skenario penelitian dan memberikan rekomendasi."""
    
    def analyze_scenario_results(self, df, backbone_col, condition_col):
        # Analisis skenario
        ...
```

### Langkah 6: Integrasi

Kelas utama `ResearchVisualizer` diperbarui untuk mengintegrasikan semua komponen:

```python
class ResearchVisualizer(BaseResearchVisualizer):
    """
    Visualisasi dan analisis hasil penelitian dengan berbagai jenis grafik.
    Mengintegrasikan komponen-komponen visualisasi eksperimen dan skenario.
    """
    
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.experiment_visualizer = ExperimentVisualizer(...)
        self.scenario_visualizer = ScenarioVisualizer(...)
    
    def visualize_experiment_comparison(self, results_df, ...):
        return self.experiment_visualizer.visualize_experiment_comparison(...)
    
    def visualize_scenario_comparison(self, results_df, ...):
        return self.scenario_visualizer.visualize_scenario_comparison(...)
```

### Langkah 7: Pembaruan __init__.py

File `__init__.py` diperbarui untuk mengekspor semua komponen yang diperlukan:

```python
from smartcash.utils.visualization.research import ResearchVisualizer
from smartcash.utils.visualization.experiment_visualizer import ExperimentVisualizer
from smartcash.utils.visualization.scenario_visualizer import ScenarioVisualizer
from smartcash.utils.visualization.analysis import ExperimentAnalyzer, ScenarioAnalyzer
...
```

## Dampak Perubahan

### 1. Dampak Positif

1. **Kode lebih terstruktur** dan lebih mudah dipahami
2. **Pemeliharaan lebih mudah** karena file lebih kecil dan terfokus
3. **Pengujian lebih mudah** karena komponen dapat diuji secara terpisah
4. **Perluasan lebih mudah** karena fitur baru dapat ditambahkan tanpa mengubah kode yang ada
5. **Penggunaan kembali lebih fleksibel** karena komponen dapat digunakan secara terpisah

### 2. Kompatibilitas

- API publik tetap dipertahankan untuk memastikan kode yang ada masih berfungsi
- Kelas dan fungsi yang sama masih tersedia melalui `__init__.py`

## Rekomendasi untuk Pengembangan ke Depan

1. **Pisahkan komponen lain**: Terapkan pendekatan yang sama untuk modul lain yang besar
2. **Standarisasi API**: Seragamkan parameter dan return value antar modul
3. **Buat unit test**: Tambahkan unit test untuk masing-masing komponen
4. **Dokumentasi inline**: Tambahkan DocStrings yang lebih lengkap
5. **Refaktor modul detection.py dan metrics.py**: Terapkan pola yang sama untuk modul ini

## Contoh Refaktor Lanjutan

```python
# Sebelum
class BigClass:
    def method1(self):
        # Kode untuk visualisasi
        # Kode untuk analisis
        # Kode untuk utilitas
        
# Sesudah
class Visualizer:
    def visualize(self):
        # Kode untuk visualisasi
        
class Analyzer:
    def analyze(self):
        # Kode untuk analisis
        
class Utils:
    def utility_function(self):
        # Kode untuk utilitas
```

## Kesimpulan

Restrukturisasi kode pada modul `utils/visualization` telah meningkatkan kualitas kode dengan mengikuti prinsip-prinsip desain yang baik. Perubahan ini memberikan dasar yang solid untuk pengembangan fitur lebih lanjut dan pemeliharaan yang lebih mudah di masa depan.