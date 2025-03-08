# Panduan Migrasi Kode

Dokumen ini membantu Anda memigrasikan kode yang ada ke struktur baru modul `utils/visualization`.

## Perubahan API

### Dari `research.py` Lama ke Struktur Baru

#### Sebelum:

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi
visualizer = ResearchVisualizer()

# Visualisasi eksperimen
result = visualizer.visualize_model_comparison(
    models_df=models_data,
    title="Perbandingan Model",
    output_path="results/comparison.png"
)

# Visualisasi skenario
result = visualizer.visualize_research_scenario(
    scenario_df=scenario_data,
    title="Skenario Penelitian",
    output_path="results/scenario.png"
)

# Analisis
recommendation = result.get('recommendation', '')
```

#### Sesudah:

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi
visualizer = ResearchVisualizer()

# Visualisasi eksperimen
result = visualizer.visualize_experiment_comparison(
    results_df=models_data,
    title="Perbandingan Model",
    filename="comparison.png"
)

# Visualisasi skenario
result = visualizer.visualize_scenario_comparison(
    results_df=scenario_data,
    title="Skenario Penelitian",
    filename="scenario.png"
)

# Analisis
analysis = result['analysis']
recommendation = analysis.get('recommendation', '')
```

### Perubahan Parameter

| Modul Lama | Parameter Lama | Modul Baru | Parameter Baru |
|------------|----------------|------------|----------------|
| `visualize_model_comparison` | `models_df` | `visualize_experiment_comparison` | `results_df` |
| `visualize_model_comparison` | `output_path` | `visualize_experiment_comparison` | `filename` |
| `visualize_research_scenario` | `scenario_df` | `visualize_scenario_comparison` | `results_df` |
| `visualize_research_scenario` | `output_path` | `visualize_scenario_comparison` | `filename` |

### Perubahan Return Value

| Return Value Lama | Return Value Baru |
|-------------------|-------------------|
| `result['recommendation']` | `result['analysis']['recommendation']` |
| `result['metrics']` | `result['analysis']['metrics']` |
| `result['best_model']` | `result['analysis']['best_model']` |

## Strategi Migrasi

### Pendekatan 1: Migrasi Langsung

Ganti semua pemanggilan API lama dengan yang baru.

#### Contoh:

```python
# Sebelum
result = visualizer.visualize_model_comparison(
    models_df=models_data,
    output_path="results/comparison.png"
)
recommendation = result.get('recommendation', '')

# Sesudah
result = visualizer.visualize_experiment_comparison(
    results_df=models_data,
    filename="comparison.png"
)
recommendation = result['analysis'].get('recommendation', '')
```

### Pendekatan 2: Lapisan Kompatibilitas

Buat fungsi wrapper untuk mempertahankan kompatibilitas.

#### Contoh:

```python
def visualize_model_comparison(models_df, output_path=None, **kwargs):
    """Fungsi kompatibilitas untuk API lama."""
    return visualizer.visualize_experiment_comparison(
        results_df=models_df,
        filename=output_path,
        **kwargs
    )

def visualize_research_scenario(scenario_df, output_path=None, **kwargs):
    """Fungsi kompatibilitas untuk API lama."""
    return visualizer.visualize_scenario_comparison(
        results_df=scenario_df,
        filename=output_path,
        **kwargs
    )
```

## Akses Komponen Terpisah

### Menggunakan Visualizer Terpisah

Anda dapat mengakses visualizer individual jika hanya membutuhkan satu jenis visualisasi:

```python
from smartcash.utils.visualization import ExperimentVisualizer, ScenarioVisualizer

# Hanya untuk eksperimen
experiment_visualizer = ExperimentVisualizer(output_dir="results/experiments")
result = experiment_visualizer.visualize_experiment_comparison(...)

# Hanya untuk skenario
scenario_visualizer = ScenarioVisualizer(output_dir="results/scenarios")
result = scenario_visualizer.visualize_scenario_comparison(...)
```

### Menggunakan Analyzer Terpisah

Anda dapat mengakses analyzer untuk melakukan analisis tanpa visualisasi:

```python
from smartcash.utils.visualization.analysis import ExperimentAnalyzer, ScenarioAnalyzer

# Analisis eksperimen
analyzer = ExperimentAnalyzer()
analysis = analyzer.analyze_experiment_results(
    df=data,
    metric_cols=['Akurasi', 'Precision', 'Recall'],
    time_col='Waktu Inferensi'
)

# Analisis skenario
analyzer = ScenarioAnalyzer()
analysis = analyzer.analyze_scenario_results(
    df=data,
    backbone_col='Backbone',
    condition_col='Kondisi'
)
```

## Menggunakan Utilitas

Utilitas dalam modul `research_utils.py` dapat diakses langsung:

```python
from smartcash.utils.visualization.research_utils import clean_dataframe, format_metric_name

clean_df = clean_dataframe(raw_df)
formatted_name = format_metric_name('f1_score')
```

## Perbandingan Fitur

| Fitur | API Lama | API Baru |
|-------|----------|----------|
| Visualisasi model | `visualize_model_comparison()` | `visualize_experiment_comparison()` |
| Visualisasi skenario | `visualize_research_scenario()` | `visualize_scenario_comparison()` |
| Styling DataFrame | `create_styled_df()` | `_create_styled_dataframe()` (dalam visualizer) |
| Plot akurasi | Terintegrasi dalam satu metode | `_create_metrics_plot()` (dalam ExperimentVisualizer) |
| Plot trade-off | Terintegrasi dalam satu metode | `_create_tradeoff_plot()` (dalam ExperimentVisualizer) |
| Analisis model | Terintegrasi dalam visualisasi | `ExperimentAnalyzer.analyze_experiment_results()` |
| Analisis skenario | Terintegrasi dalam visualisasi | `ScenarioAnalyzer.analyze_scenario_results()` |

## Contoh Migrasi Lengkap

### Kode Lama:

```python
from smartcash.utils.visualization import ResearchVisualizer

# Setup
visualizer = ResearchVisualizer()

# Visualisasi
comparison_result = visualizer.visualize_model_comparison(
    models_df=models_df,
    title="Perbandingan Model",
    output_path="results/comparison.png",
    highlight_best=True
)

scenario_result = visualizer.visualize_research_scenario(
    scenario_df=scenario_df,
    title="Skenario Penelitian",
    output_path="results/scenario.png"
)

# Akses hasil
best_model = comparison_result.get('best_model', {}).get('name', 'Unknown')
best_scenario = scenario_result.get('best_scenario', {}).get('name', 'Unknown')

recommendation = comparison_result.get('recommendation', '')
styled_df = comparison_result.get('styled_df')

print(f"Model terbaik: {best_model}")
print(f"Skenario terbaik: {best_scenario}")
print(f"Rekomendasi: {recommendation}")
display(styled_df)
```

### Kode Baru:

```python
from smartcash.utils.visualization import ResearchVisualizer

# Setup
visualizer = ResearchVisualizer()

# Visualisasi
comparison_result = visualizer.visualize_experiment_comparison(
    results_df=models_df,
    title="Perbandingan Model",
    filename="comparison.png",
    highlight_best=True
)

scenario_result = visualizer.visualize_scenario_comparison(
    results_df=scenario_df,
    title="Skenario Penelitian",
    filename="scenario.png"
)

# Akses hasil
best_model = comparison_result['analysis']['best_model'].get('name', 'Unknown')
best_scenario = scenario_result['analysis']['best_scenario'].get('name', 'Unknown')

recommendation = comparison_result['analysis'].get('recommendation', '')
styled_df = comparison_result['styled_df']

print(f"Model terbaik: {best_model}")
print(f"Skenario terbaik: {best_scenario}")
print(f"Rekomendasi: {recommendation}")
display(styled_df)
```

## Kesimpulan

Migrasi ke struktur baru modul `utils/visualization` memungkinkan penggunaan yang lebih fleksibel dan termodularisasi dari berbagai komponen visualisasi dan analisis. Meskipun ada perubahan dalam parameter dan struktur return value, sebagian besar fungsionalitas tetap kompatibel dengan kode lama.