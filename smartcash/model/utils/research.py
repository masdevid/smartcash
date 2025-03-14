
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan DataFrame hasil penelitian.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame yang sudah dibersihkan
    """
    # Buat salinan
    clean_df = df.copy()
    
    # Konversi nilai numerik
    for col in clean_df.columns:
        # Cek apakah kolom berisi data numerik
        if clean_df[col].dtype == object:
            try:
                # Coba konversi kolom ke tipe numerik
                numeric_values = pd.to_numeric(clean_df[col], errors='coerce')
                
                # Jika sebagian besar bisa dikonversi, gunakan hasil konversi
                if numeric_values.notna().sum() > 0.5 * len(numeric_values):
                    clean_df[col] = numeric_values
            except:
                pass
    
    # Buang baris dengan semua NaN
    clean_df = clean_df.dropna(how='all')
    
    # Buang kolom dengan semua NaN
    clean_df = clean_df.dropna(axis=1, how='all')
    
    # Reset index jika baris dibuang
    if len(clean_df) < len(df):
        clean_df = clean_df.reset_index(drop=True)
    
    return clean_df

def format_metric_name(name: str) -> str:
    """
    Format nama metrik untuk tampilan yang lebih baik.
    
    Args:
        name: Nama metrik
        
    Returns:
        Nama metrik yang diformat
    """
    # Perbaiki nama metrik umum
    if name.lower() == 'map':
        return 'mAP'
    elif name.lower() == 'f1_score' or name.lower() == 'f1score':
        return 'F1-Score'
    
    # Pisahkan kata menggunakan underscore
    if '_' in name:
        parts = name.split('_')
        return ' '.join(part.capitalize() for part in parts)
    
    # Kapitalisasi huruf pertama
    return name.capitalize()

def find_common_metrics(dfs: List[pd.DataFrame]) -> List[str]:
    """
    Temukan metrik yang ada di semua DataFrame.
    
    Args:
        dfs: List DataFrame
        
    Returns:
        List nama metrik yang umum
    """
    metric_candidates = ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']
    
    if not dfs:
        return []
    
    # Temukan metrik yang ada di semua DataFrame
    common_metrics = []
    for metric in metric_candidates:
        if all(metric in df.columns for df in dfs):
            common_metrics.append(metric)
    
    return common_metrics

def add_tradeoff_annotation(
    ax: plt.Axes,
    x: float,
    y: float,
    value: float,
    color: str = 'black',
    fontsize: int = 10
) -> None:
    """
    Tambahkan anotasi untuk plot trade-off.
    
    Args:
        ax: Axes untuk anotasi
        x: Koordinat x
        y: Koordinat y
        value: Nilai untuk ditampilkan
        color: Warna teks
        fontsize: Ukuran font
    """
    ax.annotate(
        f"{value:.1f}",
        (x, y),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=fontsize,
        color=color
    )

def create_benchmark_table(
    metrics: Dict[str, Dict[str, float]],
    models: List[str],
    metric_names: List[str],
    include_avg: bool = True
) -> pd.DataFrame:
    """
    Buat tabel benchmark dari metrik.
    
    Args:
        metrics: Dictionary metrik per model
        models: List nama model
        metric_names: List nama metrik
        include_avg: Sertakan nilai rata-rata
        
    Returns:
        DataFrame hasil benchmark
    """
    # Siapkan data untuk tabel
    data = []
    
    for model in models:
        row = {'Model': model}
        
        for metric in metric_names:
            if model in metrics and metric in metrics[model]:
                row[metric] = metrics[model][metric]
            else:
                row[metric] = np.nan
        
        data.append(row)
    
    # Buat DataFrame
    df = pd.DataFrame(data)
    
    # Tambahkan rata-rata jika diminta
    if include_avg and len(models) > 1:
        avg_row = {'Model': 'Average'}
        
        for metric in metric_names:
            avg_row[metric] = df[metric].mean()
        
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    return df

def create_win_rate_table(results_df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Buat tabel win rate untuk berbagai model dan metrik.
    
    Args:
        results_df: DataFrame hasil eksperimen
        metric_cols: Kolom metrik
        
    Returns:
        DataFrame win rate
    """
    # Identifikasi kolom model
    model_col = next((col for col in results_df.columns if col in 
                     ['Model', 'Backbone', 'Arsitektur']), None)
    
    if not model_col or model_col not in results_df.columns:
        return pd.DataFrame()
    
    # Identifikasi model
    models = results_df[model_col].unique()
    
    # Siapkan tabel win rate
    win_rates = {model: {metric: 0 for metric in metric_cols} for model in models}
    
    # Hitung win rate untuk setiap metrik
    for metric in metric_cols:
        # Cari nilai terbaik untuk metrik ini
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, model_col]
        
        # Tambahkan win untuk model terbaik
        win_rates[best_model][metric] += 1
    
    # Konversi ke DataFrame
    win_rate_data = []
    
    for model in models:
        row = {model_col: model}
        
        for metric in metric_cols:
            row[metric] = win_rates[model][metric]
        
        # Tambahkan total
        row['Total'] = sum(win_rates[model].values())
        
        win_rate_data.append(row)
    
    # Buat DataFrame
    win_rate_df = pd.DataFrame(win_rate_data)
    
    # Urutkan berdasarkan total win
    win_rate_df = win_rate_df.sort_values('Total', ascending=False)
    
    return win_rate_df