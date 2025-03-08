"""
File: smartcash/utils/visualization/analysis/experiment_analyzer.py
Author: Alfrida Sabar
Deskripsi: Kelas analisis untuk hasil eksperimen dengan berbagai metode analisis dan rekomendasi
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Any

class ExperimentAnalyzer:
    """Kelas untuk menganalisis hasil eksperimen dan memberikan rekomendasi."""
    
    def analyze_experiment_results(
        self, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        time_col: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analisis hasil eksperimen dan berikan rekomendasi.
        
        Args:
            df: DataFrame hasil eksperimen
            metric_cols: Kolom metrik
            time_col: Kolom waktu
            
        Returns:
            Dict berisi analisis dan rekomendasi
        """
        analysis = {
            'best_model': {},
            'metrics': {},
            'performance': {},
            'recommendation': ''
        }
        
        # Jika tidak ada data atau metrik, kembalikan analisis kosong
        if df.empty or not metric_cols:
            analysis['recommendation'] = "Data tidak cukup untuk analisis"
            return analysis
        
        # Identifikasi model terbaik berdasarkan metrik utama
        repr_metric = self._get_representative_metric(metric_cols)
        
        best_model_idx = df[repr_metric].idxmax()
        best_model_row = df.loc[best_model_idx]
        
        # Identifikasi model tercepat
        fastest_model_idx = None
        fastest_model_row = None
        
        if time_col and time_col in df.columns:
            fastest_model_idx = df[time_col].idxmin()
            fastest_model_row = df.loc[fastest_model_idx]
        
        # Identifikasi kolom untuk model/backbone
        model_col = self._find_model_column(df)
        
        # Tentukan model terbaik
        analysis['best_model'] = self._identify_best_model(
            best_model_row, 
            repr_metric, 
            model_col, 
            best_model_idx
        )
        
        # Tentukan model tercepat jika tersedia
        if fastest_model_row is not None:
            analysis['performance'] = self._identify_fastest_model(
                fastest_model_row, 
                time_col, 
                model_col, 
                fastest_model_idx
            )
        
        # Analisis metrik
        analysis['metrics'] = self._calculate_metrics_statistics(df, metric_cols)
        
        # Tentukan rekomendasi
        analysis['recommendation'] = self._generate_recommendation(
            analysis, 
            best_model_row, 
            fastest_model_row, 
            repr_metric, 
            time_col
        )
        
        return analysis
    
    def _get_representative_metric(self, metric_cols: List[str]) -> str:
        """Pilih metrik representatif dari daftar metrik."""
        if 'mAP' in metric_cols:
            return 'mAP'
        elif 'F1-Score' in metric_cols:
            return 'F1-Score'
        else:
            return metric_cols[0]
    
    def _find_model_column(self, df: pd.DataFrame) -> Optional[str]:
        """Temukan kolom yang berisi nama model."""
        candidates = ['Model', 'Backbone', 'Arsitektur']
        return next((col for col in candidates if col in df.columns), None)
    
    def _identify_best_model(
        self, 
        best_row: pd.Series, 
        metric: str, 
        model_col: Optional[str], 
        idx: Any
    ) -> Dict[str, Any]:
        """Identifikasi model terbaik berdasarkan metrik terpilih."""
        result = {
            'metric': metric,
            'value': best_row[metric]
        }
        
        if model_col:
            result['name'] = best_row[model_col]
        else:
            result['name'] = f"Model {idx}"
            
        return result
    
    def _identify_fastest_model(
        self, 
        fastest_row: pd.Series, 
        time_col: str, 
        model_col: Optional[str], 
        idx: Any
    ) -> Dict[str, Any]:
        """Identifikasi model tercepat berdasarkan waktu inferensi."""
        result = {
            'fastest_time': fastest_row[time_col]
        }
        
        if model_col:
            result['fastest'] = fastest_row[model_col]
        else:
            result['fastest'] = f"Model {idx}"
            
        return result
    
    def _calculate_metrics_statistics(
        self, 
        df: pd.DataFrame, 
        metric_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Hitung statistik untuk setiap metrik."""
        metrics_stats = {}
        
        for metric in metric_cols:
            metrics_stats[metric] = {
                'mean': df[metric].mean(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max()
            }
            
        return metrics_stats
    
    def _generate_recommendation(
        self, 
        analysis: Dict[str, Any], 
        best_model_row: pd.Series, 
        fastest_model_row: Optional[pd.Series], 
        repr_metric: str, 
        time_col: Optional[str]
    ) -> str:
        """Buat rekomendasi berdasarkan analisis."""
        best_model_name = analysis['best_model']['name']
        best_metric_value = analysis['best_model']['value']
        
        if fastest_model_row is None or time_col is None:
            # Tidak ada data waktu
            return f"Model terbaik adalah {best_model_name} dengan {repr_metric}={best_metric_value:.2f}%. Tidak tersedia data waktu inferensi untuk analisis trade-off performa."
        
        fastest_model_name = analysis['performance']['fastest']
        fastest_time = analysis['performance']['fastest_time']
        
        if best_model_row.name == fastest_model_row.name:
            # Model terbaik juga tercepat
            return f"Model terbaik adalah {best_model_name} dengan {repr_metric}={best_metric_value:.2f}% dan waktu inferensi paling cepat ({fastest_time:.2f}ms). Model ini direkomendasikan untuk semua kasus penggunaan."
        
        # Model terbaik bukan yang tercepat
        best_time = best_model_row[time_col]
        time_diff_pct = ((best_time - fastest_time) / fastest_time) * 100
        
        fastest_metric = fastest_model_row[repr_metric]
        metric_diff_pct = ((best_metric_value - fastest_metric) / fastest_metric) * 100
        
        if time_diff_pct > 30 and metric_diff_pct < 5:
            # Perbedaan waktu signifikan tetapi perbedaan metrik kecil
            return f"Model tercepat ({fastest_model_name}) direkomendasikan untuk aplikasi real-time karena hanya {metric_diff_pct:.1f}% lebih rendah dalam {repr_metric} tetapi {time_diff_pct:.1f}% lebih cepat dibandingkan model terbaik."
        elif metric_diff_pct > 10:
            # Perbedaan metrik signifikan
            return f"Model terbaik ({best_model_name}) direkomendasikan untuk akurasi tertinggi dengan {repr_metric}={best_metric_value:.2f}%. Untuk aplikasi yang membutuhkan kecepatan tinggi, pertimbangkan model {fastest_model_name} yang {time_diff_pct:.1f}% lebih cepat tetapi dengan akurasi {metric_diff_pct:.1f}% lebih rendah."
        else:
            # Rekomendasi umum
            return f"Model terbaik adalah {best_model_name} dengan {repr_metric}={best_metric_value:.2f}%. Untuk kasus penggunaan dengan keterbatasan sumber daya, model {fastest_model_name} bisa menjadi alternatif dengan performa waktu yang lebih baik."