"""
File: smartcash/utils/visualization/research_analysis.py
Author: Alfrida Sabar
Deskripsi: Modul analisis untuk hasil penelitian dengan berbagai teknik analisis
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


class ScenarioAnalyzer:
    """Kelas untuk menganalisis hasil skenario penelitian dan memberikan rekomendasi."""
    
    def analyze_scenario_results(
        self,
        df: pd.DataFrame,
        backbone_col: Optional[str],
        condition_col: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analisis hasil skenario penelitian.
        
        Args:
            df: DataFrame hasil skenario
            backbone_col: Kolom backbone
            condition_col: Kolom kondisi
            
        Returns:
            Dict berisi analisis dan rekomendasi
        """
        analysis = {
            'best_scenario': {},
            'backbone_comparison': {},
            'condition_comparison': {},
            'recommendation': ''
        }
        
        # Jika tidak ada data, kembalikan analisis kosong
        if df.empty:
            analysis['recommendation'] = "Data tidak cukup untuk analisis"
            return analysis
        
        # Identifikasi kolom metrik
        metric_cols = self._get_metric_columns(df)
        
        # Jika tidak ada metrik, kembalikan analisis kosong
        if not metric_cols:
            analysis['recommendation'] = "Tidak ada metrik untuk analisis"
            return analysis
        
        # Pilih metrik representatif
        repr_metric = self._get_representative_metric(metric_cols)
        
        # Identifikasi skenario terbaik
        analysis['best_scenario'] = self._identify_best_scenario(
            df, repr_metric, backbone_col, condition_col
        )
        
        # Analisis berdasarkan backbone
        if backbone_col and backbone_col in df.columns:
            analysis['backbone_comparison'] = self._analyze_backbone_performance(
                df, backbone_col, metric_cols, repr_metric
            )
        
        # Analisis berdasarkan kondisi
        if condition_col and condition_col in df.columns:
            analysis['condition_comparison'] = self._analyze_condition_performance(
                df, condition_col, metric_cols, repr_metric
            )
        
        # Buat rekomendasi berdasarkan analisis
        analysis['recommendation'] = self._generate_scenario_recommendation(analysis)
        
        return analysis
    
    def _get_metric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifikasi kolom metrik dalam DataFrame."""
        return [col for col in df.columns if col in 
               ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']]
    
    def _get_representative_metric(self, metric_cols: List[str]) -> str:
        """Pilih metrik representatif dari daftar metrik."""
        if 'mAP' in metric_cols:
            return 'mAP'
        elif 'F1-Score' in metric_cols:
            return 'F1-Score'
        else:
            return metric_cols[0]
    
    def _identify_best_scenario(
        self, 
        df: pd.DataFrame, 
        repr_metric: str, 
        backbone_col: Optional[str],
        condition_col: Optional[str]
    ) -> Dict[str, Any]:
        """Identifikasi skenario terbaik berdasarkan metrik terpilih."""
        result = {}
        
        if 'Skenario' in df.columns:
            best_scenario_idx = df[repr_metric].idxmax()
            best_scenario = df.loc[best_scenario_idx, 'Skenario']
            best_scenario_metric = df.loc[best_scenario_idx, repr_metric]
            
            result['name'] = best_scenario
            result['metric'] = repr_metric
            result['value'] = best_scenario_metric
            
            # Tambahkan info tambahan jika tersedia
            if backbone_col and backbone_col in df.columns:
                result['backbone'] = df.loc[best_scenario_idx, backbone_col]
            
            if condition_col and condition_col in df.columns:
                result['condition'] = df.loc[best_scenario_idx, condition_col]
                
        return result
    
    def _analyze_backbone_performance(
        self, 
        df: pd.DataFrame, 
        backbone_col: str, 
        metric_cols: List[str], 
        repr_metric: str
    ) -> Dict[str, Any]:
        """Analisis performa berdasarkan backbone."""
        result = {}
        
        # Bandingkan rata-rata metrik per backbone
        backbone_metrics = df.groupby(backbone_col)[metric_cols].mean()
        
        # Identifikasi backbone terbaik dan terburuk
        best_backbone = backbone_metrics[repr_metric].idxmax()
        worst_backbone = backbone_metrics[repr_metric].idxmin()
        
        # Hitung selisih performa
        best_backbone_metric = backbone_metrics.loc[best_backbone, repr_metric]
        worst_backbone_metric = backbone_metrics.loc[worst_backbone, repr_metric]
        backbone_diff_pct = ((best_backbone_metric - worst_backbone_metric) / worst_backbone_metric) * 100
        
        result['best'] = {
            'name': best_backbone,
            'metrics': {metric: backbone_metrics.loc[best_backbone, metric] for metric in metric_cols}
        }
        
        result['worst'] = {
            'name': worst_backbone,
            'metrics': {metric: backbone_metrics.loc[worst_backbone, metric] for metric in metric_cols}
        }
        
        result['diff_percent'] = backbone_diff_pct
        
        return result
    
    def _analyze_condition_performance(
        self, 
        df: pd.DataFrame, 
        condition_col: str, 
        metric_cols: List[str], 
        repr_metric: str
    ) -> Dict[str, Any]:
        """Analisis performa berdasarkan kondisi."""
        result = {}
        
        # Bandingkan rata-rata metrik per kondisi
        condition_metrics = df.groupby(condition_col)[metric_cols].mean()
        
        # Identifikasi kondisi terbaik dan terburuk
        best_condition = condition_metrics[repr_metric].idxmax()
        worst_condition = condition_metrics[repr_metric].idxmin()
        
        # Hitung selisih performa
        best_condition_metric = condition_metrics.loc[best_condition, repr_metric]
        worst_condition_metric = condition_metrics.loc[worst_condition, repr_metric]
        condition_diff_pct = ((best_condition_metric - worst_condition_metric) / worst_condition_metric) * 100
        
        result['best'] = {
            'name': best_condition,
            'metrics': {metric: condition_metrics.loc[best_condition, metric] for metric in metric_cols}
        }
        
        result['worst'] = {
            'name': worst_condition,
            'metrics': {metric: condition_metrics.loc[worst_condition, metric] for metric in metric_cols}
        }
        
        result['diff_percent'] = condition_diff_pct
        
        return result
    
    def _generate_scenario_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Buat rekomendasi berdasarkan analisis skenario."""
        recommendations = []
        
        # Rekomendasi skenario terbaik
        if 'name' in analysis['best_scenario']:
            best_scenario = analysis['best_scenario']['name']
            metric = analysis['best_scenario']['metric']
            value = analysis['best_scenario']['value']
            recommendations.append(f"Skenario terbaik adalah {best_scenario} dengan {metric}={value:.2f}%.")
        
        # Rekomendasi backbone
        if 'best' in analysis['backbone_comparison']:
            best_bb = analysis['backbone_comparison']['best']['name']
            worst_bb = analysis['backbone_comparison']['worst']['name']
            diff_pct = analysis['backbone_comparison']['diff_percent']
            
            if diff_pct > 10:  # Perbedaan signifikan
                recommendations.append(f"Backbone {best_bb} menunjukkan performa yang lebih baik dibandingkan {worst_bb} dengan peningkatan {diff_pct:.1f}% pada {metric}.")
            else:
                recommendations.append(f"Perbedaan performa antara backbone {best_bb} dan {worst_bb} relatif kecil ({diff_pct:.1f}%).")
        
        # Rekomendasi kondisi
        if 'best' in analysis['condition_comparison']:
            best_cond = analysis['condition_comparison']['best']['name']
            worst_cond = analysis['condition_comparison']['worst']['name']
            diff_pct = analysis['condition_comparison']['diff_percent']
            
            if diff_pct > 10:  # Perbedaan signifikan
                recommendations.append(f"Model menunjukkan performa yang lebih baik pada kondisi {best_cond} dibandingkan {worst_cond} dengan peningkatan {diff_pct:.1f}% pada {metric}.")
            else:
                recommendations.append(f"Model relatif stabil terhadap perubahan kondisi antara {best_cond} dan {worst_cond} (perbedaan {diff_pct:.1f}%).")
        
        # Tambahkan rekomendasi final
        if recommendations:
            recommendation = " ".join(recommendations)
            
            # Tambahkan saran implementasi
            if 'best' in analysis['backbone_comparison'] and 'name' in analysis['best_scenario']:
                best_bb = analysis['backbone_comparison']['best']['name']
                best_scenario = analysis['best_scenario']['name']
                recommendation += f" Untuk implementasi SmartCash, direkomendasikan menggunakan model dengan backbone {best_bb} pada skenario {best_scenario}."
            
            return recommendation
        else:
            return "Tidak cukup data untuk memberikan rekomendasi."