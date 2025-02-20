# File: smartcash/interface/menu/eval.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi menu evaluasi model

from smartcash.handlers.evaluation_handler import EvaluationHandler
from smartcash.interface.menu.base import BaseMenu, MenuItem
   
class EvaluationMenu(BaseMenu):
    """Menu evaluasi model."""
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        self.evaluator = EvaluationHandler(config=self.config_manager.current_config)

        items = [
            MenuItem(
                title="Evaluasi Model Reguler",
                action=self.evaluate_regular,
                description="Evaluasi model pada dataset testing standar",
                category="Evaluasi"
            ),
            MenuItem(
                title="Evaluasi Skenario Penelitian",
                action=self.evaluate_research,
                description="Evaluasi model pada skenario penelitian",
                category="Evaluasi"
            ),
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        ]
        
        super().__init__("Menu Evaluasi Model", items)
        
    def evaluate_regular(self) -> bool:
        """Evaluasi model pada dataset testing standar."""
        try:
            
            evaluator = EvaluationHandler(config=self.config_manager.current_config)
            self.display.show_success("Memulai evaluasi...")
            
            results = evaluator.evaluate(eval_type='regular')
            
            # Display results
            for model_name, metrics in results.items():
                self.display.show_success(f"\nHasil untuk {model_name}:")
                for metric, value in metrics.items():
                    if metric != 'confusion_matrix':
                        self.display.show_success(f"{metric}: {value:.4f}")
                        
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal melakukan evaluasi: {str(e)}")
            return True
            
    def evaluate_research(self) -> bool:
        """Evaluasi model pada skenario penelitian."""
        try:
            
            evaluator = EvaluationHandler(config=self.config_manager.current_config)
            self.display.show_success("Memulai skenario penelitian...")
            
            results = evaluator.evaluate(eval_type='research')
            
            # Display results
            df = results['research_results']
            for _, row in df.iterrows():
                self.display.show_success(f"\n{row['Skenario']}:")
                self.display.show_success(f"Akurasi: {row['Akurasi']:.4f}")
                self.display.show_success(f"F1-Score: {row['F1-Score']:.4f}")
                self.display.show_success(
                    f"Waktu Inferensi: {row['Waktu Inferensi']*1000:.1f}ms"
                )
                
            output_dir = self.config_manager.current_config.get('output_dir', 'outputs')
            self.display.show_success(
                f"\nHasil lengkap disimpan di: {output_dir}/research_results.csv"
            )
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal menjalankan skenario: {str(e)}")
            return True