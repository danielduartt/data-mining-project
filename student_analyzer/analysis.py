
import logging
from typing import Dict

class Analyzer:
    def __init__(self, data_processor, visualizer, logger: logging.Logger):
        self.data_processor = data_processor
        self.visualizer = visualizer
        self.logger = logger
    
    def analyze_feature_distributions(self) -> Dict:
        """Analisa a distribuição de todas as features."""
        self.logger.info("Starting feature distribution analysis")
        distribution_stats = {}
        df = self.data_processor.get_preprocessed_data()

        for col in df.columns:
            if col != "Target":
                self.visualizer.plot_feature_distribution(df, col)
                distribution_stats[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                }
        
        return distribution_stats
    
    def analyze_academic_performance(self) -> Dict:
        """Analisa o desempenho acadêmico dos alunos."""
        self.logger.info("Starting academic performance analysis")
        performance_stats = {}
        df = self.data_processor.get_preprocessed_data()

        if "Admission grade" in df.columns:
            performance_stats["admission_grades"] = {
                "mean": df["Admission grade"].mean(),
                "median": df["Admission grade"].median(),
                "std": df["Admission grade"].std(),
            }
            self.visualizer.plot_feature_distribution(df, "Admission grade")
        
        return performance_stats