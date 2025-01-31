# student_analyzer/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import re

class Visualizer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._setup_visualization_settings()
    
    def _setup_visualization_settings(self):
        """Configura o estilo das visualizações."""
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["figure.dpi"] = 100
        self.logger.debug("Visualization settings configured")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Remove caracteres inválidos de um nome de arquivo."""
        # Substitui caracteres inválidos por underscores
        return re.sub(r'[\\/:*?"<>|\t]', '_', filename)
    
    def plot_feature_distribution(self, df: pd.DataFrame, feature: str):
        """Plota a distribuição de uma feature e salva a figura."""
        self.logger.debug(f"Plotting distribution of feature: {feature}")
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Distribution of {feature}")
        
        # Sanitiza o nome do arquivo e define o caminho de saída
        sanitized_feature = self._sanitize_filename(feature)
        output_path = Path("results/visualizations") / f"{sanitized_feature}_distribution.png"
        
        # Cria a pasta se não existir
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva a figura
        plt.savefig(output_path)
        plt.close()  # Fecha a figura para liberar memória
        self.logger.info(f"Saved {feature} distribution plot to {output_path}")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plota um heatmap de correlação e salva a figura."""
        self.logger.debug("Plotting correlation heatmap...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlations")
        plt.tight_layout()
        
        # Define o caminho de saída
        output_path = Path("results/visualizations") / "correlation_heatmap.png"
        
        # Cria a pasta se não existir
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva a figura
        plt.savefig(output_path)
        plt.close()  # Fecha a figura para liberar memória
        self.logger.info(f"Saved correlation heatmap to {output_path}")
    
    def plot_clusters(self, X_pca: np.ndarray, labels: np.ndarray):
        """Plota clusters em um espaço 2D e salva a figura."""
        self.logger.debug("Plotting clusters...")
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Student Clusters (PCA projection)")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        
        # Define o caminho de saída
        output_path = Path("results/visualizations") / "clusters.png"
        
        # Cria a pasta se não existir
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva a figura
        plt.savefig(output_path)
        plt.close()  # Fecha a figura para liberar memória
        self.logger.info(f"Saved clusters plot to {output_path}")