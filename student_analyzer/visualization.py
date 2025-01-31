import matplotlib.pyplot as plt
import seaborn as sns
import logging

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
    
    def plot_feature_distribution(self, df: pd.DataFrame, feature: str):
        """Plota a distribuição de uma feature."""
        self.logger.debug(f"Plotting distribution of feature: {feature}")
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plota um heatmap de correlação."""
        self.logger.debug("Plotting correlation heatmap...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlations")
        plt.tight_layout()
        plt.show()
    
    def plot_clusters(self, X_pca: np.ndarray, labels: np.ndarray):
        """Plota clusters em um espaço 2D."""
        self.logger.debug("Plotting clusters...")
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Student Clusters (PCA projection)")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()