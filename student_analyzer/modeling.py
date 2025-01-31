import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

class Modeler:
    def __init__(self, data_processor, visualizer, logger: logging.Logger):
        self.data_processor = data_processor
        self.visualizer = visualizer
        self.logger = logger
    
    def perform_feature_selection(self) -> Dict:
        """Realiza a seleção de features usando Random Forest."""
        self.logger.info("Starting feature selection")
        df = self.data_processor.get_preprocessed_data()
        X = df.drop(columns=["Target"])
        y = df["Target"]

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": rf_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance_df.to_dict("records")
    
    def analyze_student_clusters(self) -> Dict:
        """Realiza a análise de clusters de alunos."""
        self.logger.info("Starting student clustering")
        df = self.data_processor.get_preprocessed_data()
        X = df.drop(columns=["Target"])

        # Redução de dimensionalidade com PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Clusterização com KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        # Visualização dos clusters
        self.visualizer.plot_clusters(X_pca, labels)

        return {"cluster_labels": labels.tolist()}