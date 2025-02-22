import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, List

logger = logging.getLogger("StudentAnalyzer")


class CorrelationAnalyzer:
    """Analyzes correlations and relationships between features."""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
    ):
        """
        Initialize the analyzer with dataset and feature lists.

        Args:
            df: Input DataFrame
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
        """
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.correlation_threshold = 0.5
        self.significance_threshold = 0.05

    def analyze_correlations(self) -> Dict:
        """
        Analyze correlations between features and identify significant relationships.

        Returns:
            Dict: Correlation analysis results including significant correlations
                 and categorical associations
        """
        logger.info("Starting correlation analysis")
        correlation_analysis = {}

        try:
            # Analyze numeric correlations
            numeric_correlations = self._analyze_numeric_correlations()
            if numeric_correlations:
                correlation_analysis["significant_correlations"] = numeric_correlations

            # Analyze categorical associations
            categorical_associations = self._analyze_categorical_associations()
            if categorical_associations:
                correlation_analysis["categorical_associations"] = (
                    categorical_associations
                )

            logger.info("Correlation analysis completed")
            return correlation_analysis

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}

    def _analyze_numeric_correlations(self) -> Dict:
        """
        Analyze correlations between numeric features.

        Returns:
            Dict: Significant correlations between numeric features
        """
        logger.debug("Analyzing numeric correlations")
        significant_corr = {}

        try:
            # Calculate correlation matrix
            numeric_corr = self.df[self.numeric_features].corr()

            # Plot correlation heatmap
            self._plot_correlation_matrix(numeric_corr)

            # Store significant correlations
            for i in range(len(numeric_corr.columns)):
                for j in range(i + 1, len(numeric_corr.columns)):
                    corr_value = numeric_corr.iloc[i, j]
                    if abs(corr_value) > self.correlation_threshold:
                        feature_pair = (
                            f"{numeric_corr.columns[i]}_{numeric_corr.columns[j]}"
                        )
                        significant_corr[feature_pair] = {
                            "correlation": float(corr_value)
                        }

            return significant_corr

        except Exception as e:
            logger.error(f"Error in numeric correlation analysis: {str(e)}")
            return {}

    def _analyze_categorical_associations(self) -> Dict:
        """
        Analyze associations between categorical features using chi-square tests.

        Returns:
            Dict: Significant associations between categorical features
        """
        logger.debug("Analyzing categorical associations")
        categorical_associations = {}

        try:
            for col1 in self.categorical_features:
                for col2 in self.categorical_features:
                    if col1 < col2:  # Avoid duplicate combinations
                        contingency = pd.crosstab(self.df[col1], self.df[col2])
                        chi2, p_value = stats.chi2_contingency(contingency)[:2]

                        if p_value < self.significance_threshold:
                            feature_pair = f"{col1}_{col2}"
                            categorical_associations[feature_pair] = {
                                "chi2": float(chi2),
                                "p_value": float(p_value),
                            }

            return categorical_associations

        except Exception as e:
            logger.error(f"Error in categorical association analysis: {str(e)}")
            return {}

    def _plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            corr_matrix: Correlation matrix to visualize
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlations")
        plt.tight_layout()
        plt.show()

    def get_feature_correlations(self, feature: str) -> Dict[str, float]:
        """
        Get correlations for a specific feature with all other numeric features.

        Args:
            feature: Feature name to analyze

        Returns:
            Dict: Correlations with other features
        """
        if feature not in self.numeric_features:
            logger.warning(f"Feature {feature} not found in numeric features")
            return {}

        correlations = {}
        for other_feature in self.numeric_features:
            if other_feature != feature:
                corr = self.df[feature].corr(self.df[other_feature])
                if abs(corr) > self.correlation_threshold:
                    correlations[other_feature] = float(corr)

        return correlations
