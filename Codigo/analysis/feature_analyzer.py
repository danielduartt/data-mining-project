import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List

logger = logging.getLogger("StudentAnalyzer")


class FeatureAnalyzer:
    """Analyzes and visualizes feature distributions and relationships."""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
        target_names: Dict[int, str],
    ):
        """
        Initialize the analyzer with dataset and feature lists.

        Args:
            df: Input DataFrame
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            target_names: Mapping of target values to their names
        """
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_names = target_names

    def analyze_feature_distributions(self) -> Dict:
        """
        Analyze and visualize the distribution of all features.

        Returns:
            Dict: Distribution statistics for all features
        """
        logger.info("Starting feature distribution analysis")
        distribution_stats = {}

        try:
            # Analyze numeric features
            distribution_stats.update(self._analyze_numeric_features())

            # Analyze categorical features
            distribution_stats.update(self._analyze_categorical_features())

            logger.info("Feature distribution analysis completed")
            return distribution_stats

        except Exception as e:
            logger.error(f"Error in feature distribution analysis: {str(e)}")
            return {}

    def _analyze_numeric_features(self) -> Dict:
        """Analyze numeric feature distributions."""
        stats = {}

        for col in self.numeric_features:
            logger.debug(f"Analyzing distribution of numeric feature: {col}")

            # Calculate statistics
            stats[col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "skew": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis()),
                "quantiles": self.df[col].quantile([0.25, 0.5, 0.75]).to_dict(),
            }

            # Visualize distribution
            self._plot_numeric_feature(col)

        return stats

    def _analyze_categorical_features(self) -> Dict:
        """Analyze categorical feature distributions."""
        stats = {}

        for col in self.categorical_features:
            if col != "Target":
                logger.debug(f"Analyzing distribution of categorical feature: {col}")

                # Calculate frequencies
                value_counts = self.df[col].value_counts()
                frequencies = (value_counts / len(self.df) * 100).round(2)

                # Create contingency table with target
                contingency = pd.crosstab(self.df[col], self.df["Target"])
                chi2, p_value = stats.chi2_contingency(contingency)[:2]

                stats[col] = {
                    "frequencies": frequencies.to_dict(),
                    "modal_category": value_counts.index[0],
                    "unique_values": self.df[col].nunique(),
                    "chi2_with_target": float(chi2),
                    "p_value_with_target": float(p_value),
                }

                # Visualize distribution
                self._plot_categorical_feature(col, value_counts, contingency)

        return stats

    def _plot_numeric_feature(self, col: str) -> None:
        """
        Create distribution plots for numeric features.

        Args:
            col: Feature name to plot
        """
        plt.figure(figsize=(12, 6))

        # Create subplot with histogram and KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.df, x=col, kde=True)
        plt.title(f"Distribution of {col}")

        # Create boxplot by target
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x="Target", y=col)
        plt.title(f"{col} by Target")
        plt.xticks(range(len(self.target_names)), self.target_names.values())

        plt.tight_layout()
        plt.show()

    def _plot_categorical_feature(
        self, col: str, value_counts: pd.Series, contingency: pd.DataFrame
    ) -> None:
        """
        Create distribution plots for categorical features.

        Args:
            col: Feature name to plot
            value_counts: Value counts for the feature
            contingency: Contingency table with target
        """
        plt.figure(figsize=(12, 5))

        # Create frequency plot
        plt.subplot(1, 2, 1)
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)

        # Create stacked bar plot with target
        plt.subplot(1, 2, 2)
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0)
        contingency_pct.plot(kind="bar", stacked=True)
        plt.title(f"{col} by Target")
        plt.legend(self.target_names.values())
        plt.tight_layout()
        plt.show()
