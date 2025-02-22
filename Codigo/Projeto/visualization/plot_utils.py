import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .plot_config import (
    get_figure_size,
    get_color_scheme,
    style_axis_labels,
    style_title,
)


class PlotUtils:
    """Utility class for creating standardized plots."""

    @staticmethod
    def plot_numeric_distribution(
        data: pd.DataFrame,
        column: str,
        target_col: str = "Target",
        target_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Create distribution plots for numeric features.

        Args:
            data: Input DataFrame
            column: Column name to plot
            target_col: Target column name
            target_names: Dictionary mapping target values to names
        """
        plt.figure(figsize=get_figure_size("distribution"))

        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=column, kde=True)
        style_title(plt.gca(), f"Distribution of {column}")

        # Boxplot by target
        plt.subplot(1, 2, 2)
        sns.boxplot(data=data, x=target_col, y=column)
        if target_names:
            plt.xticks(range(len(target_names)), target_names.values())
        style_title(plt.gca(), f"{column} by Target")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_categorical_distribution(
        data: pd.DataFrame,
        column: str,
        target_col: str = "Target",
        target_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Create distribution plots for categorical features.

        Args:
            data: Input DataFrame
            column: Column name to plot
            target_col: Target column name
            target_names: Dictionary mapping target values to names
        """
        plt.figure(figsize=get_figure_size("categorical"))

        # Value counts plot
        plt.subplot(1, 2, 1)
        value_counts = data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        style_title(plt.gca(), f"Distribution of {column}")
        style_axis_labels(plt.gca(), column, "Count", rotation=45)

        # Stacked bar plot by target
        plt.subplot(1, 2, 2)
        contingency = pd.crosstab(data[column], data[target_col])
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0)
        contingency_pct.plot(kind="bar", stacked=True)
        if target_names:
            plt.legend(target_names.values())
        style_title(plt.gca(), f"{column} by Target")
        style_axis_labels(plt.gca(), column, "Proportion", rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(
        corr_matrix: pd.DataFrame,
        title: str = "Feature Correlations",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Create correlation matrix heatmap.

        Args:
            corr_matrix: Correlation matrix
            title: Plot title
            figsize: Optional figure size override
        """
        plt.figure(figsize=figsize or get_figure_size("correlation"))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap=get_color_scheme("correlation")
        )
        style_title(plt.gca(), title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        x_col: str = "importance",
        y_col: str = "feature",
        title: str = "Feature Importance",
        top_n: int = 10,
    ) -> None:
        """
        Create feature importance bar plot.

        Args:
            importance_df: DataFrame with feature importance scores
            x_col: Column name for importance values
            y_col: Column name for feature names
            title: Plot title
            top_n: Number of top features to show
        """
        plt.figure(figsize=get_figure_size("default"))
        sns.barplot(data=importance_df.head(top_n), x=x_col, y=y_col)
        style_title(plt.gca(), title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_performance_progression(
        data: pd.Series, title: str, xlabel: str, ylabel: str, marker: str = "o"
    ) -> None:
        """
        Create performance progression line plot.

        Args:
            data: Series of performance values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            marker: Marker style for line plot
        """
        plt.figure(figsize=get_figure_size("performance"))
        plt.plot(range(1, len(data) + 1), data.values, f"-{marker}")
        style_title(plt.gca(), title)
        style_axis_labels(plt.gca(), xlabel, ylabel)
        plt.grid(True)
        plt.show()

    @staticmethod
    def save_plot(
        plot_path: Path, filename: str, dpi: int = 300, bbox_inches: str = "tight"
    ) -> None:
        """
        Save current plot to file.

        Args:
            plot_path: Directory path for saving plot
            filename: Name of the plot file
            dpi: Resolution of the saved plot
            bbox_inches: Bounding box parameter for saving
        """
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path / filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()

    @staticmethod
    def plot_risk_distribution(
        risk_scores: pd.Series,
        risk_validation: pd.DataFrame,
        target_names: Dict[int, str],
    ) -> None:
        """
        Create risk distribution plots.

        Args:
            risk_scores: Series of risk scores
            risk_validation: Risk validation results
            target_names: Dictionary mapping target values to names
        """
        plt.figure(figsize=get_figure_size("distribution"))

        # Risk score distribution
        plt.subplot(1, 2, 1)
        sns.histplot(risk_scores, bins=20)
        style_title(plt.gca(), "Distribution of Risk Scores")

        # Risk levels by target
        plt.subplot(1, 2, 2)
        sns.heatmap(risk_validation, annot=True, fmt="d", cmap=get_color_scheme("risk"))
        style_title(plt.gca(), "Risk Levels vs Actual Outcomes")
        plt.tight_layout()
        plt.show()
