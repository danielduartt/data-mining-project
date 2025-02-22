import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List

logger = logging.getLogger("StudentAnalyzer")


class PerformanceAnalyzer:
    """Analyzes academic performance patterns and progression."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with dataset.

        Args:
            df: Input DataFrame with academic performance data
        """
        self.df = df
        self.demographic_cols = ["Gender", "Age at enrollment", "Scholarship holder"]

    def analyze_academic_performance(self) -> Dict:
        """
        Analyze academic performance patterns including grades and progression.

        Returns:
            Dict: Comprehensive academic performance statistics and patterns
        """
        logger.info("Starting academic performance analysis")
        performance_stats = {}

        try:
            # Analyze admission grades if available
            if "Admission grade" in self.df.columns:
                performance_stats.update(self._analyze_admission_grades())

            # Analyze semester performance
            semester_stats = self._analyze_semester_performance()
            if semester_stats:
                performance_stats["semester_success_rates"] = semester_stats

            # Analyze performance by demographics
            demographic_stats = self._analyze_demographic_performance()
            if demographic_stats:
                performance_stats.update(demographic_stats)

            logger.info("Academic performance analysis completed")
            return performance_stats

        except Exception as e:
            logger.error(f"Error in academic performance analysis: {str(e)}")
            return {}

    def _analyze_admission_grades(self) -> Dict:
        """Analyze admission grade distribution."""
        logger.debug("Analyzing admission grade distribution")

        stats = {
            "admission_grades": {
                "mean": self.df["Admission grade"].mean(),
                "median": self.df["Admission grade"].median(),
                "std": self.df["Admission grade"].std(),
                "quantiles": self.df["Admission grade"]
                .quantile([0.25, 0.5, 0.75])
                .to_dict(),
            }
        }

        # Visualize grade distribution by outcome
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x="Target", y="Admission grade")
        plt.title("Admission Grades by Student Outcome")
        plt.xlabel("Student Outcome")
        plt.ylabel("Admission Grade")
        plt.xticks(range(3), ["Dropout", "Graduate", "Enrolled"])
        plt.show()

        return stats

    def _analyze_semester_performance(self) -> Dict:
        """Analyze performance patterns across semesters."""
        success_rates = {}
        semester_cols = [col for col in self.df.columns if "sem" in col.lower()]

        if not semester_cols:
            return {}

        logger.debug("Analyzing semester performance")

        # Calculate semester-wise success rates
        for sem in range(1, len(semester_cols) // 2 + 1):
            enrolled_col = f"Curricular units {sem}st sem (enrolled)"
            approved_col = f"Curricular units {sem}st sem (approved)"

            if enrolled_col in self.df.columns and approved_col in self.df.columns:
                success_rate = (self.df[approved_col] / self.df[enrolled_col]).fillna(0)

                success_rates[f"Semester_{sem}"] = {
                    "mean_success_rate": success_rate.mean(),
                    "median_success_rate": success_rate.median(),
                    "std_success_rate": success_rate.std(),
                }

        # Visualize semester progression
        if success_rates:
            self._plot_semester_progression(success_rates)

        return success_rates

    def _analyze_demographic_performance(self) -> Dict:
        """Analyze performance patterns by demographic factors."""
        demographic_stats = {}

        for col in self.demographic_cols:
            if col not in self.df.columns:
                continue

            logger.debug(f"Analyzing performance by {col}")

            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Correlation analysis for numeric demographics
                correlation = self.df[col].corr(self.df["Admission grade"])
                demographic_stats[f"performance_by_{col.lower()}"] = {
                    "correlation": correlation
                }
            else:
                # Group analysis for categorical demographics
                group_stats = (
                    self.df.groupby(col)["Admission grade"]
                    .agg(["mean", "std"])
                    .to_dict()
                )
                demographic_stats[f"performance_by_{col.lower()}"] = group_stats

                # Visualize distribution by demographic factor
                self._plot_demographic_performance(col)

        return demographic_stats

    def _plot_semester_progression(self, success_rates: Dict) -> None:
        """
        Plot success rate progression across semesters.

        Args:
            success_rates: Dictionary containing success rates by semester
        """
        plt.figure(figsize=(12, 6))
        semester_means = pd.DataFrame(success_rates).applymap(
            lambda x: x["mean_success_rate"]
        )
        semester_means.plot(kind="line", marker="o")
        plt.title("Average Success Rate Progression by Semester")
        plt.xlabel("Semester")
        plt.ylabel("Average Success Rate")
        plt.grid(True)
        plt.show()

    def _plot_demographic_performance(self, col: str) -> None:
        """
        Plot performance distribution by demographic factor.

        Args:
            col: Demographic factor column name
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x=col, y="Admission grade")
        plt.title(f"Performance Distribution by {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
