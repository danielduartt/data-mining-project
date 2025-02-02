import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("StudentAnalyzer")


class SemesterAnalyzer:
    """Analyzes patterns and trends across academic semesters."""

    def __init__(self, df: pd.DataFrame, target_names: Dict[int, str]):
        """
        Initialize the semester analyzer.

        Args:
            df: Input DataFrame
            target_names: Dictionary mapping target values to names
        """
        self.df = df
        self.target_names = target_names
        self._identify_semester_columns()

    def _identify_semester_columns(self) -> None:
        """Identify and categorize semester-related columns."""
        semester_cols = [col for col in self.df.columns if "sem" in col.lower()]
        self.enrolled_cols = [col for col in semester_cols if "enrolled" in col]
        self.approved_cols = [col for col in semester_cols if "approved" in col]
        self.semester_count = len(self.enrolled_cols)

    def analyze_semester_patterns(self) -> Dict:
        """
        Analyze patterns and trends across academic semesters.

        Returns:
            Dict: Semester-wise analysis results
        """
        logger.info("Starting semester pattern analysis")
        semester_patterns = {}

        try:
            if not self.enrolled_cols or not self.approved_cols:
                logger.warning("No semester columns found in dataset")
                return {}

            # Calculate semester-wise metrics
            semester_metrics = self._calculate_semester_metrics()
            if semester_metrics:
                semester_patterns["semester_metrics"] = semester_metrics

            # Analyze performance transitions
            transition_patterns = self._analyze_semester_transitions()
            if transition_patterns:
                semester_patterns["transitions"] = transition_patterns

            # Analyze cumulative performance
            cumulative_stats = self._analyze_cumulative_performance()
            if cumulative_stats:
                semester_patterns["cumulative_performance"] = cumulative_stats

            # Generate visualizations
            self._visualize_semester_patterns(semester_metrics, transition_patterns)

            logger.info("Semester pattern analysis completed")
            return semester_patterns

        except Exception as e:
            logger.error(f"Error in semester pattern analysis: {str(e)}")
            return {}

    def _calculate_semester_metrics(self) -> Dict:
        """Calculate metrics for each semester."""
        semester_metrics = {}

        for sem in range(1, self.semester_count + 1):
            enrolled_col = f"Curricular units {sem}st sem (enrolled)"
            approved_col = f"Curricular units {sem}st sem (approved)"

            if enrolled_col in self.df.columns and approved_col in self.df.columns:
                metrics = {
                    "enrollment_mean": float(self.df[enrolled_col].mean()),
                    "approval_mean": float(self.df[approved_col].mean()),
                    "success_rate": float(
                        (self.df[approved_col] / self.df[enrolled_col]).mean()
                    ),
                    "dropout_rate": float((self.df[enrolled_col] == 0).mean()),
                }
                semester_metrics[f"Semester_{sem}"] = metrics

        return semester_metrics

    def _analyze_semester_transitions(self) -> Dict:
        """Analyze transitions in performance between consecutive semesters."""
        transitions = {}

        try:
            for i in range(len(self.enrolled_cols) - 1):
                current_success = (
                    self.df[self.approved_cols[i]] / self.df[self.enrolled_cols[i]]
                ).fillna(0)

                next_success = (
                    self.df[self.approved_cols[i + 1]]
                    / self.df[self.enrolled_cols[i + 1]]
                ).fillna(0)

                # Calculate transition probabilities
                improved = (next_success > current_success).mean()
                declined = (next_success < current_success).mean()
                maintained = (next_success == current_success).mean()

                transitions[f"Semester_{i+1}_to_{i+2}"] = {
                    "improved": float(improved),
                    "declined": float(declined),
                    "maintained": float(maintained),
                    "correlation": float(current_success.corr(next_success)),
                }

        except Exception as e:
            logger.error(f"Error in transition analysis: {str(e)}")

        return transitions

    def _analyze_cumulative_performance(self) -> Dict:
        """Analyze cumulative performance patterns across semesters."""
        try:
            # Calculate cumulative units and success rates
            self.df["total_units_enrolled"] = self.df[self.enrolled_cols].sum(axis=1)
            self.df["total_units_approved"] = self.df[self.approved_cols].sum(axis=1)
            self.df["overall_success_rate"] = (
                self.df["total_units_approved"] / self.df["total_units_enrolled"]
            ).fillna(0)

            # Calculate statistics by target group
            cumulative_stats = {}
            for target, name in self.target_names.items():
                mask = self.df["Target"] == target
                target_stats = {
                    "avg_total_enrolled": float(
                        self.df.loc[mask, "total_units_enrolled"].mean()
                    ),
                    "avg_total_approved": float(
                        self.df.loc[mask, "total_units_approved"].mean()
                    ),
                    "avg_success_rate": float(
                        self.df.loc[mask, "overall_success_rate"].mean()
                    ),
                    "count": int(mask.sum()),
                }
                cumulative_stats[name] = target_stats

            return cumulative_stats

        except Exception as e:
            logger.error(f"Error in cumulative performance analysis: {str(e)}")
            return {}

    def _visualize_semester_patterns(
        self, semester_metrics: Dict, transition_patterns: Dict
    ) -> None:
        """
        Create visualizations for semester patterns analysis.

        Args:
            semester_metrics: Dictionary of semester metrics
            transition_patterns: Dictionary of transition patterns
        """
        try:
            plt.figure(figsize=(15, 10))

            # Plot enrollment and approval trends
            self._plot_enrollment_trends(semester_metrics)

            # Plot success rates
            self._plot_success_rates(semester_metrics)

            # Plot transition patterns
            self._plot_transition_patterns(transition_patterns)

            # Plot dropout rates
            self._plot_dropout_rates(semester_metrics)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error in semester pattern visualization: {str(e)}")

    def _plot_enrollment_trends(self, semester_metrics: Dict) -> None:
        """Plot enrollment and approval trends."""
        plt.subplot(2, 2, 1)
        semesters = list(range(1, len(semester_metrics) + 1))
        enrollment_means = [m["enrollment_mean"] for m in semester_metrics.values()]
        approval_means = [m["approval_mean"] for m in semester_metrics.values()]

        plt.plot(semesters, enrollment_means, "b-o", label="Enrolled")
        plt.plot(semesters, approval_means, "g-o", label="Approved")
        plt.title("Units Enrolled vs Approved by Semester")
        plt.xlabel("Semester")
        plt.ylabel("Average Units")
        plt.legend()
        plt.grid(True)

    def _plot_success_rates(self, semester_metrics: Dict) -> None:
        """Plot success rates by semester."""
        plt.subplot(2, 2, 2)
        semesters = list(range(1, len(semester_metrics) + 1))
        success_rates = [m["success_rate"] for m in semester_metrics.values()]

        plt.plot(semesters, success_rates, "r-o")
        plt.title("Success Rate by Semester")
        plt.xlabel("Semester")
        plt.ylabel("Success Rate")
        plt.grid(True)

    def _plot_transition_patterns(self, transition_patterns: Dict) -> None:
        """Plot transition patterns between semesters."""
        plt.subplot(2, 2, 3)
        transition_semesters = list(range(1, len(transition_patterns) + 1))
        improvements = [t["improved"] for t in transition_patterns.values()]
        declines = [t["declined"] for t in transition_patterns.values()]

        plt.bar(transition_semesters, improvements, label="Improved", alpha=0.6)
        plt.bar(
            transition_semesters,
            declines,
            bottom=improvements,
            label="Declined",
            alpha=0.6,
        )
        plt.title("Performance Transitions Between Semesters")
        plt.xlabel("Transition Period")
        plt.ylabel("Proportion of Students")
        plt.legend()

    def _plot_dropout_rates(self, semester_metrics: Dict) -> None:
        """Plot dropout rates by semester."""
        plt.subplot(2, 2, 4)
        semesters = list(range(1, len(semester_metrics) + 1))
        dropout_rates = [m["dropout_rate"] for m in semester_metrics.values()]

        plt.plot(semesters, dropout_rates, "m-o")
        plt.title("Dropout Rate by Semester")
        plt.xlabel("Semester")
        plt.ylabel("Dropout Rate")
        plt.grid(True)
