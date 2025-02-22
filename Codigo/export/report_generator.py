import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("StudentAnalyzer")


class ReportGenerator:
    """Generates comprehensive analysis reports and summaries."""

    def __init__(self, results: Dict, df: pd.DataFrame):
        """
        Initialize the report generator.

        Args:
            results: Dictionary containing analysis results
            df: DataFrame with the analyzed data
        """
        self.results = results
        self.df = df
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_summary_report(self, output_path: Path) -> None:
        """
        Generate a comprehensive summary report of the analysis.

        Args:
            output_path: Path where the report should be saved
        """
        try:
            with open(output_path / "summary_report.txt", "w") as f:
                self._write_header(f)
                self._write_dataset_overview(f)
                self._write_key_findings(f)
                self._write_recommendations(f)

            logger.info(f"Summary report generated successfully at {output_path}")

        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            raise

    def _write_header(self, f) -> None:
        """Write report header."""
        f.write("Student Performance Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _write_dataset_overview(self, f) -> None:
        """Write dataset overview section."""
        f.write("Dataset Overview:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total students: {len(self.df)}\n")
        f.write(f"Time period analyzed: {self._get_time_period()}\n")
        f.write(f"Number of features analyzed: {self._get_feature_count()}\n")
        f.write(self._get_feature_breakdown())
        f.write("\n")

    def _write_key_findings(self, f) -> None:
        """Write key findings section."""
        f.write("\nKey Findings:\n")
        f.write("-" * 20 + "\n")

        # Add dropout statistics
        self._write_dropout_statistics(f)

        # Add risk assessment
        self._write_risk_assessment(f)

        # Add performance metrics
        self._write_performance_metrics(f)

        # Add correlation findings
        self._write_correlation_findings(f)

    def _write_dropout_statistics(self, f) -> None:
        """Write dropout-related statistics."""
        if "dropout_patterns" in self.results:
            dropout_rate = (self.df["Target"] == 0).mean() * 100
            f.write(f"Overall dropout rate: {dropout_rate:.1f}%\n")

            if "temporal_patterns" in self.results["dropout_patterns"]:
                f.write("\nDropout rates by semester:\n")
                for sem, stats in self.results["dropout_patterns"][
                    "temporal_patterns"
                ].items():
                    f.write(f"  {sem}: {stats['rate']*100:.1f}%\n")

    def _write_risk_assessment(self, f) -> None:
        """Write risk assessment findings."""
        if (
            "risk_assessment" in self.results
            and "risk_distribution" in self.results["risk_assessment"]
        ):
            f.write("\nRisk Distribution:\n")
            for level, pct in self.results["risk_assessment"][
                "risk_distribution"
            ].items():
                f.write(f"{level} risk: {pct:.1f}%\n")

    def _write_performance_metrics(self, f) -> None:
        """Write academic performance metrics."""
        if "academic_performance" in self.results:
            f.write("\nAcademic Performance:\n")
            if "admission_grades" in self.results["academic_performance"]:
                stats = self.results["academic_performance"]["admission_grades"]
                f.write(f"Average admission grade: {stats['mean']:.2f}\n")
                f.write(f"Standard deviation: {stats['std']:.2f}\n")

            if "semester_success_rates" in self.results["academic_performance"]:
                f.write("\nSemester Success Rates:\n")
                for sem, rates in self.results["academic_performance"][
                    "semester_success_rates"
                ].items():
                    f.write(f"  {sem}: {rates['mean_success_rate']*100:.1f}%\n")

    def _write_correlation_findings(self, f) -> None:
        """Write correlation analysis findings."""
        if "correlations" in self.results:
            f.write("\nSignificant Correlations:\n")
            if "significant_correlations" in self.results["correlations"]:
                for pair, stats in self.results["correlations"][
                    "significant_correlations"
                ].items():
                    f.write(f"  {pair}: {stats['correlation']:.2f}\n")

    def _write_recommendations(self, f) -> None:
        """Write recommendations based on analysis results."""
        f.write("\nRecommendations:\n")
        f.write("-" * 20 + "\n")

        # Early intervention recommendations
        if self._should_recommend_early_intervention():
            f.write("1. Implement early intervention program:\n")
            f.write("   - Monitor first-semester performance closely\n")
            f.write("   - Provide additional support for struggling students\n")
            f.write("   - Establish mentor programs for at-risk students\n")

        # Academic support recommendations
        if self._should_recommend_academic_support():
            f.write("\n2. Enhance academic support:\n")
            f.write("   - Provide targeted tutoring in challenging areas\n")
            f.write("   - Develop study skills workshops\n")
            f.write("   - Create peer study groups\n")

        # General recommendations
        f.write("\n3. General recommendations:\n")
        f.write("   - Monitor student progress regularly\n")
        f.write("   - Maintain communication with at-risk students\n")
        f.write("   - Review and adjust support programs as needed\n")

    def _should_recommend_early_intervention(self) -> bool:
        """Determine if early intervention should be recommended."""
        if "first_semester_success_rate" in self.df.columns:
            low_performers = (self.df["first_semester_success_rate"] < 0.6).mean() * 100
            return low_performers > 20
        return False

    def _should_recommend_academic_support(self) -> bool:
        """Determine if additional academic support should be recommended."""
        if "Admission grade" in self.df.columns:
            return self.df["Admission grade"].std() > 2
        return False

    def _get_time_period(self) -> str:
        """Get the analyzed time period."""
        if "Date" in self.df.columns:
            start_date = self.df["Date"].min()
            end_date = self.df["Date"].max()
            return f"{start_date} to {end_date}"
        return "Not available"

    def _get_feature_count(self) -> int:
        """Get total number of features analyzed."""
        numeric_count = len(
            [
                col
                for col in self.df.columns
                if pd.api.types.is_numeric_dtype(self.df[col])
            ]
        )
        categorical_count = len(
            [
                col
                for col in self.df.columns
                if pd.api.types.is_categorical_dtype(self.df[col])
            ]
        )
        return numeric_count + categorical_count

    def _get_feature_breakdown(self) -> str:
        """Get breakdown of feature types."""
        numeric_features = [
            col
            for col in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[col])
        ]
        categorical_features = [
            col
            for col in self.df.columns
            if pd.api.types.is_categorical_dtype(self.df[col])
        ]

        return (
            f"Numeric features: {len(numeric_features)}\n"
            f"Categorical features: {len(categorical_features)}\n"
        )
