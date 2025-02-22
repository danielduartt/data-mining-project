import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("StudentAnalyzer")


class DataExporter:
    """Handles export of analysis results to various formats."""

    def __init__(self, results: Dict):
        """
        Initialize the data exporter.

        Args:
            results: Dictionary containing analysis results
        """
        self.results = results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_results(self, output_dir: str = "analysis_results") -> None:
        """
        Export analysis results to various formats.

        Args:
            output_dir: Directory for output files
        """
        logger.info(f"Exporting analysis results to {output_dir}")

        try:
            # Create output directory with timestamp
            output_path = Path(output_dir) / self.timestamp
            output_path.mkdir(parents=True, exist_ok=True)

            # Export results to different formats
            self._export_json(output_path)
            self._export_csv_files(output_path)
            self._export_excel(output_path)

            logger.info(f"Results exported successfully to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise

    def _export_json(self, output_path: Path) -> None:
        """
        Export results to JSON format.

        Args:
            output_path: Path to save the JSON file
        """
        try:
            json_path = output_path / "analysis_results.json"
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=4, default=str)
            logger.debug(f"Results exported to JSON: {json_path}")

        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise

    def _export_csv_files(self, output_path: Path) -> None:
        """
        Export specific results to CSV files.

        Args:
            output_path: Directory to save CSV files
        """
        try:
            # Export feature importance if available
            if (
                "dropout_patterns" in self.results
                and "feature_importance" in self.results["dropout_patterns"]
            ):
                importance_df = pd.DataFrame(
                    self.results["dropout_patterns"]["feature_importance"]
                )
                importance_df.to_csv(
                    output_path / "feature_importance.csv", index=False
                )
                logger.debug("Feature importance exported to CSV")

            # Export risk assessment if available
            if "risk_assessment" in self.results:
                risk_df = pd.DataFrame(self.results["risk_assessment"])
                risk_df.to_csv(output_path / "risk_assessment.csv", index=False)
                logger.debug("Risk assessment exported to CSV")

            # Export performance metrics if available
            if "academic_performance" in self.results:
                perf_df = pd.DataFrame(self.results["academic_performance"])
                perf_df.to_csv(output_path / "performance_metrics.csv", index=False)
                logger.debug("Performance metrics exported to CSV")

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise

    def _export_excel(self, output_path: Path) -> None:
        """
        Export all results to a single Excel file with multiple sheets.

        Args:
            output_path: Path to save the Excel file
        """
        try:
            excel_path = output_path / "complete_analysis.xlsx"
            with pd.ExcelWriter(excel_path) as writer:
                # Feature importance
                if (
                    "dropout_patterns" in self.results
                    and "feature_importance" in self.results["dropout_patterns"]
                ):
                    importance_df = pd.DataFrame(
                        self.results["dropout_patterns"]["feature_importance"]
                    )
                    importance_df.to_excel(
                        writer, sheet_name="Feature_Importance", index=False
                    )

                # Risk assessment
                if "risk_assessment" in self.results:
                    risk_df = pd.DataFrame(self.results["risk_assessment"])
                    risk_df.to_excel(writer, sheet_name="Risk_Assessment", index=False)

                # Performance metrics
                if "academic_performance" in self.results:
                    perf_df = pd.DataFrame(self.results["academic_performance"])
                    perf_df.to_excel(
                        writer, sheet_name="Performance_Metrics", index=False
                    )

                # Correlation analysis
                if "correlations" in self.results:
                    corr_df = pd.DataFrame(self.results["correlations"])
                    corr_df.to_excel(writer, sheet_name="Correlations", index=False)

            logger.debug(f"Results exported to Excel: {excel_path}")

        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise

    def export_specific_results(
        self, result_key: str, output_path: Path, format: str = "csv"
    ) -> None:
        """
        Export specific analysis results to a file.

        Args:
            result_key: Key of the result to export
            output_path: Path to save the file
            format: Output format (csv or excel)
        """
        try:
            if result_key not in self.results:
                raise KeyError(f"Result key '{result_key}' not found in results")

            data = pd.DataFrame(self.results[result_key])

            if format.lower() == "csv":
                data.to_csv(output_path / f"{result_key}.csv", index=False)
            elif format.lower() == "excel":
                data.to_excel(output_path / f"{result_key}.xlsx", index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.debug(f"Exported {result_key} to {format}")

        except Exception as e:
            logger.error(f"Error exporting specific results: {str(e)}")
            raise
