import pandas as pd
from pathlib import Path
from typing import Dict

from config.logging_config import setup_logging
from config.analysis_config import ANALYSIS_PARAMS, TARGET_NAMES, RISK_LEVELS

from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor

from analysis.feature_analyzer import FeatureAnalyzer
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from analysis.semester_analyzer import SemesterAnalyzer

from visualization.plot_config import setup_plot_style

from utils.validation_utils import DataValidator

from export.report_generator import ReportGenerator
from export.data_exporter import DataExporter


def main(file_path: str = "data/cleaned_dataset.csv") -> Dict:
    """
    Main execution function with comprehensive analysis and logging.

    Args:
        file_path: Path to the input dataset

    Returns:
        Dict containing all analysis results
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting main execution")

    try:
        # Setup visualization
        setup_plot_style()

        # Load and preprocess data
        logger.info(f"Loading data from: {file_path}")
        df = DataLoader.load_data(file_path)

        # Validate data
        logger.info("Validating data")
        DataValidator.validate_required_columns(df, ["Target"])
        DataValidator.validate_target_values(
            df, {0, 1, 2}
        )  # Dropout, Graduate, Enrolled

        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        df = preprocessor.setup_data_types(df)

        # Initialize results dictionary
        results = {}

        # Feature Analysis
        logger.info("Starting feature analysis")
        feature_analyzer = FeatureAnalyzer(
            df=df,
            numeric_features=preprocessor.numeric_features,
            categorical_features=preprocessor.categorical_features,
            target_names=TARGET_NAMES,
        )
        results["feature_distributions"] = (
            feature_analyzer.analyze_feature_distributions()
        )

        # Performance Analysis
        logger.info("Starting performance analysis")
        performance_analyzer = PerformanceAnalyzer(df)
        results["academic_performance"] = (
            performance_analyzer.analyze_academic_performance()
        )

        # Correlation Analysis
        logger.info("Starting correlation analysis")
        correlation_analyzer = CorrelationAnalyzer(
            df=df,
            numeric_features=preprocessor.numeric_features,
            categorical_features=preprocessor.categorical_features,
        )
        results["correlations"] = correlation_analyzer.analyze_correlations()

        # Risk Analysis
        logger.info("Starting risk analysis")
        risk_analyzer = RiskAnalyzer(
            df=df,
            numeric_features=preprocessor.numeric_features,
            analysis_params=ANALYSIS_PARAMS,
        )
        results["dropout_patterns"] = risk_analyzer.analyze_dropout_patterns()
        results["risk_assessment"] = risk_analyzer.generate_risk_assessment()

        # Semester Analysis
        logger.info("Starting semester analysis")
        semester_analyzer = SemesterAnalyzer(df, TARGET_NAMES)
        results["semester_patterns"] = semester_analyzer.analyze_semester_patterns()

        # Generate Reports and Export Results
        logger.info("Generating reports and exporting results")

        # Create report
        report_generator = ReportGenerator(results, df)
        output_path = Path("analysis_results")
        report_generator.generate_summary_report(output_path)

        # Export results
        data_exporter = DataExporter(results)
        data_exporter.export_results()

        # Print summary statistics
        _print_summary_statistics(df, results, preprocessor)

        logger.info("Analysis completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception("Detailed error trace:")
        return {}


def _print_summary_statistics(
    df: pd.DataFrame, results: Dict, preprocessor: DataPreprocessor
) -> None:
    """Print summary statistics to console."""
    print("\nAnalysis Summary:")
    print("-" * 50)
    print(f"Total students analyzed: {len(df)}")
    print(
        f"Features processed: "
        f"{len(preprocessor.numeric_features) + len(preprocessor.categorical_features)}"
    )
    print(f"Dropout rate: {(df['Target'] == 0).mean():.2%}")

    if (
        "risk_assessment" in results
        and "risk_distribution" in results["risk_assessment"]
    ):
        print("\nRisk Distribution:")
        for level, pct in results["risk_assessment"]["risk_distribution"].items():
            print(f"{level} risk: {pct:.1f}%")


if __name__ == "__main__":
    results = main()
