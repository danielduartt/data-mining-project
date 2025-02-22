from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import seaborn as sns
import time

from config.logging_config import setup_logging
from config.analysis_config import ANALYSIS_PARAMS, TARGET_NAMES, RISK_LEVELS

from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor

from analysis.feature_analyzer import FeatureAnalyzer
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from analysis.semester_analyzer import SemesterAnalyzer
from analysis.ensemble_analyzer import EnsembleAnalyzer
from analysis.neural_network_analyzer import NeuralNetworkAnalyzer
from analysis.neural_network_integration import run_neural_network_analysis

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
    start_time = time.time()

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

        # Ensemble Analysis
        logger.info("Starting ensemble analysis")
        ensemble_analyzer = EnsembleAnalyzer(
            n_jobs=-1, random_state=42, use_threading=True, cache_results=True
        )

        # Prepare features for ensemble analysis
        X, y = ensemble_analyzer.prepare_features(
            df, preprocessor.numeric_features, preprocessor.categorical_features
        )

        # Optimize hyperparameters
        logger.info("Optimizing model hyperparameters...")
        best_params = ensemble_analyzer.optimize_hyperparameters(X, y, n_trials=50)
        logger.info(f"Best hyperparameters found: {best_params}")

        # Perform ensemble analysis with optimized models
        ensemble_results = ensemble_analyzer.analyze_dropout_patterns(
            X,
            y,
            feature_names=preprocessor.numeric_features
            + preprocessor.categorical_features,
        )
        results["ensemble_analysis"] = ensemble_results

        # Generate risk scores using ensemble methods
        risk_scores = ensemble_analyzer.generate_risk_scores(X)
        results["ensemble_risk_scores"] = risk_scores

        # Neural Network Analysis
        logger.info("Starting neural network analysis")
        nn_results = run_neural_network_analysis(
            df=df,
            numeric_features=preprocessor.numeric_features,
            categorical_features=preprocessor.categorical_features,
            output_dir="analysis_results/neural_network",
            model_dir="models/neural_network",
        )
        results["neural_network_analysis"] = nn_results

        # Model Performance Comparison
        if "performance" in nn_results:
            # Add neural network performance to ensemble comparison
            if "model_comparison" in ensemble_results:
                ensemble_results["model_comparison"]["metrics_comparison"][
                    "Neural Network"
                ] = {
                    "accuracy": nn_results["performance"]["accuracy"],
                    "precision": nn_results["performance"]["precision"],
                    "recall": nn_results["performance"]["recall"],
                    "f1_score": nn_results["performance"]["f1_score"],
                }

        # Print detailed model comparison including neural network
        print("\nModel Performance Comparison (Including Neural Network)")
        print("=" * 80)

        if "model_comparison" in ensemble_results:
            metrics_comparison = ensemble_results["model_comparison"][
                "metrics_comparison"
            ]
            metrics = list(next(iter(metrics_comparison.values())).keys())

            # Print header
            header = f"{'Model':<20} " + " ".join(f"{metric:>12}" for metric in metrics)
            print(header)
            print("-" * len(header))

            # Print metrics for each model
            for model_name, model_metrics in metrics_comparison.items():
                metrics_str = " ".join(
                    f"{value:>12.3f}" for value in model_metrics.values()
                )
                print(f"{model_name:<20} {metrics_str}")

        # Generate Reports and Export Results
        logger.info("Generating reports and exporting results")

        # Create report
        report_generator = ReportGenerator(results, df)
        output_path = Path("analysis_results")
        report_generator.generate_summary_report(output_path)

        # Export results
        data_exporter = DataExporter(results)
        data_exporter.export_results()

        # Save trained ensemble models
        models_path = output_path / "models"
        models_path.mkdir(exist_ok=True)
        ensemble_analyzer.save_models(str(models_path))

        # Print summary statistics
        _print_summary_statistics(
            df,
            results,
            preprocessor,
            ensemble_results,
            nn_results,
            output_dir="analysis_results/plots",
        )

        # Calculate and print execution time
        execution_time = time.time() - start_time
        logger.info(
            f"Analysis completed successfully in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )
        print(
            f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )

        return results

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception("Detailed error trace:")
        return {}


def plot_model_comparison(metrics_comparison: Dict):
    """Create a bar plot comparing model performances with value labels."""
    # Convert metrics to DataFrame for plotting
    metrics_df = pd.DataFrame(metrics_comparison).T

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot grouped bar chart
    x = np.arange(len(metrics_df.index))
    width = 0.15
    multiplier = 0

    # To store references to bar collections for legend
    bar_collections = []

    for metric in metrics_df.columns:
        offset = width * multiplier
        bars = plt.bar(x + offset, metrics_df[metric], width, label=metric)
        bar_collections.append(bars)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

        multiplier += 1

    # Customize plot
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(x + width * 2, metrics_df.index, rotation=45, ha="right")

    # Adjust legend placement
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Metrics")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    return plt.gcf()


def plot_confusion_matrix(conf_matrix: list):
    """Create a heatmap visualization of confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def plot_feature_importance(feature_importance: list, top_n: int = 10):
    """Create a horizontal bar plot of feature importance."""
    # Convert to DataFrame and get top N features
    importance_df = pd.DataFrame(feature_importance).nlargest(top_n, "importance")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y="feature", x="importance", palette="viridis")
    plt.title(f"Top {top_n} Most Important Features")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def _print_summary_statistics(
    df: pd.DataFrame,
    results: Dict,
    preprocessor: DataPreprocessor,
    ensemble_results: Dict,
    output_dir: str = "analysis_results/plots",
) -> None:
    """Print summary statistics and create visualizations."""
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Basic error checking
    if not ensemble_results or "best_model" not in ensemble_results:
        print("No ensemble results available for visualization.")
        return

    # Print basic statistics
    print("\nAnalysis Summary:")
    print("-" * 50)
    print(f"Total students analyzed: {len(df)}")
    print(
        f"Features processed: "
        f"{len(preprocessor.numeric_features) + len(preprocessor.categorical_features)}"
    )
    print(f"Dropout rate: {(df['Target'] == 0).mean():.2%}")

    # Confusion Matrix Plot
    try:
        # Create and save confusion matrix plot
        if "best_model" in ensemble_results:
            best_model = ensemble_results["best_model"]

            # Ensure confusion matrix is present
            if "metrics" in best_model and "confusion_matrix" in best_model["metrics"]:
                conf_matrix = best_model["metrics"]["confusion_matrix"]

                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"],
                )
                plt.title("Confusion Matrix Heatmap")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")

                # Save the plot
                conf_matrix_path = output_path / "confusion_matrix.png"
                plt.tight_layout()
                plt.savefig(conf_matrix_path)
                plt.close()

                print(f"\nConfusion matrix plot saved to: {conf_matrix_path}")
            else:
                print("Confusion matrix data not available.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")

    # Model Comparison Plot
    try:
        if "model_comparison" in ensemble_results:
            metrics_comparison = ensemble_results["model_comparison"][
                "metrics_comparison"
            ]

            # Create model comparison plot
            plt.figure(figsize=(12, 6))
            metrics_df = pd.DataFrame(metrics_comparison).T

            x = np.arange(len(metrics_df.index))
            width = 0.15
            multiplier = 0

            for metric in metrics_df.columns:
                offset = width * multiplier
                bars = plt.bar(x + offset, metrics_df[metric], width, label=metric)

                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=45,
                    )

                multiplier += 1

            plt.xlabel("Models")
            plt.ylabel("Score")
            plt.title("Model Performance Comparison")
            plt.xticks(x + width * 2, metrics_df.index, rotation=45, ha="right")
            plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Metrics")
            plt.tight_layout()

            # Save the plot
            model_comp_path = output_path / "model_comparison.png"
            plt.savefig(model_comp_path, bbox_inches="tight")
            plt.close()

            print(f"Model comparison plot saved to: {model_comp_path}")
    except Exception as e:
        print(f"Error creating model comparison plot: {e}")

    print(f"\nVisualization plots have been saved to: {output_dir}")


if __name__ == "__main__":
    results = main()
