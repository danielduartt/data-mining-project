from analysis.neural_network_analyzer import NeuralNetworkAnalyzer
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any

logger = logging.getLogger("StudentAnalyzer")


def run_neural_network_analysis(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    output_dir: str = "analysis_results/neural_network",
    model_dir: str = "models/neural_network",
) -> Dict[str, Any]:
    """
    Run neural network analysis on student data.

    Args:
        df: Processed DataFrame with student data
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        output_dir: Directory for output visualizations
        model_dir: Directory for saving models

    Returns:
        Dictionary with neural network analysis results
    """
    logger.info("Starting neural network analysis")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize neural network analyzer
    nn_analyzer = NeuralNetworkAnalyzer(
        model_dir=model_dir,
        random_state=42,
        verbose=1,
        use_gpu=True,
        cache_results=True,
    )

    try:
        # Prepare data for neural network
        data, feature_names = nn_analyzer.prepare_data(
            df=df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col="Target",
            test_size=0.2,
            validation_size=0.1,
            binary_classification=True,  # Dropout (1) vs Non-dropout (0)
        )

        # Build neural network model
        nn_analyzer.build_model(
            input_shape=(data["X_train"].shape[1],),
            num_classes=2,  # Binary classification
            learning_rate=0.001,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            use_batch_norm=True,
        )

        # Train model
        training_results = nn_analyzer.train_model(
            data=data,
            epochs=100,
            batch_size=32,
            patience=10,
            model_path=str(Path(model_dir) / "best_nn_model.h5"),
        )

        # Plot and save training history
        history_fig = nn_analyzer.plot_training_history(
            save_path=str(output_path / "training_history.png")
        )
        plt.close(history_fig)

        # Plot and save confusion matrix
        conf_matrix = training_results["test_results"]["confusion_matrix"]
        cm_fig = nn_analyzer.plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=["Non-Dropout", "Dropout"],
            save_path=str(output_path / "confusion_matrix.png"),
        )
        plt.close(cm_fig)

        # Feature importance analysis
        feature_importance = nn_analyzer.perform_feature_importance_analysis(
            data=data,
            feature_names=feature_names,
            n_iterations=10,  # Lower for faster execution, increase for production
        )

        # Plot and save feature importance
        fi_fig = nn_analyzer.plot_feature_importance(
            importance_data=feature_importance,
            top_n=15,
            save_path=str(output_path / "feature_importance.png"),
        )
        plt.close(fi_fig)

        # Generate risk scores for all students
        risk_scores = nn_analyzer.generate_risk_scores(
            X=np.vstack([data["X_train"], data["X_val"], data["X_test"]])
        )

        # Save final model
        final_model_path = nn_analyzer.save_model()

        # Prepare results dictionary
        nn_results = {
            "model_info": {
                "type": "neural_network",
                "architecture": "MLP",
                "hidden_layers": [128, 64, 32],
                "model_path": final_model_path,
            },
            "training": {
                "epochs_trained": training_results["epochs_trained"],
                "best_epoch": training_results["best_epoch"],
                "history": training_results["history"],
            },
            "performance": {
                "accuracy": float(training_results["test_results"]["accuracy"]),
                "precision": float(training_results["test_results"]["precision"]),
                "recall": float(training_results["test_results"]["recall"]),
                "f1_score": float(training_results["test_results"]["f1_score"]),
                "confusion_matrix": conf_matrix.tolist(),
            },
            "feature_importance": feature_importance,
            "plots": {
                "training_history": str(output_path / "training_history.png"),
                "confusion_matrix": str(output_path / "confusion_matrix.png"),
                "feature_importance": str(output_path / "feature_importance.png"),
            },
        }

        logger.info("Neural network analysis completed successfully")
        return nn_results

    except Exception as e:
        logger.error(f"Error in neural network analysis: {str(e)}")
        logger.exception("Detailed error trace:")
        return {"error": str(e)}
