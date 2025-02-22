import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import joblib
from pathlib import Path

logger = logging.getLogger("StudentAnalyzer")


class NeuralNetworkAnalyzer:
    """Analyzes student data using neural network models for dropout prediction."""

    def __init__(
        self,
        model_dir: str = "models/neural_network",
        random_state: int = 42,
        verbose: int = 1,
        use_gpu: bool = True,
        cache_results: bool = True,
    ):
        """
        Initialize the neural network analyzer.

        Args:
            model_dir: Directory for storing model files
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch)
            use_gpu: Whether to use GPU for training if available
            cache_results: Whether to cache preprocessing results
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.verbose = verbose
        self.cache_results = cache_results
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

        # Configure GPU settings
        if use_gpu:
            self._configure_gpu()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        logger.info("Neural Network Analyzer initialized")

    def _configure_gpu(self) -> None:
        """Configure GPU memory growth to prevent OOM errors."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configured successfully. Found {len(gpus)} GPU(s).")
            else:
                logger.info("No GPU found. Using CPU for training.")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {str(e)}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
        target_col: str = "Target",
        test_size: float = 0.2,
        validation_size: float = 0.1,
        binary_classification: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Prepare data for neural network training.

        Args:
            df: Input DataFrame
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            target_col: Target column name
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            binary_classification: Whether to use binary classification (dropout vs non-dropout)

        Returns:
            Tuple containing:
                - Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
                - List of feature names
        """
        logger.info("Preparing data for neural network training")

        try:
            # Handle binary or multiclass classification
            if binary_classification:
                # Convert to binary classification: dropout (1) vs non-dropout (0)
                y = (df[target_col] == 0).astype(int)
                num_classes = 2
            else:
                # Use original multiclass targets
                y = df[target_col].values
                num_classes = len(df[target_col].unique())

            # Process numeric features
            X_numeric = df[numeric_features].copy()

            # Process categorical features using one-hot encoding
            X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)

            # Combine all features
            X = pd.concat([X_numeric, X_categorical], axis=1)
            feature_names = X.columns.tolist()

            # Scale numeric features
            X[numeric_features] = self.scaler.fit_transform(X[numeric_features])

            # Save scaler for later use
            if self.cache_results:
                joblib.dump(self.scaler, self.model_dir / "scaler.joblib")

            # First split: training and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            # Second split: training and validation sets
            val_ratio = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=y_train_val,
            )

            # Convert targets to categorical for multi-class classification
            if num_classes > 2 and not binary_classification:
                y_train = to_categorical(y_train, num_classes)
                y_val = to_categorical(y_val, num_classes)
                y_test = to_categorical(y_test, num_classes)

            logger.info(
                f"Data prepared successfully. Training set: {X_train.shape}, "
                f"Validation set: {X_val.shape}, Test set: {X_test.shape}"
            )

            # Return data in a dictionary
            return {
                "X_train": X_train.values,
                "X_val": X_val.values,
                "X_test": X_test.values,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "num_classes": num_classes,
                "feature_names": feature_names,
            }, feature_names

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def build_model(
        self,
        input_shape: Tuple[int],
        num_classes: int = 2,
        learning_rate: float = 0.001,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ) -> Sequential:
        """
        Build a neural network model for classification.

        Args:
            input_shape: Shape of input features
            num_classes: Number of target classes
            learning_rate: Learning rate for optimizer
            hidden_layers: List of neuron counts for each hidden layer
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer (None for auto)

        Returns:
            Compiled Keras Sequential model
        """
        logger.info(f"Building neural network with {len(hidden_layers)} hidden layers")

        # Determine appropriate output activation function
        if output_activation is None:
            output_activation = "sigmoid" if num_classes == 2 else "softmax"

        # Determine loss function
        loss = "binary_crossentropy" if num_classes == 2 else "categorical_crossentropy"

        # Build model
        model = Sequential()

        # Input layer
        model.add(
            Dense(hidden_layers[0], input_shape=input_shape, activation=activation)
        )
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(
            Dense(1 if num_classes == 2 else num_classes, activation=output_activation)
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"]
        )

        # Print model summary
        model.summary()

        self.model = model
        return model

    def train_model(
        self,
        data: Dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        min_delta: float = 0.001,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the neural network model.

        Args:
            data: Dictionary containing training, validation and test data
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            min_delta: Minimum change to qualify as improvement
            model_path: Path to save best model (None to use default)

        Returns:
            Dictionary with training results and metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info(f"Training neural network for up to {epochs} epochs")

        # Set model path
        if model_path is None:
            model_path = str(self.model_dir / "best_model.h5")

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # Train model
        self.history = self.model.fit(
            data["X_train"],
            data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        # Load best model
        self.model = load_model(model_path)

        # Evaluate on test set
        test_results = self.evaluate_model(data)

        # Return combined results
        return {
            "history": {
                "accuracy": self.history.history["accuracy"],
                "val_accuracy": self.history.history["val_accuracy"],
                "loss": self.history.history["loss"],
                "val_loss": self.history.history["val_loss"],
            },
            "test_results": test_results,
            "model_path": model_path,
            "epochs_trained": len(self.history.history["loss"]),
            "best_epoch": np.argmin(self.history.history["val_loss"]) + 1,
        }

    def evaluate_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            data: Dictionary containing test data

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        logger.info("Evaluating neural network model")

        # Get predictions
        y_pred_prob = self.model.predict(data["X_test"])

        # Convert probabilities to class predictions
        if y_pred_prob.shape[1] > 1:  # Multi-class
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(data["y_test"], axis=1)
        else:  # Binary
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_true = data["y_test"]

        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred)

        if data["num_classes"] == 2:  # Binary classification metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:  # Multi-class metrics (macro average)
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")

        # Log results
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"F1 score: {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
            "y_true": y_true,
        }

    def generate_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Generate dropout risk scores for students.

        Args:
            X: Feature matrix

        Returns:
            Array of risk scores (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get prediction probabilities
        y_pred_prob = self.model.predict(X)

        # For binary classification, return dropout probability
        if y_pred_prob.shape[1] == 1:
            return y_pred_prob.flatten()
        else:
            # For multi-class, return dropout class probability
            return y_pred_prob[:, 0]

    def perform_feature_importance_analysis(
        self,
        data: Dict[str, np.ndarray],
        feature_names: List[str],
        n_iterations: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Perform permutation feature importance analysis.

        Args:
            data: Dictionary containing test data
            feature_names: List of feature names
            n_iterations: Number of permutation iterations

        Returns:
            List of dictionaries with feature importance info
        """
        logger.info("Performing neural network feature importance analysis")

        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        X_test = data["X_test"]
        y_test = data["y_test"]

        # Get baseline performance
        baseline_pred = self.model.predict(X_test)
        if len(baseline_pred.shape) > 1 and baseline_pred.shape[1] > 1:
            baseline_pred = np.argmax(baseline_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
        else:
            baseline_pred = (baseline_pred > 0.5).astype(int).flatten()
            y_true = y_test

        baseline_score = accuracy_score(y_true, baseline_pred)

        # Calculate feature importance
        importance_result = []
        for i, feature_name in enumerate(feature_names):
            importance_scores = []

            for _ in range(n_iterations):
                # Create a copy of the test data
                X_permuted = X_test.copy()

                # Permute the feature
                np.random.shuffle(X_permuted[:, i])

                # Predict with permuted feature
                perm_pred = self.model.predict(X_permuted)
                if len(perm_pred.shape) > 1 and perm_pred.shape[1] > 1:
                    perm_pred = np.argmax(perm_pred, axis=1)
                else:
                    perm_pred = (perm_pred > 0.5).astype(int).flatten()

                # Calculate permuted score
                perm_score = accuracy_score(y_true, perm_pred)

                # Calculate importance as decrease in performance
                importance = baseline_score - perm_score
                importance_scores.append(importance)

            # Calculate average importance
            mean_importance = np.mean(importance_scores)
            std_importance = np.std(importance_scores)

            importance_result.append(
                {
                    "feature": feature_name,
                    "importance": mean_importance,
                    "std": std_importance,
                }
            )

        # Sort by importance
        importance_result = sorted(
            importance_result, key=lambda x: x["importance"], reverse=True
        )

        return importance_result

    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.

        Args:
            save_path: Path to save the plot (None to not save)

        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history.history["loss"], label="Training Loss")
        ax1.plot(self.history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax2.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: List[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            conf_matrix: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot (None to not save)

        Returns:
            Matplotlib figure
        """
        if class_names is None:
            if conf_matrix.shape[0] == 2:
                class_names = ["Non-Dropout", "Dropout"]
            else:
                class_names = [f"Class {i}" for i in range(conf_matrix.shape[0])]

        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_feature_importance(
        self,
        importance_data: List[Dict[str, Any]],
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            importance_data: Feature importance data
            top_n: Number of top features to show
            save_path: Path to save the plot (None to not save)

        Returns:
            Matplotlib figure
        """
        # Get top features
        top_features = importance_data[:top_n]

        # Create DataFrame
        df = pd.DataFrame(top_features)

        # Create figure
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="importance", y="feature", data=df, palette="viridis")

        # Add error bars
        for i, feature in enumerate(df.itertuples()):
            ax.errorbar(
                x=feature.importance,
                y=i,
                xerr=feature.std,
                fmt="none",
                color="black",
                capsize=3,
            )

        plt.title(f"Top {top_n} Most Important Features")
        plt.xlabel("Importance Score (Accuracy Decrease)")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            path: Path to save the model (None to use default)

        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        if path is None:
            path = str(self.model_dir / "final_model.h5")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

        return path

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
