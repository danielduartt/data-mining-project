import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from concurrent.futures import ThreadPoolExecutor
import optuna

from analysis.optimize_ensemble import EnhancedEnsembleOptimizer

logger = logging.getLogger("StudentAnalyzer")


class EnsembleAnalyzer:
    def __init__(
        self,
        n_jobs: int = -1,
        random_state: int = 42,
        use_threading: bool = True,
        cache_results: bool = True,
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_threading = use_threading
        self.cache_results = cache_results
        self.scaler = StandardScaler()

        # Initialize base models
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                max_features=0.8,
                random_state=self.random_state,
            ),
            "adaboost": AdaBoostClassifier(
                n_estimators=200, learning_rate=0.05, random_state=self.random_state
            ),
        }

        self._cache = {} if cache_results else None

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, n_trials: int = 100
    ):
        """Optimize hyperparameters for each model."""
        optimizer = EnhancedEnsembleOptimizer(
            n_jobs=self.n_jobs, random_state=self.random_state, n_trials=n_trials
        )

        # Get optimized parameters
        best_params = optimizer.optimize_hyperparameters(X, y)

        # Update models with best parameters
        self.models["random_forest"].set_params(**best_params["random_forest"])
        self.models["gradient_boosting"].set_params(**best_params["gradient_boosting"])
        self.models["adaboost"].set_params(**best_params["adaboost"])

        return best_params

    def prepare_features(
        self,
        df: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced feature preparation with advanced engineering."""
        logger.info("Preparing features with advanced engineering")
        try:
            # Target variable
            y = (df["Target"] == 0).astype(int)

            # Create a copy for feature engineering
            df_engineered = df.copy()

            # Initialize a list to track all features
            all_feature_arrays = []

            # 1. Original numeric features
            if numeric_features:
                numeric_data = df_engineered[numeric_features].values
                all_feature_arrays.append(numeric_data)
                logger.info(f"Original numeric features: {len(numeric_features)}")

            # 2. Semester performance features
            semester_features = []
            for sem in range(1, 3):  # For first two semesters
                enrolled_col = f"Curricular units {sem}st sem (enrolled)"
                approved_col = f"Curricular units {sem}st sem (approved)"
                grade_col = f"Curricular units {sem}st sem (grade)"

                if all(
                    col in df.columns for col in [enrolled_col, approved_col, grade_col]
                ):
                    # Success rate
                    df_engineered[f"success_rate_sem_{sem}"] = df[approved_col] / df[
                        enrolled_col
                    ].replace(0, 1)
                    semester_features.append(f"success_rate_sem_{sem}")

                    # Average grade per enrolled unit
                    df_engineered[f"grade_per_unit_sem_{sem}"] = df[grade_col] / df[
                        enrolled_col
                    ].replace(0, 1)
                    semester_features.append(f"grade_per_unit_sem_{sem}")

                    # Performance efficiency
                    df_engineered[f"performance_efficiency_sem_{sem}"] = (
                        df[approved_col]
                        * df[grade_col]
                        / (df[enrolled_col].replace(0, 1) * 20)
                    )
                    semester_features.append(f"performance_efficiency_sem_{sem}")

            # Add semester features
            if semester_features:
                semester_data = df_engineered[semester_features].values
                all_feature_arrays.append(semester_data)
                logger.info(f"Semester features added: {len(semester_features)}")

            # 3. Socioeconomic indicators
            socio_features = []
            if (
                "Scholarship holder" in df.columns
                and "Tuition fees up to date" in df.columns
            ):
                df_engineered["financial_risk"] = (1 - df["Scholarship holder"]) * (
                    1 - df["Tuition fees up to date"]
                )
                socio_features.append("financial_risk")

            # Add socioeconomic features
            if socio_features:
                socio_data = df_engineered[socio_features].values
                all_feature_arrays.append(socio_data)
                logger.info(f"Socioeconomic features added: {len(socio_features)}")

            # 4. Age-related features
            age_features = []
            if "Age at enrollment" in df.columns:
                df_engineered["age_group"] = pd.qcut(
                    df["Age at enrollment"], q=5, labels=False  # Use numeric labels
                )
                age_features.append("age_group")

            # Add age features
            if age_features:
                age_data = df_engineered[age_features].values
                all_feature_arrays.append(age_data)
                logger.info(f"Age features added: {len(age_features)}")

            # 5. Categorical features (one-hot encoding)
            cat_feature_list = categorical_features + (
                ["age_group"] if "age_group" in df_engineered.columns else []
            )

            categorical_dummies = []
            for cat_feature in cat_feature_list:
                if cat_feature in df_engineered.columns:
                    dummies = pd.get_dummies(
                        df_engineered[cat_feature], prefix=cat_feature, dummy_na=False
                    ).values
                    categorical_dummies.append(dummies)

            # Add categorical features
            if categorical_dummies:
                all_feature_arrays.extend(categorical_dummies)
                logger.info(f"Categorical features added: {len(categorical_dummies)}")

            # Combine all features
            if all_feature_arrays:
                X = np.hstack(all_feature_arrays)

                # Scale features
                X = self.scaler.fit_transform(X)

                logger.info(f"Total features prepared: {X.shape[1]}")
                return X, y
            else:
                raise ValueError("No features available for analysis")

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def _evaluate_model_performance(
        self, model: object, X: np.ndarray, y: np.ndarray, cv_splits: int = 5
    ) -> Dict:
        """Evaluate model performance with multiple metrics"""
        try:
            # Define scoring metrics
            scoring = {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            }

            # Perform cross-validation
            cv = TimeSeriesSplit(n_splits=cv_splits)
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                return_train_score=True,
            )

            # Fit model on full dataset
            model.fit(X, y)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            metrics = {
                "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
                "classification_report": classification_report(
                    y, y_pred, output_dict=True
                ),
                "full_dataset": {
                    "accuracy": (y_pred == y).mean(),
                    "precision": precision_score(y, y_pred),
                    "recall": recall_score(y, y_pred),
                    "f1": f1_score(y, y_pred),
                    "roc_auc": roc_auc_score(y, y_prob),
                },
                "cross_validation": {
                    "accuracy": {
                        "mean": cv_results["test_accuracy"].mean(),
                        "std": cv_results["test_accuracy"].std(),
                    },
                    "precision": {
                        "mean": cv_results["test_precision"].mean(),
                        "std": cv_results["test_precision"].std(),
                    },
                    "recall": {
                        "mean": cv_results["test_recall"].mean(),
                        "std": cv_results["test_recall"].std(),
                    },
                    "f1": {
                        "mean": cv_results["test_f1"].mean(),
                        "std": cv_results["test_f1"].std(),
                    },
                    "roc_auc": {
                        "mean": cv_results["test_roc_auc"].mean(),
                        "std": cv_results["test_roc_auc"].std(),
                    },
                },
            }

            if hasattr(model, "feature_importances_"):
                metrics["feature_importance"] = model.feature_importances_

            return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return {}

    def analyze_dropout_patterns(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Analyze dropout patterns with feature selection and model evaluation."""
        logger.info("Starting ensemble analysis of dropout patterns")

        try:
            # Check cache
            cache_key = hash(str(X) + str(y))
            if self.cache_results and cache_key in self._cache:
                logger.info("Using cached results")
                return self._cache[cache_key]

            # Feature selection using Random Forest
            rf_selector = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )
            rf_selector.fit(X, y)

            # Select top 80% most important features
            importances = rf_selector.feature_importances_
            threshold = np.percentile(importances, 20)  # Keep top 80%
            selected_features = importances >= threshold

            # Use selected features
            X_selected = X[:, selected_features]

            # Update feature names if provided
            selected_feature_names = None
            if feature_names:
                selected_feature_names = [
                    f for f, s in zip(feature_names, selected_features) if s
                ]

            # Train and evaluate models
            models_results = {}
            if self.use_threading:
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            self._train_and_evaluate_model, name, model, X_selected, y
                        )
                        for name, model in self.models.items()
                    ]

                    for future in futures:
                        name, metrics = future.result()
                        if metrics:
                            models_results[name] = metrics

            # Find best model based on F1 score
            best_model_name = max(
                models_results.keys(),
                key=lambda k: models_results[k]["cross_validation"]["f1"]["mean"],
            )

            # Prepare results
            results = {
                "models": models_results,
                "best_model": {
                    "name": best_model_name,
                    "metrics": models_results[best_model_name],
                },
                "model_comparison": self._create_model_comparison(models_results),
            }

            # Add feature importance if feature names were provided
            if (
                selected_feature_names
                and "feature_importance" in models_results[best_model_name]
            ):
                importance_df = pd.DataFrame(
                    {
                        "feature": selected_feature_names,
                        "importance": models_results[best_model_name][
                            "feature_importance"
                        ],
                    }
                ).sort_values("importance", ascending=False)

                results["feature_importance"] = importance_df.to_dict("records")

            # Cache results
            if self.cache_results:
                self._cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Error in dropout pattern analysis: {str(e)}")
            return {}

    def _train_and_evaluate_model(
        self, name: str, model: object, X: np.ndarray, y: np.ndarray
    ) -> Tuple[str, Dict]:
        """Train and evaluate a single model"""
        try:
            metrics = self._evaluate_model_performance(model, X, y)
            return name, metrics
        except Exception as e:
            logger.error(f"Error training model {name}: {str(e)}")
            return name, None

    def _create_model_comparison(self, models_results: Dict) -> Dict:
        """Create a comparative summary of all models"""
        # Print detailed model metrics
        print("\nEnsemble Models Performance Comparison")
        print("=" * 50)

        # Prepare comparison dictionary
        comparison = {"metrics_comparison": {}}

        # Print header
        print(
            f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'ROC AUC':>10}"
        )
        print("-" * 80)

        # Store results for return and detailed printing
        for model_name, res in models_results.items():
            cv_metrics = res["cross_validation"]

            # Prepare metrics for comparison dictionary
            model_metrics = {
                "Accuracy": cv_metrics["accuracy"]["mean"],
                "Precision": cv_metrics["precision"]["mean"],
                "Recall": cv_metrics["recall"]["mean"],
                "F1 Score": cv_metrics["f1"]["mean"],
                "ROC AUC": cv_metrics["roc_auc"]["mean"],
            }

            # Add to comparison dictionary
            comparison["metrics_comparison"][model_name] = model_metrics

            # Print metrics
            print(
                f"{model_name:<20} "
                f"{model_metrics['Accuracy']:10.4f} "
                f"{model_metrics['Precision']:10.4f} "
                f"{model_metrics['Recall']:10.4f} "
                f"{model_metrics['F1 Score']:10.4f} "
                f"{model_metrics['ROC AUC']:10.4f}"
            )

        return comparison

    def generate_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """Generate risk scores using the Random Forest model"""
        try:
            model = self.models["random_forest"]
            probabilities = model.predict_proba(X)
            return probabilities[:, 1]  # Probability of dropout
        except Exception as e:
            logger.error(f"Error generating risk scores: {str(e)}")
            return np.array([])

    def save_models(self, path: str) -> None:
        """Save trained models to disk"""
        try:
            for name, model in self.models.items():
                joblib.dump(model, f"{path}/{name}_model.joblib")
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def load_models(self, path: str) -> None:
        """Load trained models from disk"""
        try:
            for name in self.models.keys():
                self.models[name] = joblib.load(f"{path}/{name}_model.joblib")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
