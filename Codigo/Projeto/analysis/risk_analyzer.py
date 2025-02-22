import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, List

logger = logging.getLogger("StudentAnalyzer")


class RiskAnalyzer:
    """Analyzes dropout risk patterns and generates risk assessments."""

    def __init__(
        self, df: pd.DataFrame, numeric_features: List[str], analysis_params: Dict
    ):
        """
        Initialize the risk analyzer.

        Args:
            df: Input DataFrame
            numeric_features: List of numeric feature names
            analysis_params: Dictionary of analysis parameters
        """
        self.df = df
        self.numeric_features = numeric_features
        self.analysis_params = analysis_params
        self.demographic_cols = ["Gender", "Age at enrollment", "Scholarship holder"]

    def analyze_dropout_patterns(self) -> Dict:
        """
        Analyze patterns and factors associated with student dropout.

        Returns:
            Dict: Dropout patterns and risk factors
        """
        logger.info("Starting dropout pattern analysis")
        dropout_patterns = {}

        try:
            # Create binary dropout indicator
            self.df["is_dropout"] = (self.df["Target"] == 0).astype(int)

            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns()
            if temporal_patterns:
                dropout_patterns["temporal_patterns"] = temporal_patterns

            # Analyze feature importance
            feature_importance = self._analyze_feature_importance()
            if feature_importance:
                dropout_patterns["feature_importance"] = feature_importance

            # Analyze demographic factors
            demographic_patterns = self._analyze_demographic_factors()
            if demographic_patterns:
                dropout_patterns["demographic_factors"] = demographic_patterns

            logger.info("Dropout pattern analysis completed")
            return dropout_patterns

        except Exception as e:
            logger.error(f"Error in dropout pattern analysis: {str(e)}")
            return {}

    def generate_risk_assessment(self) -> Dict:
        """
        Generate comprehensive risk assessment for students.

        Returns:
            Dict: Risk assessment results and indicators
        """
        logger.info("Starting risk assessment generation")
        risk_assessment = {}

        try:
            # Create risk factors DataFrame
            risk_factors = self._calculate_risk_factors()

            if not risk_factors.empty:
                # Calculate and categorize risk scores
                risk_scores = self._calculate_risk_scores(risk_factors)

                # Get risk statistics
                risk_assessment.update(self._get_risk_statistics(risk_scores))

                # Visualize risk distribution
                self._plot_risk_distribution(risk_scores)

            logger.info("Risk assessment generation completed")
            return risk_assessment

        except Exception as e:
            logger.error(f"Error in risk assessment generation: {str(e)}")
            return {}

    def _analyze_temporal_patterns(self) -> Dict:
        """Analyze dropout patterns across semesters."""
        dropout_rates = {}
        semester_cols = [col for col in self.df.columns if "sem" in col.lower()]

        if semester_cols:
            logger.debug("Analyzing temporal dropout patterns")

            for sem in range(1, len(semester_cols) // 2 + 1):
                enrolled_col = f"Curricular units {sem}st sem (enrolled)"
                if enrolled_col in self.df.columns:
                    dropout_rates[f"Semester_{sem}"] = {
                        "rate": float((self.df[enrolled_col] == 0).mean()),
                        "count": int((self.df[enrolled_col] == 0).sum()),
                    }

        return dropout_rates

    def _analyze_feature_importance(self) -> List[Dict]:
        """Analyze feature importance for dropout prediction."""
        logger.debug("Analyzing feature importance for dropout prediction")

        # Prepare data for model
        X = self.df[self.numeric_features]
        y = self.df["is_dropout"]

        # Train Random Forest for feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Calculate feature importance scores
        importance_df = pd.DataFrame(
            {
                "feature": self.numeric_features,
                "importance": rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Visualize feature importance
        self._plot_feature_importance(importance_df)

        return importance_df.to_dict("records")

    def _analyze_demographic_factors(self) -> Dict:
        """Analyze dropout patterns by demographic factors."""
        logger.debug("Analyzing demographic factors in dropout")
        demographic_patterns = {}

        for col in self.demographic_cols:
            if col in self.df.columns:
                group_stats = self.df.groupby(col)["is_dropout"].agg(["mean", "count"])
                demographic_patterns[col] = group_stats.to_dict("index")

                # Visualize demographic patterns
                self._plot_demographic_dropout_rate(col)

        return demographic_patterns

    def _calculate_risk_factors(self) -> pd.DataFrame:
        """Calculate risk factors for each student."""
        risk_factors = pd.DataFrame()

        # Academic performance risk factors
        if "first_semester_success_rate" in self.df.columns:
            risk_factors["low_first_sem_performance"] = (
                self.df["first_semester_success_rate"]
                < self.analysis_params["risk_threshold_high"]
            ).astype(int)

        if "Admission grade" in self.df.columns:
            risk_factors["low_admission_grade"] = (
                self.df["Admission grade"] < self.df["Admission grade"].quantile(0.25)
            ).astype(int)

        return risk_factors

    def _calculate_risk_scores(self, risk_factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall risk scores and levels."""
        risk_factors["risk_score"] = risk_factors.sum(axis=1)
        self.df["risk_score"] = risk_factors["risk_score"]

        # Define risk levels
        self.df["risk_level"] = pd.qcut(
            self.df["risk_score"], q=3, labels=["Low", "Medium", "High"]
        )

        return risk_factors

    def _get_risk_statistics(self, risk_factors: pd.DataFrame) -> Dict:
        """Calculate risk statistics and validation metrics."""
        stats = {}

        # Calculate risk distribution
        stats["risk_distribution"] = (
            self.df["risk_level"].value_counts(normalize=True) * 100
        ).to_dict()

        # Analyze risk level effectiveness
        risk_validation = pd.crosstab(self.df["risk_level"], self.df["Target"])
        stats["risk_validation"] = risk_validation.to_dict()

        return stats

    def _plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """Plot feature importance visualization."""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(10), x="importance", y="feature")
        plt.title("Top 10 Features for Dropout Prediction")
        plt.tight_layout()
        plt.show()

    def _plot_demographic_dropout_rate(self, col: str) -> None:
        """Plot dropout rate by demographic factor."""
        plt.figure(figsize=(10, 5))
        sns.barplot(data=self.df, x=col, y="is_dropout")
        plt.title(f"Dropout Rate by {col}")
        plt.ylabel("Dropout Rate")
        plt.tight_layout()
        plt.show()

    def _plot_risk_distribution(self, risk_factors: pd.DataFrame) -> None:
        """Plot risk score distribution and validation."""
        plt.figure(figsize=(12, 5))

        # Risk score distribution
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.df, x="risk_score", bins=20)
        plt.title("Distribution of Risk Scores")

        # Risk levels by target
        plt.subplot(1, 2, 2)
        risk_validation = pd.crosstab(self.df["risk_level"], self.df["Target"])
        sns.heatmap(risk_validation, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Risk Levels vs Actual Outcomes")
        plt.tight_layout()
        plt.show()
