import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("StudentAnalyzer")


class DataPreprocessor:
    """Handles data preprocessing, type optimization, and feature engineering."""

    def __init__(self):
        # Initialize feature lists and preprocessing objects
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.scaler = StandardScaler()

    def setup_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Setup and optimize data types for better memory usage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized data types
        """
        logger.info("Setting up data types and optimizing memory usage")

        # Track initial memory usage
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        try:
            # Reset feature lists
            self.numeric_features.clear()
            self.categorical_features.clear()

            # Process each column
            for col in df.columns:
                # Skip target column
                if col == "Target":
                    continue

                # Process numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    logger.debug(f"Processing numeric column: {col}")
                    self.numeric_features.append(col)
                    df[col] = self._optimize_numeric_column(df[col])

                # Process categorical columns
                else:
                    logger.debug(f"Processing categorical column: {col}")
                    self.categorical_features.append(col)
                    df[col] = df[col].astype("category")
                    logger.debug(f"Converted {col} to category type")

            # Track final memory usage
            final_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_reduction = ((initial_memory - final_memory) / initial_memory) * 100

            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Memory reduction: {memory_reduction:.2f}%")
            logger.info(f"Number of numeric features: {len(self.numeric_features)}")
            logger.info(
                f"Number of categorical features: {len(self.categorical_features)}"
            )

            return df

        except Exception as e:
            logger.error(f"Error in data type setup: {str(e)}")
            raise

    def _optimize_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Optimize numeric column by converting to appropriate data type.

        Args:
            series: Input numeric series

        Returns:
            Optimized series
        """
        try:
            # Optimize float columns
            if series.dtype == "float64":
                if series.nunique() < 100:
                    return series.astype("float32")

            # Optimize integer columns
            elif series.dtype == "int64":
                n_unique = series.nunique()
                if n_unique < 50:
                    return series.astype("int8")
                elif n_unique < 100:
                    return series.astype("int16")
                else:
                    return series.astype("int32")

            return series

        except Exception as e:
            logger.error(f"Error optimizing numeric column: {str(e)}")
            return series

    def prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling by creating features and scaling.

        Args:
            df: Input DataFrame

        Returns:
            Tuple containing features matrix (X) and target vector (y)
        """
        try:
            logger.info("Preparing data for modeling")

            # Create binary target (dropout vs non-dropout)
            y = (df["Target"] == 0).astype(int)

            # Prepare feature matrix
            X = df[self.numeric_features].copy()

            # Handle categorical features
            for col in self.categorical_features:
                if col != "Target":
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col)
                    X = pd.concat([X, dummies], axis=1)

            # Scale numeric features
            X[self.numeric_features] = self.scaler.fit_transform(
                X[self.numeric_features]
            )

            logger.info(f"Prepared modeling data with shape: {X.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
