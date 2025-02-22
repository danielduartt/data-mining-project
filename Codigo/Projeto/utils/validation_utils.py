import pandas as pd
import numpy as np
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger("StudentAnalyzer")


class DataValidator:
    """Validates data integrity and structure for student analysis."""

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_columns: List[str]
    ) -> bool:
        """
        Check if all required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            bool: True if all required columns are present

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True

    @staticmethod
    def validate_target_values(df: pd.DataFrame, valid_targets: Set[int]) -> bool:
        """
        Validate that target column contains only valid values.

        Args:
            df: DataFrame to validate
            valid_targets: Set of valid target values

        Returns:
            bool: True if all target values are valid

        Raises:
            ValueError: If invalid target values are found
        """
        invalid_targets = set(df["Target"].unique()) - valid_targets
        if invalid_targets:
            raise ValueError(f"Invalid target values found: {invalid_targets}")
        return True

    @staticmethod
    def validate_numeric_ranges(
        df: pd.DataFrame, numeric_ranges: Dict[str, tuple]
    ) -> bool:
        """
        Validate numeric columns are within specified ranges.

        Args:
            df: DataFrame to validate
            numeric_ranges: Dictionary of column names and their valid (min, max) ranges

        Returns:
            bool: True if all values are within valid ranges

        Raises:
            ValueError: If values outside valid ranges are found
        """
        for col, (min_val, max_val) in numeric_ranges.items():
            if col not in df.columns:
                continue

            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col]

            if not out_of_range.empty:
                raise ValueError(
                    f"Column {col} contains values outside valid range "
                    f"({min_val}, {max_val}): {out_of_range.tolist()}"
                )
        return True

    @staticmethod
    def validate_no_missing_values(
        df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> bool:
        """
        Check for missing values in specified columns.

        Args:
            df: DataFrame to validate
            columns: Optional list of columns to check (defaults to all columns)

        Returns:
            bool: True if no missing values are found

        Raises:
            ValueError: If missing values are found
        """
        cols_to_check = columns if columns is not None else df.columns
        missing_counts = df[cols_to_check].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if not cols_with_missing.empty:
            raise ValueError(
                "Missing values found in columns: " f"{cols_with_missing.to_dict()}"
            )
        return True

    @staticmethod
    def validate_semester_consistency(df: pd.DataFrame) -> bool:
        """
        Validate consistency between semester-related columns.

        Returns:
            bool: True if semester data is consistent

        Raises:
            ValueError: If inconsistencies are found in semester data
        """
        for sem in range(1, 7):  # Assuming maximum 6 semesters
            enrolled_col = f"Curricular units {sem}st sem (enrolled)"
            approved_col = f"Curricular units {sem}st sem (approved)"

            if enrolled_col not in df.columns or approved_col not in df.columns:
                continue

            # Check if approved units don't exceed enrolled units
            invalid_rows = df[df[approved_col] > df[enrolled_col]]
            if not invalid_rows.empty:
                raise ValueError(
                    f"Found {len(invalid_rows)} rows where approved units exceed "
                    f"enrolled units in semester {sem}"
                )
        return True

    @staticmethod
    def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Validate that columns have expected data types.

        Args:
            df: DataFrame to validate
            expected_types: Dictionary of column names and their expected types

        Returns:
            bool: True if all columns have expected types

        Raises:
            ValueError: If columns have incorrect types
        """
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue

            if str(df[col].dtype) != expected_type:
                raise ValueError(
                    f"Column {col} has type {df[col].dtype}, "
                    f"expected {expected_type}"
                )
        return True

    @staticmethod
    def validate_unique_constraints(
        df: pd.DataFrame, unique_columns: List[str]
    ) -> bool:
        """
        Validate uniqueness constraints on specified columns.

        Args:
            df: DataFrame to validate
            unique_columns: List of columns that should contain unique values

        Returns:
            bool: True if all uniqueness constraints are satisfied

        Raises:
            ValueError: If duplicate values are found
        """
        for col in unique_columns:
            if col not in df.columns:
                continue

            duplicates = df[df[col].duplicated()]
            if not duplicates.empty:
                raise ValueError(
                    f"Found {len(duplicates)} duplicate values in column {col}"
                )
        return True

    @staticmethod
    def validate_category_values(
        df: pd.DataFrame, valid_categories: Dict[str, Set]
    ) -> bool:
        """
        Validate that categorical columns contain only valid values.

        Args:
            df: DataFrame to validate
            valid_categories: Dictionary of column names and their valid values

        Returns:
            bool: True if all categorical values are valid

        Raises:
            ValueError: If invalid category values are found
        """
        for col, valid_values in valid_categories.items():
            if col not in df.columns:
                continue

            invalid_values = set(df[col].unique()) - valid_values
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in column {col}: {invalid_values}"
                )
        return True
