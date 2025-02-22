import pandas as pd
from pathlib import Path
import logging
from typing import Union

logger = logging.getLogger("StudentAnalyzer")


class DataLoader:
    """Handles data loading and initial validation of the dataset."""

    @staticmethod
    def load_data(file_path: Union[str, Path], sep: str = ";") -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Args:
            file_path (Union[str, Path]): Path to the CSV file
            sep (str, optional): Separator used in the CSV. Defaults to ";".

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other loading errors
        """
        try:
            # Convert to Path object for better handling
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Load the dataset
            logger.debug("Loading dataset...")
            df = pd.read_csv(path, sep=sep)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
