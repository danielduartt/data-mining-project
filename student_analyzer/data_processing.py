import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class DataProcessor:
    def __init__(self, file_path: str, logger: logging.Logger):
        self.logger = logger
        self.file_path = file_path
        self.df = self._load_data()
        self._setup_data_types()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def _load_data(self) -> pd.DataFrame:
        self.logger.debug("Loading dataset...")
        df = pd.read_csv(self.file_path, sep=";")
        self.logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    
    def _setup_data_types(self):
        """Converte tipos de dados para otimizar memória."""
        self.logger.info("Setting up data types and optimizing memory usage")
        self.numeric_features = []
        self.categorical_features = []

        for col in self.df.columns:
            if col == "Target":
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_features.append(col)
                self._optimize_numeric_column(col)
            else:
                self.categorical_features.append(col)
                self.df[col] = self.df[col].astype("category")
    
    def _optimize_numeric_column(self, col: str):
        """Otimiza colunas numéricas para reduzir o uso de memória."""
        if self.df[col].dtype == "float64":
            if self.df[col].nunique() < 100:
                self.df[col] = self.df[col].astype("float32")
        elif self.df[col].dtype == "int64":
            n_unique = self.df[col].nunique()
            if n_unique < 50:
                self.df[col] = self.df[col].astype("int8")
            elif n_unique < 100:
                self.df[col] = self.df[col].astype("int16")
            else:
                self.df[col] = self.df[col].astype("int32")
    
    def preprocess_data(self):
        """Pré-processa os dados para análise e modelagem."""
        self.logger.info("Preprocessing data...")
        self._handle_missing_values()
        self._encode_categorical_features()
        self._scale_numeric_features()
    
    def _handle_missing_values(self):
        """Trata valores ausentes."""
        self.logger.debug("Handling missing values...")
        self.df.fillna(0, inplace=True)  # Substitui valores ausentes por 0
    
    def _encode_categorical_features(self):
        """Codifica variáveis categóricas."""
        self.logger.debug("Encoding categorical features...")
        for col in self.categorical_features:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])
    
    def _scale_numeric_features(self):
        """Normaliza variáveis numéricas."""
        self.logger.debug("Scaling numeric features...")
        self.df[self.numeric_features] = self.scaler.fit_transform(self.df[self.numeric_features])
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Retorna os dados pré-processados."""
        return self.df