"""
Feature Normalizer module for Air Quality Prediction System.

This module implements feature normalization using StandardScaler with
proper train-test separation and serialization for reproducibility.
"""

import os
import pickle
from datetime import datetime
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger


class FeatureNormalizerError(Exception):
    """Custom exception for Feature Normalizer operations."""
    pass


class FeatureNormalizer:
  
    def __init__(self):
        """Initialize Feature Normalizer."""
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.logger = get_logger(__name__)

    def fit_and_transform(
        self,
        X_train: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
      
        try:
            # Determine feature columns if not provided
            if feature_columns is None:
                feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

            self.feature_columns = feature_columns

            self.logger.info(
                f"Fitting scaler on {len(X_train)} training records "
                f"with {len(feature_columns)} features"
            )

            # Create and fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(X_train[feature_columns])

            # Transform training data
            X_train_normalized = X_train.copy()
            X_train_normalized[feature_columns] = self.scaler.transform(
                X_train[feature_columns]
            )

            self.logger.info(
                f"Scaler fitted successfully. Mean: {self.scaler.mean_}, "
                f"Std: {self.scaler.scale_}"
            )

            return X_train_normalized

        except Exception as e:
            error_msg = f"Failed to fit and transform training data: {e}"
            self.logger.error(error_msg)
            raise FeatureNormalizerError(error_msg)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       
        try:
            if self.scaler is None:
                raise FeatureNormalizerError("Scaler not fitted. Call fit_and_transform first.")

            self.logger.info(
                f"Transforming {len(X)} records with {len(self.feature_columns)} features"
            )

            X_normalized = X.copy()
            X_normalized[self.feature_columns] = self.scaler.transform(
                X[self.feature_columns]
            )

            return X_normalized

        except Exception as e:
            error_msg = f"Failed to transform data: {e}"
            self.logger.error(error_msg)
            raise FeatureNormalizerError(error_msg)

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        
        return self.fit_and_transform(X_train, feature_columns)

    def serialize(self, path: str) -> None:
     
        try:
            if self.scaler is None:
                raise FeatureNormalizerError("Scaler not fitted. Cannot serialize.")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

            # Serialize scaler and metadata
            artifacts = {
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.now().isoformat()
            }

            with open(path, 'wb') as f:
                pickle.dump(artifacts, f)

            self.logger.info(f"Scaler serialized to {path}")

        except Exception as e:
            error_msg = f"Failed to serialize scaler: {e}"
            self.logger.error(error_msg)
            raise FeatureNormalizerError(error_msg)

    def deserialize(self, path: str) -> None:
        
        try:
            if not os.path.exists(path):
                raise FeatureNormalizerError(f"Serialized scaler not found at {path}")

            with open(path, 'rb') as f:
                artifacts = pickle.load(f)

            self.scaler = artifacts['scaler']
            self.feature_columns = artifacts['feature_columns']

            self.logger.info(
                f"Scaler deserialized from {path}. "
                f"Features: {len(self.feature_columns)}"
            )

        except Exception as e:
            error_msg = f"Failed to deserialize scaler: {e}"
            self.logger.error(error_msg)
            raise FeatureNormalizerError(error_msg)

    def get_scaler_params(self) -> dict:
      
        try:
            if self.scaler is None:
                raise FeatureNormalizerError("Scaler not fitted.")

            return {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': self.scaler.var_.tolist(),
                'feature_columns': self.feature_columns
            }

        except Exception as e:
            error_msg = f"Failed to get scaler parameters: {e}"
            self.logger.error(error_msg)
            raise FeatureNormalizerError(error_msg)

    def is_fitted(self) -> bool:
        
        return self.scaler is not None
