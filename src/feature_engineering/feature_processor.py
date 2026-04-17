import logging
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

# PySpark is optional – only needed for distributed processing
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import StringType
    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False
    SparkSession = None
    SparkDataFrame = None

from src.utils.logger import get_logger
from src.utils.constants import LAG_OFFSETS, ROLLING_WINDOWS, SEASONS

logger = get_logger(__name__)


class FeatureProcessorError(Exception):
    """Custom exception for FeatureProcessor errors."""
    pass


class FeatureProcessor:
    

    def __init__(
        self,
        spark=None,
        lag_offsets: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        
        if spark is not None and not _PYSPARK_AVAILABLE:
            raise FeatureProcessorError(
                "PySpark is not installed. Pass spark=None to use pandas-only mode."
            )

        self.spark = spark
        self.lag_offsets = lag_offsets or LAG_OFFSETS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

        mode = "Spark (distributed)" if spark is not None else "pandas"
        logger.info(
            f"FeatureProcessor initialised in {mode} mode | "
            f"lag_offsets={self.lag_offsets}, rolling_windows={self.rolling_windows}"
        )

    # ------------------------------------------------------------------ #
    # Public API – pandas                                                  #
    # ------------------------------------------------------------------ #

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
       
        try:
            logger.info(f"Processing features for {len(df)} records (pandas mode)")

            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            df = self._compute_lag_features(df)
            df = self._compute_rolling_statistics(df)
            df = self._extract_temporal_features(df)
            df = self._add_seasonal_indicators(df)

            logger.info(f"Feature processing complete. Output shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Feature processing failed: {e}")
            raise FeatureProcessorError(f"Feature processing failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Public API – Spark                                                   #
    # ------------------------------------------------------------------ #

    def process_spark(self, sdf) -> object:
        
        if not _PYSPARK_AVAILABLE:
            raise FeatureProcessorError(
                "PySpark is not installed. Use process() for pandas-only processing."
            )
        try:
            logger.info("Processing features using Spark (distributed)")
            sdf = self._compute_lag_features_spark(sdf)
            sdf = self._compute_rolling_statistics_spark(sdf)
            sdf = self._extract_temporal_features_spark(sdf)
            sdf = self._add_seasonal_indicators_spark(sdf)
            logger.info("Spark feature processing complete")
            return sdf
        except Exception as e:
            logger.error(f"Spark feature processing failed: {e}")
            raise FeatureProcessorError(f"Spark feature processing failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Pandas helpers                                                       #
    # ------------------------------------------------------------------ #

    def _compute_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lag features for each city (pandas)."""
        logger.debug(f"Computing lag features: {self.lag_offsets}")
        for lag in self.lag_offsets:
            df[f'aqi_lag_{lag}h'] = df.groupby('city')['aqi'].shift(lag)
        return df

    def _compute_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling window statistics for each city (pandas)."""
        logger.debug(f"Computing rolling statistics: windows={self.rolling_windows}")
        for window in self.rolling_windows:
            grouped = df.groupby('city')['aqi'].rolling(window=window, min_periods=1)
            df[f'aqi_mean_{window}h'] = grouped.mean().reset_index(level=0, drop=True)
            df[f'aqi_std_{window}h']  = grouped.std().reset_index(level=0, drop=True)
            df[f'aqi_min_{window}h']  = grouped.min().reset_index(level=0, drop=True)
            df[f'aqi_max_{window}h']  = grouped.max().reset_index(level=0, drop=True)
        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, day-of-week, month, weekend flag from timestamp (pandas)."""
        logger.debug("Extracting temporal features")
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month']       = df['timestamp'].dt.month
        df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def _add_seasonal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Indian meteorological season label (pandas)."""
        logger.debug("Adding seasonal indicators")

        def _season(month: int) -> str:
            if month in SEASONS['Winter']:    return 'Winter'
            if month in SEASONS['Summer']:    return 'Summer'
            if month in SEASONS['Monsoon']:   return 'Monsoon'
            return 'Post-Monsoon'

        df['season'] = df['month'].apply(_season)
        return df

    # ------------------------------------------------------------------ #
    # Spark helpers                                                        #
    # ------------------------------------------------------------------ #

    def _compute_lag_features_spark(self, sdf):
        """Compute lag features using Spark window functions."""
        window_spec = Window.partitionBy('city').orderBy('timestamp')
        for lag in self.lag_offsets:
            sdf = sdf.withColumn(f'aqi_lag_{lag}h', F.lag('aqi', lag).over(window_spec))
        return sdf

    def _compute_rolling_statistics_spark(self, sdf):
        """Compute rolling statistics using Spark range windows."""
        for window in self.rolling_windows:
            window_seconds = window * 3600
            ws = Window.partitionBy('city').orderBy('timestamp').rangeBetween(-window_seconds, 0)
            sdf = sdf.withColumn(f'aqi_mean_{window}h', F.avg('aqi').over(ws))
            sdf = sdf.withColumn(f'aqi_std_{window}h',  F.stddev('aqi').over(ws))
            sdf = sdf.withColumn(f'aqi_min_{window}h',  F.min('aqi').over(ws))
            sdf = sdf.withColumn(f'aqi_max_{window}h',  F.max('aqi').over(ws))
        return sdf

    def _extract_temporal_features_spark(self, sdf):
        """Extract temporal features using Spark built-ins."""
        sdf = sdf.withColumn('hour_of_day', F.hour('timestamp'))
        sdf = sdf.withColumn('day_of_week', F.dayofweek('timestamp') - 1)
        sdf = sdf.withColumn('month',       F.month('timestamp'))
        sdf = sdf.withColumn('is_weekend',
                             F.when(F.col('day_of_week').isin([5, 6]), 1).otherwise(0))
        return sdf

    def _add_seasonal_indicators_spark(self, sdf):
        """Add season column using a Spark UDF."""
        def _season(month: int) -> str:
            if month in SEASONS['Winter']:  return 'Winter'
            if month in SEASONS['Summer']:  return 'Summer'
            if month in SEASONS['Monsoon']: return 'Monsoon'
            return 'Post-Monsoon'

        season_udf = F.udf(_season, StringType())
        sdf = sdf.withColumn('season', season_udf(F.col('month')))
        return sdf

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def get_feature_columns(self) -> List[str]:
        """Return the list of all feature column names that will be created."""
        features = []
        features.extend([f'aqi_lag_{lag}h' for lag in self.lag_offsets])
        for window in self.rolling_windows:
            features.extend([
                f'aqi_mean_{window}h', f'aqi_std_{window}h',
                f'aqi_min_{window}h',  f'aqi_max_{window}h',
            ])
        features.extend(['hour_of_day', 'day_of_week', 'month', 'is_weekend', 'season'])
        return features
