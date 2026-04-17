import os
from datetime import datetime
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

from ..utils.logger import get_logger
from ..utils.constants import (
    LAG_OFFSETS,
    ROLLING_WINDOWS,
    SEASONS,
    CRITICAL_FIELDS
)


class GoldLayerError(Exception):
    """Custom exception for Gold Layer operations."""
    pass


class GoldLayer:
  

    def __init__(self, storage_path: str, spark: SparkSession):
        
        self.storage_path = storage_path
        self.spark = spark
        self.logger = get_logger(__name__)

        # Create storage directory if it doesn't exist
        try:
            os.makedirs(storage_path, exist_ok=True)
            self.logger.info(
                f"Gold Layer initialized at {storage_path} "
                f"at {datetime.now().isoformat()}"
            )
        except Exception as e:
            error_msg = f"Failed to create Gold Layer directory: {e}"
            self.logger.error(error_msg)
            raise GoldLayerError(error_msg)

    def transform_silver_to_gold(
        self,
        silver_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        try:
            self.logger.info(
                f"Starting Gold Layer transformation with {len(silver_df)} records"
            )

            # Convert to Spark DataFrame for distributed processing
            sdf = self.spark.createDataFrame(silver_df)

            # Sort by city and timestamp
            sdf = sdf.sort('city', 'timestamp')

            # Step 1: Compute lag features
            sdf = self._compute_lag_features(sdf)

            # Step 2: Compute rolling statistics
            sdf = self._compute_rolling_statistics(sdf)

            # Step 3: Extract temporal features
            sdf = self._extract_temporal_features(sdf)

            # Step 4: Add seasonal indicators
            sdf = self._add_seasonal_indicators(sdf)

            # Convert to Pandas for missing value handling
            gold_df = sdf.toPandas()

            # Step 5: Handle missing values
            gold_df = self._handle_missing_values(gold_df)

            self.logger.info(
                f"Gold Layer transformation complete with {len(gold_df)} records"
            )

            return gold_df

        except Exception as e:
            error_msg = f"Failed to transform Silver to Gold: {e}"
            self.logger.error(error_msg)
            raise GoldLayerError(error_msg)

    def store_data(self, df: pd.DataFrame) -> int:
        
        try:
            self.logger.info(
                f"Starting data storage in Gold Layer with {len(df)} records"
            )

            # Convert to Spark DataFrame
            sdf = self.spark.createDataFrame(df)

            # Add date column for partitioning
            sdf = sdf.withColumn(
                'date',
                F.to_date(F.col('timestamp'))
            )

            # Write to Parquet with partitioning by city and date
            sdf.write \
                .mode("append") \
                .partitionBy("city", "date") \
                .parquet(self.storage_path)

            record_count = len(df)
            self.logger.info(
                f"Successfully stored {record_count} records in Gold Layer"
            )

            return record_count

        except Exception as e:
            error_msg = f"Failed to store data in Gold Layer: {e}"
            self.logger.error(error_msg)
            raise GoldLayerError(error_msg)

    def read_data(
        self,
        city: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        
        try:
            # Check if path exists
            if not os.path.exists(self.storage_path):
                self.logger.warning(f"Gold Layer path does not exist: {self.storage_path}")
                return pd.DataFrame()

            # Read Parquet files
            sdf = self.spark.read.parquet(self.storage_path)

            # Apply city filter if provided
            if city:
                sdf = sdf.filter(F.col('city') == city)

            # Apply date filter if provided
            if date:
                sdf = sdf.filter(F.col('date') == date)

            # Convert to Pandas
            df = sdf.toPandas()

            self.logger.info(
                f"Read {len(df)} records from Gold Layer "
                f"(city={city}, date={date})"
            )

            return df

        except Exception as e:
            error_msg = f"Failed to read data from Gold Layer: {e}"
            self.logger.error(error_msg)
            raise GoldLayerError(error_msg)

    def _compute_lag_features(self, sdf) -> any:
        
        try:
            # Define window specification for lag computation
            window_spec = Window.partitionBy('city').orderBy('timestamp')

            # Compute lag features
            for lag_offset in LAG_OFFSETS:
                sdf = sdf.withColumn(
                    f'aqi_lag_{lag_offset}h',
                    F.lag('aqi', lag_offset).over(window_spec)
                )

            self.logger.info(f"Computed {len(LAG_OFFSETS)} lag features")

            return sdf

        except Exception as e:
            self.logger.error(f"Failed to compute lag features: {e}")
            raise

    def _compute_rolling_statistics(self, sdf) -> any:
        
        try:
            # Define window specifications for rolling statistics
            # Using row-based windows (assuming hourly data)
            for window_size in ROLLING_WINDOWS:
                window_spec = Window.partitionBy('city').orderBy('timestamp').rowsBetween(
                    -(window_size - 1), 0
                )

                sdf = sdf.withColumn(
                    f'aqi_mean_{window_size}h',
                    F.avg('aqi').over(window_spec)
                )
                sdf = sdf.withColumn(
                    f'aqi_std_{window_size}h',
                    F.stddev('aqi').over(window_spec)
                )
                sdf = sdf.withColumn(
                    f'aqi_min_{window_size}h',
                    F.min('aqi').over(window_spec)
                )
                sdf = sdf.withColumn(
                    f'aqi_max_{window_size}h',
                    F.max('aqi').over(window_spec)
                )

            self.logger.info(
                f"Computed rolling statistics for {len(ROLLING_WINDOWS)} windows"
            )

            return sdf

        except Exception as e:
            self.logger.error(f"Failed to compute rolling statistics: {e}")
            raise

    def _extract_temporal_features(self, sdf) -> any:
        
        try:
            sdf = sdf.withColumn('hour_of_day', F.hour('timestamp'))
            sdf = sdf.withColumn('day_of_week', F.dayofweek('timestamp') - 1)
            sdf = sdf.withColumn('month', F.month('timestamp'))
            sdf = sdf.withColumn(
                'is_weekend',
                F.col('day_of_week').isin([5, 6]).cast('int')
            )

            self.logger.info("Extracted temporal features")

            return sdf

        except Exception as e:
            self.logger.error(f"Failed to extract temporal features: {e}")
            raise

    def _add_seasonal_indicators(self, sdf) -> any:
        
        try:
            def get_season(month):
                """Determine season from month."""
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Summer'
                elif month in [6, 7, 8, 9]:
                    return 'Monsoon'
                else:
                    return 'Post-Monsoon'

            # Create UDF for season determination
            season_udf = F.udf(get_season, StringType())
            sdf = sdf.withColumn('season', season_udf(F.col('month')))

            self.logger.info("Added seasonal indicators")

            return sdf

        except Exception as e:
            self.logger.error(f"Failed to add seasonal indicators: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        
        try:
            initial_nulls = df.isnull().sum().sum()

            # Group by city and apply forward-fill then backward-fill
            df_filled = df.groupby('city', group_keys=False).apply(
                lambda group: group.fillna(method='ffill').fillna(method='bfill')
            )

            # Handle any remaining nulls (e.g., all missing in a column)
            # Fill with 0 for numeric columns
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_filled[col] = df_filled[col].fillna(0)

            final_nulls = df_filled.isnull().sum().sum()

            self.logger.info(
                f"Missing value handling: {initial_nulls} nulls -> {final_nulls} nulls"
            )

            return df_filled

        except Exception as e:
            self.logger.error(f"Failed to handle missing values: {e}")
            raise
