import os
from datetime import datetime
from typing import Optional
import pandas as pd
import logging

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

from ..utils.logger import get_logger
from ..utils.constants import (
    CRITICAL_FIELDS,
    AQI_MIN,
    AQI_MAX
)


class BronzeLayerError(Exception):
    """Custom exception for Bronze Layer storage failures."""
    pass


class BronzeLayer:
   

    def __init__(self, storage_path: str, spark: SparkSession):
     
        self.storage_path = storage_path
        self.spark = spark
        self.logger = get_logger(__name__)

        # Create storage directory if it doesn't exist
        try:
            os.makedirs(storage_path, exist_ok=True)
            self.logger.info(
                f"Bronze Layer initialized at {storage_path} "
                f"at {datetime.now().isoformat()}"
            )
        except Exception as e:
            error_msg = f"Failed to create Bronze Layer directory: {e}"
            self.logger.error(error_msg)
            raise BronzeLayerError(error_msg)

    def store_data(self, df: pd.DataFrame, source: str) -> int:
       
        try:
            # Validate schema
            if not self._validate_schema(df):
                raise ValueError("Schema validation failed")

            self.logger.info(
                f"Starting data storage from source '{source}' "
                f"with {len(df)} records"
            )

            # Add metadata columns
            df_with_metadata = self._add_metadata(df, source)

            # Convert to Spark DataFrame
            sdf = self.spark.createDataFrame(df_with_metadata)

            # Add date column for partitioning
            sdf = sdf.withColumn(
                'date',
                F.to_date(F.col('timestamp'))
            )

            # Write to Parquet with partitioning
            output_path = os.path.join(self.storage_path, source)
            sdf.write \
                .mode("append") \
                .partitionBy("date") \
                .parquet(output_path)

            record_count = len(df_with_metadata)
            self.logger.info(
                f"Successfully stored {record_count} records from '{source}' "
                f"in Bronze Layer"
            )

            return record_count

        except ValueError as e:
            self.logger.error(f"Schema validation error: {e}")
            raise
        except Exception as e:
            error_msg = f"Failed to store data in Bronze Layer: {e}"
            self.logger.error(error_msg)
            raise BronzeLayerError(error_msg)

    def read_data(
        self,
        source: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
      
        try:
            # Build path based on filters
            if source:
                path = os.path.join(self.storage_path, source)
            else:
                path = self.storage_path

            # Check if path exists
            if not os.path.exists(path):
                self.logger.warning(f"Bronze Layer path does not exist: {path}")
                return pd.DataFrame()

            # Read Parquet files
            sdf = self.spark.read.parquet(path)

            # Apply date filter if provided
            if date:
                sdf = sdf.filter(F.col('date') == date)

            # Convert to Pandas
            df = sdf.toPandas()

            self.logger.info(
                f"Read {len(df)} records from Bronze Layer "
                f"(source={source}, date={date})"
            )

            return df

        except Exception as e:
            error_msg = f"Failed to read data from Bronze Layer: {e}"
            self.logger.error(error_msg)
            raise BronzeLayerError(error_msg)

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame schema.

        Checks for:
        - Required columns (city, timestamp, aqi)
        - Correct data types
        - Non-empty DataFrame

        Args:
            df: DataFrame to validate

        Returns:
            True if schema is valid, False otherwise
        """
        try:
            # Check for empty DataFrame
            if df.empty:
                self.logger.error("Cannot store empty DataFrame")
                return False

            # Check for required columns
            missing_columns = [col for col in CRITICAL_FIELDS if col not in df.columns]
            if missing_columns:
                self.logger.error(
                    f"Missing required columns: {missing_columns}"
                )
                return False

            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                self.logger.error("Column 'timestamp' must be datetime type")
                return False

            if not pd.api.types.is_numeric_dtype(df['aqi']):
                self.logger.error("Column 'aqi' must be numeric type")
                return False

            if not pd.api.types.is_object_dtype(df['city']):
                self.logger.error("Column 'city' must be string type")
                return False

            # Validate AQI range
            invalid_aqi = df[(df['aqi'] < AQI_MIN) | (df['aqi'] > AQI_MAX)]
            if not invalid_aqi.empty:
                self.logger.warning(
                    f"Found {len(invalid_aqi)} records with AQI outside "
                    f"valid range [{AQI_MIN}, {AQI_MAX}]"
                )

            self.logger.info("Schema validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False

    def _add_metadata(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
      
        df_copy = df.copy()

        # Add source column
        df_copy['source'] = source

        # Add ingestion timestamp
        df_copy['ingestion_timestamp'] = datetime.now()

        # Add retrieval time placeholder
        df_copy['retrieval_time'] = 0.0

        self.logger.debug(
            f"Added metadata columns to {len(df_copy)} records from '{source}'"
        )

        return df_copy
