
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pandas as pd
import logging

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, ArrayType

from ..utils.logger import get_logger
from ..utils.constants import (
    CRITICAL_FIELDS,
    AQI_MIN,
    AQI_MAX
)


class SilverLayerError(Exception):
    """Custom exception for Silver Layer operations."""
    pass


class SilverLayer:
    

    def __init__(self, storage_path: str, spark: SparkSession):
       
        self.storage_path = storage_path
        self.spark = spark
        self.logger = get_logger(__name__)
        self.validation_errors: List[Dict] = []

        # Create storage directory if it doesn't exist
        try:
            os.makedirs(storage_path, exist_ok=True)
            self.logger.info(
                f"Silver Layer initialized at {storage_path} "
                f"at {datetime.now().isoformat()}"
            )
        except Exception as e:
            error_msg = f"Failed to create Silver Layer directory: {e}"
            self.logger.error(error_msg)
            raise SilverLayerError(error_msg)

    def transform_bronze_to_silver(
        self,
        bronze_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, int, int]:
        
        try:
            total_records = len(bronze_df)
            self.logger.info(
                f"Starting Silver Layer transformation with {total_records} records"
            )

            # Reset validation errors
            self.validation_errors = []

            # Convert to Spark DataFrame for distributed processing
            sdf = self.spark.createDataFrame(bronze_df)

            # Step 1: Remove duplicates on (city, timestamp, pollutant_type)
            sdf = self._deduplicate(sdf)

            # Step 2: Validate critical fields
            sdf = self._validate_critical_fields(sdf)

            # Step 3: Validate AQI range
            sdf = self._validate_aqi_range(sdf)

            # Step 4: Validate timestamp ordering per city
            sdf = self._validate_timestamp_ordering(sdf)

            # Step 5: Add quality flags
            sdf = self._add_quality_flags(sdf)

            # Convert back to Pandas
            cleaned_df = sdf.toPandas()

            valid_records = len(cleaned_df)
            rejected_records = total_records - valid_records

            self.logger.info(
                f"Silver Layer transformation complete: "
                f"total={total_records}, valid={valid_records}, "
                f"rejected={rejected_records}"
            )

            return cleaned_df, total_records, valid_records, rejected_records

        except Exception as e:
            error_msg = f"Failed to transform Bronze to Silver: {e}"
            self.logger.error(error_msg)
            raise SilverLayerError(error_msg)

    def store_data(self, df: pd.DataFrame) -> int:
        
        try:
            self.logger.info(
                f"Starting data storage in Silver Layer with {len(df)} records"
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
                f"Successfully stored {record_count} records in Silver Layer"
            )

            return record_count

        except Exception as e:
            error_msg = f"Failed to store data in Silver Layer: {e}"
            self.logger.error(error_msg)
            raise SilverLayerError(error_msg)

    def read_data(
        self,
        city: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        
        try:
            # Check if path exists
            if not os.path.exists(self.storage_path):
                self.logger.warning(f"Silver Layer path does not exist: {self.storage_path}")
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
                f"Read {len(df)} records from Silver Layer "
                f"(city={city}, date={date})"
            )

            return df

        except Exception as e:
            error_msg = f"Failed to read data from Silver Layer: {e}"
            self.logger.error(error_msg)
            raise SilverLayerError(error_msg)

    def get_validation_errors(self) -> List[Dict]:
        
        return self.validation_errors

    def _deduplicate(self, sdf) -> any:
        
        try:
            initial_count = sdf.count()

            # Drop duplicates on (city, timestamp, pollutant_type)
            # If pollutant_type doesn't exist, use (city, timestamp)
            if 'pollutant_type' in sdf.columns:
                sdf_dedup = sdf.dropDuplicates(['city', 'timestamp', 'pollutant_type'])
            else:
                sdf_dedup = sdf.dropDuplicates(['city', 'timestamp'])

            final_count = sdf_dedup.count()
            duplicates_removed = initial_count - final_count

            self.logger.info(
                f"Deduplication: removed {duplicates_removed} duplicate records"
            )

            return sdf_dedup

        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            raise

    def _validate_critical_fields(self, sdf) -> any:
        
        try:
            initial_count = sdf.count()

            # Filter out records with missing critical fields
            for field in CRITICAL_FIELDS:
                if field in sdf.columns:
                    sdf = sdf.filter(F.col(field).isNotNull())

            final_count = sdf.count()
            rejected = initial_count - final_count

            self.logger.info(
                f"Critical field validation: rejected {rejected} records with missing fields"
            )

            return sdf

        except Exception as e:
            self.logger.error(f"Critical field validation failed: {e}")
            raise

    def _validate_aqi_range(self, sdf) -> any:
        
        try:
            initial_count = sdf.count()

            # Filter records with AQI in valid range
            sdf = sdf.filter(
                (F.col('aqi') >= AQI_MIN) & (F.col('aqi') <= AQI_MAX)
            )

            final_count = sdf.count()
            rejected = initial_count - final_count

            self.logger.info(
                f"AQI range validation: rejected {rejected} records with AQI outside [{AQI_MIN}, {AQI_MAX}]"
            )

            return sdf

        except Exception as e:
            self.logger.error(f"AQI range validation failed: {e}")
            raise

    def _validate_timestamp_ordering(self, sdf) -> any:
        
        try:
            initial_count = sdf.count()

            # Sort by city and timestamp
            sdf = sdf.sort('city', 'timestamp')

            # Create window to check if timestamp is >= previous timestamp for same city
            window_spec = Window.partitionBy('city').orderBy('timestamp')
            sdf = sdf.withColumn(
                'prev_timestamp',
                F.lag('timestamp').over(window_spec)
            )

            # Keep records where timestamp is >= previous (or first record)
            sdf = sdf.filter(
                (F.col('prev_timestamp').isNull()) |
                (F.col('timestamp') >= F.col('prev_timestamp'))
            )

            # Drop the helper column
            sdf = sdf.drop('prev_timestamp')

            final_count = sdf.count()
            rejected = initial_count - final_count

            self.logger.info(
                f"Timestamp ordering validation: rejected {rejected} records with out-of-order timestamps"
            )

            return sdf

        except Exception as e:
            self.logger.error(f"Timestamp ordering validation failed: {e}")
            raise

    def _add_quality_flags(self, sdf) -> any:
        
        try:
            # Add quality_flags column (empty list for all records that passed validation)
            sdf = sdf.withColumn('quality_flags', F.array())

            self.logger.info("Quality flags added to all records")

            return sdf

        except Exception as e:
            self.logger.error(f"Failed to add quality flags: {e}")
            raise
