
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

from ..utils.logger import get_logger
from ..utils.constants import (
    CRITICAL_FIELDS,
    AQI_MIN,
    AQI_MAX
)


class DataQualityValidatorError(Exception):
    """Custom exception for Data Quality Validator operations."""
    pass


class DataQualityValidator:
    """
    Comprehensive data quality validator for air quality data.

    This class performs validation checks including:
    - Missing value detection
    - Out-of-range value detection
    - Duplicate record detection
    - Quality score calculation
    - Alert generation for quality issues

    Attributes:
        logger (logging.Logger): Logger instance for this module
        validation_results (Dict): Results from last validation run
    """

    def __init__(self):
        """Initialize Data Quality Validator."""
        self.logger = get_logger(__name__)
        self.validation_results: Dict = {}

    def validate_data(self, df: pd.DataFrame) -> Dict:
       
        try:
            self.logger.info(f"Starting data quality validation on {len(df)} records")

            # Initialize results
            results = {
                'total_records': len(df),
                'timestamp': datetime.now().isoformat(),
                'missing_values': {},
                'out_of_range': {},
                'duplicates': 0,
                'quality_score': 0.0,
                'alerts': []
            }

            # Check for empty DataFrame
            if df.empty:
                results['quality_score'] = 0.0
                results['alerts'].append({
                    'level': 'critical',
                    'message': 'Empty DataFrame provided for validation'
                })
                self.validation_results = results
                return results

            # Check for missing values
            results['missing_values'] = self._check_missing_values(df)

            # Check for out-of-range values
            results['out_of_range'] = self._check_out_of_range(df)

            # Check for duplicates
            results['duplicates'] = self._check_duplicates(df)

            # Calculate quality score
            results['quality_score'] = self._calculate_quality_score(df, results)

            # Generate alerts
            results['alerts'] = self._generate_alerts(results)

            self.validation_results = results

            self.logger.info(
                f"Data quality validation complete: "
                f"quality_score={results['quality_score']:.2f}%, "
                f"alerts={len(results['alerts'])}"
            )

            return results

        except Exception as e:
            error_msg = f"Data quality validation failed: {e}"
            self.logger.error(error_msg)
            raise DataQualityValidatorError(error_msg)

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality report including:
            - summary: Summary statistics
            - by_column: Statistics by column
            - by_city: Statistics by city
            - recommendations: Recommendations for improvement
        """
        try:
            self.logger.info("Generating data quality report")

            # Run validation first
            validation_results = self.validate_data(df)

            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_records': len(df),
                    'quality_score': validation_results['quality_score'],
                    'missing_values_total': sum(validation_results['missing_values'].values()),
                    'out_of_range_total': sum(validation_results['out_of_range'].values()),
                    'duplicates': validation_results['duplicates'],
                    'alerts': len(validation_results['alerts'])
                },
                'by_column': {},
                'by_city': {},
                'recommendations': []
            }

            # Statistics by column
            for col in df.columns:
                report['by_column'][col] = {
                    'missing': df[col].isna().sum(),
                    'missing_pct': (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0,
                    'unique': df[col].nunique(),
                    'dtype': str(df[col].dtype)
                }

            # Statistics by city if city column exists
            if 'city' in df.columns:
                for city in df['city'].unique():
                    city_data = df[df['city'] == city]
                    report['by_city'][city] = {
                        'records': len(city_data),
                        'missing_aqi': city_data['aqi'].isna().sum() if 'aqi' in city_data.columns else 0,
                        'aqi_range': {
                            'min': city_data['aqi'].min() if 'aqi' in city_data.columns else None,
                            'max': city_data['aqi'].max() if 'aqi' in city_data.columns else None
                        }
                    }

            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(validation_results)

            self.logger.info("Data quality report generated successfully")

            return report

        except Exception as e:
            error_msg = f"Failed to generate quality report: {e}"
            self.logger.error(error_msg)
            raise DataQualityValidatorError(error_msg)

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for missing values in each column.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with missing value counts by column
        """
        missing_values = {}

        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
                self.logger.warning(
                    f"Column '{col}' has {missing_count} missing values "
                    f"({missing_count/len(df)*100:.2f}%)"
                )

        return missing_values

    def _check_out_of_range(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for out-of-range values.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with out-of-range counts by column
        """
        out_of_range = {}

        # Check AQI range
        if 'aqi' in df.columns:
            invalid_aqi = ((df['aqi'] < AQI_MIN) | (df['aqi'] > AQI_MAX)).sum()
            if invalid_aqi > 0:
                out_of_range['aqi'] = int(invalid_aqi)
                self.logger.warning(
                    f"Column 'aqi' has {invalid_aqi} values outside range [{AQI_MIN}, {AQI_MAX}]"
                )

        return out_of_range

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """
        Check for duplicate records.

        Args:
            df: DataFrame to check

        Returns:
            Number of duplicate records
        """
        # Check for duplicates on (city, timestamp, pollutant_type) if available
        if 'city' in df.columns and 'timestamp' in df.columns:
            if 'pollutant_type' in df.columns:
                duplicates = df.duplicated(subset=['city', 'timestamp', 'pollutant_type']).sum()
            else:
                duplicates = df.duplicated(subset=['city', 'timestamp']).sum()

            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate records")

            return int(duplicates)

        return 0

    def _calculate_quality_score(self, df: pd.DataFrame, results: Dict) -> float:
        """
        Calculate overall data quality score.

        Quality score is the percentage of records that:
        - Have no missing critical fields
        - Have valid AQI values
        - Are not duplicates

        Args:
            df: DataFrame being validated
            results: Validation results dictionary

        Returns:
            Quality score as percentage (0-100)
        """
        if len(df) == 0:
            return 0.0

        # Count records with issues
        issues = 0

        # Records with missing critical fields
        for field in CRITICAL_FIELDS:
            if field in df.columns:
                issues += df[field].isna().sum()

        # Records with out-of-range AQI
        if 'aqi' in df.columns:
            issues += ((df['aqi'] < AQI_MIN) | (df['aqi'] > AQI_MAX)).sum()

        # Duplicate records
        if 'city' in df.columns and 'timestamp' in df.columns:
            if 'pollutant_type' in df.columns:
                issues += df.duplicated(subset=['city', 'timestamp', 'pollutant_type']).sum()
            else:
                issues += df.duplicated(subset=['city', 'timestamp']).sum()

        # Calculate quality score
        quality_score = ((len(df) - issues) / len(df)) * 100

        return round(quality_score, 2)

    def _generate_alerts(self, results: Dict) -> List[Dict]:
        """
        Generate alerts for data quality issues.

        Args:
            results: Validation results dictionary

        Returns:
            List of alert dictionaries with level and message
        """
        alerts = []

        # Alert for low quality score
        if results['quality_score'] < 80:
            alerts.append({
                'level': 'warning',
                'message': f"Quality score below 80%: {results['quality_score']:.2f}%"
            })

        # Alert for missing values
        if results['missing_values']:
            missing_count = sum(results['missing_values'].values())
            alerts.append({
                'level': 'warning',
                'message': f"Found {missing_count} missing values in {len(results['missing_values'])} columns"
            })

        # Alert for out-of-range values
        if results['out_of_range']:
            out_of_range_count = sum(results['out_of_range'].values())
            alerts.append({
                'level': 'warning',
                'message': f"Found {out_of_range_count} out-of-range values"
            })

        # Alert for duplicates
        if results['duplicates'] > 0:
            alerts.append({
                'level': 'info',
                'message': f"Found {results['duplicates']} duplicate records"
            })

        return alerts

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """
        Generate recommendations for data quality improvement.

        Args:
            results: Validation results dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Recommendation for low quality score
        if results['quality_score'] < 80:
            recommendations.append(
                "Quality score is below 80%. Consider investigating data sources and validation rules."
            )

        # Recommendation for missing values
        if results['missing_values']:
            recommendations.append(
                f"Address missing values in {len(results['missing_values'])} columns. "
                "Consider imputation or data source verification."
            )

        # Recommendation for out-of-range values
        if results['out_of_range']:
            recommendations.append(
                "Out-of-range values detected. Verify data source and validation thresholds."
            )

        # Recommendation for duplicates
        if results['duplicates'] > 0:
            recommendations.append(
                "Duplicate records detected. Implement deduplication in ETL pipeline."
            )

        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring.")

        return recommendations

    def get_validation_results(self) -> Dict:
        """
        Get results from last validation run.

        Returns:
            Dictionary with validation results
        """
        return self.validation_results
