import pytest
import pandas as pd
from datetime import datetime

from src.etl_pipeline.data_validator import DataQualityValidator, DataQualityValidatorError


@pytest.fixture
def validator():
    """Create a Data Quality Validator instance."""
    return DataQualityValidator()


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'city': ['Delhi', 'Delhi', 'Mumbai', 'Mumbai'],
        'timestamp': [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 1, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 1, 0)
        ],
        'aqi': [150.0, 160.0, 120.0, 130.0],
        'pm25': [50.0, 55.0, 40.0, 45.0],
        'pm10': [100.0, 110.0, 80.0, 90.0],
        'no2': [30.0, 35.0, 25.0, 30.0],
        'o3': [20.0, 25.0, 15.0, 20.0],
        'so2': [10.0, 12.0, 8.0, 10.0],
        'co': [5.0, 6.0, 4.0, 5.0]
    })


class TestValidateData:
    """Tests for data validation."""

    def test_validate_data_returns_results_dict(self, validator):
        """Test that validate_data returns a results dictionary."""
        df = create_sample_dataframe()

        results = validator.validate_data(df)

        assert isinstance(results, dict)
        assert 'total_records' in results
        assert 'quality_score' in results
        assert 'alerts' in results

    def test_validate_data_with_valid_data(self, validator):
        """Test validation with all valid data."""
        df = create_sample_dataframe()

        results = validator.validate_data(df)

        assert results['total_records'] == 4
        assert results['quality_score'] == 100.0
        assert len(results['alerts']) == 0

    def test_validate_data_with_empty_dataframe(self, validator):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()

        results = validator.validate_data(df)

        assert results['total_records'] == 0
        assert results['quality_score'] == 0.0
        assert len(results['alerts']) > 0

    def test_validate_data_detects_missing_values(self, validator):
        """Test that missing values are detected."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = None

        results = validator.validate_data(df)

        assert 'aqi' in results['missing_values']
        assert results['missing_values']['aqi'] == 1

    def test_validate_data_detects_out_of_range_aqi(self, validator):
        """Test that out-of-range AQI values are detected."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = 550.0

        results = validator.validate_data(df)

        assert 'aqi' in results['out_of_range']
        assert results['out_of_range']['aqi'] == 1

    def test_validate_data_detects_duplicates(self, validator):
        """Test that duplicate records are detected."""
        df = create_sample_dataframe()
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        results = validator.validate_data(df_with_dup)

        assert results['duplicates'] > 0

    def test_quality_score_calculation(self, validator):
        """Test quality score calculation."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = None  # 1 missing value

        results = validator.validate_data(df)

        # Quality score should be 75% (3 valid out of 4)
        assert results['quality_score'] == 75.0


class TestMissingValueDetection:
    """Tests for missing value detection."""

    def test_detect_missing_in_single_column(self, validator):
        """Test detection of missing values in a single column."""
        df = create_sample_dataframe()
        df.loc[0, 'pm25'] = None

        results = validator.validate_data(df)

        assert 'pm25' in results['missing_values']

    def test_detect_missing_in_multiple_columns(self, validator):
        """Test detection of missing values in multiple columns."""
        df = create_sample_dataframe()
        df.loc[0, 'pm25'] = None
        df.loc[1, 'pm10'] = None

        results = validator.validate_data(df)

        assert len(results['missing_values']) >= 2

    def test_no_missing_values_reported_when_none_exist(self, validator):
        """Test that no missing values are reported when none exist."""
        df = create_sample_dataframe()

        results = validator.validate_data(df)

        assert len(results['missing_values']) == 0


class TestOutOfRangeDetection:
    """Tests for out-of-range value detection."""

    def test_detect_negative_aqi(self, validator):
        """Test detection of negative AQI values."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = -10.0

        results = validator.validate_data(df)

        assert 'aqi' in results['out_of_range']

    def test_detect_aqi_above_500(self, validator):
        """Test detection of AQI values above 500."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = 550.0

        results = validator.validate_data(df)

        assert 'aqi' in results['out_of_range']

    def test_accept_boundary_aqi_values(self, validator):
        """Test that boundary AQI values are accepted."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = 0.0
        df.loc[1, 'aqi'] = 500.0

        results = validator.validate_data(df)

        assert 'aqi' not in results['out_of_range']


class TestDuplicateDetection:
    """Tests for duplicate record detection."""

    def test_detect_exact_duplicates(self, validator):
        """Test detection of exact duplicate records."""
        df = create_sample_dataframe()
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        results = validator.validate_data(df_with_dup)

        assert results['duplicates'] == 1

    def test_no_duplicates_reported_when_none_exist(self, validator):
        """Test that no duplicates are reported when none exist."""
        df = create_sample_dataframe()

        results = validator.validate_data(df)

        assert results['duplicates'] == 0


class TestAlertGeneration:
    """Tests for alert generation."""

    def test_alert_generated_for_low_quality_score(self, validator):
        """Test that alert is generated for low quality score."""
        df = create_sample_dataframe()
        # Create multiple issues to lower quality score
        df.loc[0, 'aqi'] = None
        df.loc[1, 'aqi'] = None
        df.loc[2, 'aqi'] = None

        results = validator.validate_data(df)

        # Should have warning alert for low quality score
        assert any(alert['level'] == 'warning' for alert in results['alerts'])

    def test_alert_generated_for_missing_values(self, validator):
        """Test that alert is generated for missing values."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = None

        results = validator.validate_data(df)

        assert any('missing' in alert['message'].lower() for alert in results['alerts'])

    def test_alert_generated_for_duplicates(self, validator):
        """Test that alert is generated for duplicates."""
        df = create_sample_dataframe()
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        results = validator.validate_data(df_with_dup)

        assert any('duplicate' in alert['message'].lower() for alert in results['alerts'])

    def test_no_alerts_for_valid_data(self, validator):
        """Test that no alerts are generated for valid data."""
        df = create_sample_dataframe()

        results = validator.validate_data(df)

        assert len(results['alerts']) == 0


class TestQualityReport:
    """Tests for quality report generation."""

    def test_generate_quality_report_returns_dict(self, validator):
        """Test that quality report is a dictionary."""
        df = create_sample_dataframe()

        report = validator.generate_quality_report(df)

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'by_column' in report
        assert 'recommendations' in report

    def test_quality_report_includes_summary(self, validator):
        """Test that quality report includes summary statistics."""
        df = create_sample_dataframe()

        report = validator.generate_quality_report(df)

        assert 'total_records' in report['summary']
        assert 'quality_score' in report['summary']
        assert 'alerts' in report['summary']

    def test_quality_report_includes_by_column_stats(self, validator):
        """Test that quality report includes per-column statistics."""
        df = create_sample_dataframe()

        report = validator.generate_quality_report(df)

        assert len(report['by_column']) > 0
        for col_stats in report['by_column'].values():
            assert 'missing' in col_stats
            assert 'unique' in col_stats

    def test_quality_report_includes_by_city_stats(self, validator):
        """Test that quality report includes per-city statistics."""
        df = create_sample_dataframe()

        report = validator.generate_quality_report(df)

        assert len(report['by_city']) > 0
        for city_stats in report['by_city'].values():
            assert 'records' in city_stats

    def test_quality_report_includes_recommendations(self, validator):
        """Test that quality report includes recommendations."""
        df = create_sample_dataframe()

        report = validator.generate_quality_report(df)

        assert len(report['recommendations']) > 0

    def test_recommendations_for_low_quality_data(self, validator):
        """Test that recommendations are provided for low quality data."""
        df = create_sample_dataframe()
        df.loc[0, 'aqi'] = None
        df.loc[1, 'aqi'] = None

        report = validator.generate_quality_report(df)

        # Should have recommendations for improvement
        assert len(report['recommendations']) > 0


class TestValidationResults:
    """Tests for validation results tracking."""

    def test_get_validation_results_returns_last_run(self, validator):
        """Test that get_validation_results returns last validation run."""
        df = create_sample_dataframe()

        validator.validate_data(df)
        results = validator.get_validation_results()

        assert results['total_records'] == 4

    def test_validation_results_updated_on_each_run(self, validator):
        """Test that validation results are updated on each run."""
        df1 = create_sample_dataframe()
        df2 = create_sample_dataframe().iloc[:2]

        validator.validate_data(df1)
        results1 = validator.get_validation_results()

        validator.validate_data(df2)
        results2 = validator.get_validation_results()

        assert results1['total_records'] == 4
        assert results2['total_records'] == 2
