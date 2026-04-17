import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from src.feature_engineering.feature_analyzer import (
    FeatureAnalyzer,
    FeatureAnalyzerError
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with features."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'city': ['Delhi'] * 50 + ['Mumbai'] * 50,
        'aqi': np.random.uniform(50, 200, 100),
        'pm25': np.random.uniform(20, 100, 100),
        'pm10': np.random.uniform(30, 150, 100),
        'no2': np.random.uniform(10, 80, 100),
        'o3': np.random.uniform(5, 60, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    return df


@pytest.fixture
def feature_analyzer(sample_dataframe):
    """Create a FeatureAnalyzer instance."""
    return FeatureAnalyzer(df=sample_dataframe, target_col='aqi')


class TestFeatureAnalyzerInit:
    """Tests for FeatureAnalyzer initialization."""
    
    def test_init_with_valid_dataframe(self, sample_dataframe):
        """Test initialization with valid DataFrame."""
        analyzer = FeatureAnalyzer(df=sample_dataframe)
        assert analyzer.df is not None
        assert analyzer.target_col == 'aqi'
        assert len(analyzer.numeric_cols) > 0
    
    def test_init_with_custom_target_col(self, sample_dataframe):
        """Test initialization with custom target column."""
        analyzer = FeatureAnalyzer(df=sample_dataframe, target_col='pm25')
        assert analyzer.target_col == 'pm25'
    
    def test_init_with_none_dataframe_raises_error(self):
        """Test that None DataFrame raises error."""
        with pytest.raises(FeatureAnalyzerError):
            FeatureAnalyzer(df=None)
    
    def test_init_with_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises error."""
        with pytest.raises(FeatureAnalyzerError):
            FeatureAnalyzer(df=pd.DataFrame())
    
    def test_init_with_missing_target_col_raises_error(self, sample_dataframe):
        """Test that missing target column raises error."""
        with pytest.raises(FeatureAnalyzerError):
            FeatureAnalyzer(df=sample_dataframe, target_col='nonexistent')


class TestComputeFeatureStatistics:
    """Tests for compute_feature_statistics method."""
    
    def test_compute_feature_statistics_returns_dataframe(self, feature_analyzer):
        """Test that statistics are returned as DataFrame."""
        stats = feature_analyzer.compute_feature_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
    
    def test_compute_feature_statistics_includes_required_columns(
        self,
        feature_analyzer
    ):
        """Test that statistics include required columns."""
        stats = feature_analyzer.compute_feature_statistics()
        
        required_cols = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']
        for col in required_cols:
            assert col in stats.columns
    
    def test_compute_feature_statistics_includes_null_info(self, feature_analyzer):
        """Test that statistics include null information."""
        stats = feature_analyzer.compute_feature_statistics()
        
        assert 'null_count' in stats.columns
        assert 'null_percentage' in stats.columns
    
    def test_compute_feature_statistics_with_missing_values(self):
        """Test statistics computation with missing values."""
        df = pd.DataFrame({
            'aqi': [10, 20, np.nan, 40, 50],
            'pm25': [5, np.nan, 15, 20, 25]
        })
        
        analyzer = FeatureAnalyzer(df=df)
        stats = analyzer.compute_feature_statistics()
        
        assert stats.loc['aqi', 'null_count'] == 1
        assert stats.loc['pm25', 'null_count'] == 1


class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix method."""
    
    def test_compute_correlation_matrix_returns_dataframe(self, feature_analyzer):
        """Test that correlation matrix is returned as DataFrame."""
        corr = feature_analyzer.compute_correlation_matrix()
        
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]  # Square matrix
    
    def test_compute_correlation_matrix_diagonal_is_one(self, feature_analyzer):
        """Test that diagonal of correlation matrix is 1."""
        corr = feature_analyzer.compute_correlation_matrix()
        
        for i in range(len(corr)):
            assert abs(corr.iloc[i, i] - 1.0) < 0.01
    
    def test_compute_correlation_matrix_symmetric(self, feature_analyzer):
        """Test that correlation matrix is symmetric."""
        corr = feature_analyzer.compute_correlation_matrix()
        
        assert np.allclose(corr, corr.T)
    
    def test_compute_correlation_matrix_exclude_target(self, feature_analyzer):
        """Test correlation matrix without target column."""
        corr = feature_analyzer.compute_correlation_matrix(include_target=False)
        
        # Should not include target column
        assert 'aqi' not in corr.columns or len(corr) > 0


class TestComputeTargetCorrelations:
    """Tests for compute_target_correlations method."""
    
    def test_compute_target_correlations_returns_series(self, feature_analyzer):
        """Test that target correlations are returned as Series."""
        corr = feature_analyzer.compute_target_correlations()
        
        assert isinstance(corr, pd.Series)
        assert len(corr) > 0
    
    def test_compute_target_correlations_sorted(self, feature_analyzer):
        """Test that correlations are sorted by absolute value."""
        corr = feature_analyzer.compute_target_correlations()
        
        abs_corr = corr.abs()
        assert abs_corr.is_monotonic_decreasing
    
    def test_compute_target_correlations_range(self, feature_analyzer):
        """Test that correlations are in valid range."""
        corr = feature_analyzer.compute_target_correlations()
        
        assert (corr >= -1).all() and (corr <= 1).all()


class TestIdentifyFeatureImportance:
    """Tests for identify_feature_importance method."""
    
    def test_identify_feature_importance_returns_dict(self, feature_analyzer):
        """Test that importance is returned as dictionary."""
        importance = feature_analyzer.identify_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_identify_feature_importance_range(self, feature_analyzer):
        """Test that importance scores are in 0-1 range."""
        importance = feature_analyzer.identify_feature_importance()
        
        for score in importance.values():
            assert 0 <= score <= 1
    
    def test_identify_feature_importance_includes_all_features(
        self,
        feature_analyzer
    ):
        """Test that importance includes all numeric features."""
        importance = feature_analyzer.identify_feature_importance()
        
        for col in feature_analyzer.numeric_cols:
            assert col in importance


class TestGetFeatureDistributions:
    """Tests for get_feature_distributions method."""
    
    def test_get_feature_distributions_returns_dict(self, feature_analyzer):
        """Test that distributions are returned as dictionary."""
        distributions = feature_analyzer.get_feature_distributions()
        
        assert isinstance(distributions, dict)
        assert len(distributions) > 0
    
    def test_get_feature_distributions_includes_required_stats(
        self,
        feature_analyzer
    ):
        """Test that distributions include required statistics."""
        distributions = feature_analyzer.get_feature_distributions()
        
        required_stats = ['mean', 'std', 'min', 'max', 'median']
        
        for dist in distributions.values():
            for stat in required_stats:
                assert stat in dist
    
    def test_get_feature_distributions_for_all_features(self, feature_analyzer):
        """Test that distributions are computed for all features."""
        distributions = feature_analyzer.get_feature_distributions()
        
        for col in feature_analyzer.numeric_cols:
            assert col in distributions


class TestIdentifyOutliers:
    """Tests for identify_outliers method."""
    
    def test_identify_outliers_iqr_method(self, feature_analyzer):
        """Test outlier identification using IQR method."""
        outliers = feature_analyzer.identify_outliers(method='iqr')
        
        assert isinstance(outliers, dict)
        assert len(outliers) > 0
    
    def test_identify_outliers_zscore_method(self, feature_analyzer):
        """Test outlier identification using zscore method."""
        outliers = feature_analyzer.identify_outliers(method='zscore')
        
        assert isinstance(outliers, dict)
        assert len(outliers) > 0
    
    def test_identify_outliers_invalid_method_raises_error(self, feature_analyzer):
        """Test that invalid method raises error."""
        with pytest.raises(FeatureAnalyzerError):
            feature_analyzer.identify_outliers(method='invalid')
    
    def test_identify_outliers_with_extreme_values(self):
        """Test outlier detection with extreme values."""
        df = pd.DataFrame({
            'aqi': [10, 20, 30, 40, 50, 1000],  # 1000 is outlier
            'pm25': [5, 10, 15, 20, 25, 500]    # 500 is outlier
        })
        
        analyzer = FeatureAnalyzer(df=df)
        outliers = analyzer.identify_outliers(method='iqr', threshold=1.5)
        
        # Should detect outliers
        assert len(outliers['aqi']) > 0
        assert len(outliers['pm25']) > 0


class TestCheckFeatureConsistencyAcrossCities:
    """Tests for check_feature_consistency_across_cities method."""
    
    def test_check_consistency_returns_dict(self, feature_analyzer):
        """Test that consistency check returns dictionary."""
        consistency = feature_analyzer.check_feature_consistency_across_cities()
        
        assert isinstance(consistency, dict)
        assert len(consistency) > 0
    
    def test_check_consistency_includes_global_stats(self, feature_analyzer):
        """Test that consistency includes global statistics."""
        consistency = feature_analyzer.check_feature_consistency_across_cities()
        
        for feature_stats in consistency.values():
            assert 'global_mean' in feature_stats
            assert 'global_std' in feature_stats
            assert 'by_city' in feature_stats
    
    def test_check_consistency_includes_per_city_stats(self, feature_analyzer):
        """Test that consistency includes per-city statistics."""
        consistency = feature_analyzer.check_feature_consistency_across_cities()
        
        for feature_stats in consistency.values():
            for city_stats in feature_stats['by_city'].values():
                assert 'mean' in city_stats
                assert 'std' in city_stats
                assert 'count' in city_stats
    
    def test_check_consistency_missing_city_col_raises_error(self, sample_dataframe):
        """Test that missing city column raises error."""
        df = sample_dataframe.drop(columns=['city'])
        analyzer = FeatureAnalyzer(df=df)
        
        with pytest.raises(FeatureAnalyzerError):
            analyzer.check_feature_consistency_across_cities()


class TestGenerateAnalysisReport:
    """Tests for generate_analysis_report method."""
    
    def test_generate_analysis_report_returns_dict(self, feature_analyzer):
        """Test that report is returned as dictionary."""
        report = feature_analyzer.generate_analysis_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'statistics' in report
        assert 'correlations' in report
        assert 'importance' in report
    
    def test_generate_analysis_report_includes_summary(self, feature_analyzer):
        """Test that report includes summary section."""
        report = feature_analyzer.generate_analysis_report()
        
        summary = report['summary']
        assert 'total_records' in summary
        assert 'total_features' in summary
        assert 'target_column' in summary
    
    def test_generate_analysis_report_save_to_file(self, feature_analyzer):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'report.json'
            
            report = feature_analyzer.generate_analysis_report(
                output_path=str(output_path)
            )
            
            assert output_path.exists()
            
            # Verify file content
            with open(output_path) as f:
                saved_report = json.load(f)
            
            assert saved_report['summary']['total_records'] == len(feature_analyzer.df)


class TestGetHighCorrelationFeatures:
    """Tests for get_high_correlation_features method."""
    
    def test_get_high_correlation_features_returns_list(self, feature_analyzer):
        """Test that high correlation features are returned as list."""
        high_corr = feature_analyzer.get_high_correlation_features(threshold=0.5)
        
        assert isinstance(high_corr, list)
    
    def test_get_high_correlation_features_format(self, feature_analyzer):
        """Test that high correlation features have correct format."""
        high_corr = feature_analyzer.get_high_correlation_features(threshold=0.5)
        
        for feature1, feature2, corr_value in high_corr:
            assert isinstance(feature1, str)
            assert isinstance(feature2, str)
            assert isinstance(corr_value, (int, float))
            assert abs(corr_value) > 0.5
    
    def test_get_high_correlation_features_sorted(self, feature_analyzer):
        """Test that high correlation features are sorted."""
        high_corr = feature_analyzer.get_high_correlation_features(threshold=0.3)
        
        if len(high_corr) > 1:
            abs_corrs = [abs(corr) for _, _, corr in high_corr]
            assert abs_corrs == sorted(abs_corrs, reverse=True)


class TestGetLowVarianceFeatures:
    """Tests for get_low_variance_features method."""
    
    def test_get_low_variance_features_returns_list(self, feature_analyzer):
        """Test that low variance features are returned as list."""
        low_var = feature_analyzer.get_low_variance_features(threshold=0.01)
        
        assert isinstance(low_var, list)
    
    def test_get_low_variance_features_with_constant_column(self):
        """Test detection of constant columns."""
        df = pd.DataFrame({
            'aqi': [100, 100, 100, 100, 100],  # Constant
            'pm25': [10, 20, 30, 40, 50]       # Variable
        })
        
        analyzer = FeatureAnalyzer(df=df)
        low_var = analyzer.get_low_variance_features(threshold=0.1)
        
        assert 'aqi' in low_var
        assert 'pm25' not in low_var
