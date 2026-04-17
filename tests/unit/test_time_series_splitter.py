import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.time_series_splitter import (
    TimeSeriesSplitter,
    TimeSeriesSplitterError
)


@pytest.fixture
def sample_dataframe():
    """Create a sample time-series DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'city': ['Delhi'] * 100,
        'aqi': np.random.uniform(50, 200, 100),
        'pm25': np.random.uniform(20, 100, 100)
    })
    
    return df


@pytest.fixture
def sample_target():
    """Create a sample target series."""
    return pd.Series(np.random.uniform(50, 200, 100), name='aqi')


class TestTimeSeriesSplitterInit:
    """Tests for TimeSeriesSplitter initialization."""
    
    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        splitter = TimeSeriesSplitter()
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.25  # 1/(3+1)
    
    def test_init_with_custom_n_splits(self):
        """Test initialization with custom n_splits."""
        splitter = TimeSeriesSplitter(n_splits=5)
        assert splitter.n_splits == 5
        assert splitter.test_size == 1/6  # 1/(5+1)
    
    def test_init_with_custom_test_size(self):
        """Test initialization with custom test_size."""
        splitter = TimeSeriesSplitter(n_splits=3, test_size=0.2)
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.2
    
    def test_init_with_invalid_n_splits(self):
        """Test that invalid n_splits raises error."""
        with pytest.raises(TimeSeriesSplitterError):
            TimeSeriesSplitter(n_splits=0)
        
        with pytest.raises(TimeSeriesSplitterError):
            TimeSeriesSplitter(n_splits=-1)
    
    def test_init_with_invalid_test_size(self):
        """Test that invalid test_size raises error."""
        with pytest.raises(TimeSeriesSplitterError):
            TimeSeriesSplitter(test_size=0)
        
        with pytest.raises(TimeSeriesSplitterError):
            TimeSeriesSplitter(test_size=1)
        
        with pytest.raises(TimeSeriesSplitterError):
            TimeSeriesSplitter(test_size=-0.1)


class TestSplit:
    """Tests for split method."""
    
    def test_split_generates_correct_number_of_splits(self, sample_dataframe):
        """Test that split generates correct number of folds."""
        splitter = TimeSeriesSplitter(n_splits=3)
        splits = splitter.split(sample_dataframe)
        
        assert len(splits) == 3
    
    def test_split_with_none_dataframe_raises_error(self):
        """Test that None DataFrame raises error."""
        splitter = TimeSeriesSplitter()
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.split(None)
    
    def test_split_with_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises error."""
        splitter = TimeSeriesSplitter()
        empty_df = pd.DataFrame()
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.split(empty_df)
    
    def test_split_temporal_ordering(self, sample_dataframe):
        """Test that splits maintain temporal ordering."""
        splitter = TimeSeriesSplitter(n_splits=3)
        splits = splitter.split(sample_dataframe)
        
        for train_idx, test_idx in splits:
            # Training indices should be before test indices
            assert train_idx[-1] < test_idx[0]
    
    def test_split_no_overlap(self, sample_dataframe):
        """Test that train and test sets don't overlap."""
        splitter = TimeSeriesSplitter(n_splits=3)
        splits = splitter.split(sample_dataframe)
        
        for train_idx, test_idx in splits:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_split_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        splitter = TimeSeriesSplitter(n_splits=10)
        small_df = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.split(small_df)
    
    def test_split_indices_are_sequential(self, sample_dataframe):
        """Test that split indices are sequential."""
        splitter = TimeSeriesSplitter(n_splits=3)
        splits = splitter.split(sample_dataframe)
        
        for train_idx, test_idx in splits:
            # Check that indices are sequential
            assert np.array_equal(train_idx, np.arange(train_idx[0], train_idx[-1] + 1))
            assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))


class TestGetTrainTestSplit:
    """Tests for get_train_test_split method."""
    
    def test_get_train_test_split_basic(self, sample_dataframe):
        """Test basic train-test split."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.get_train_test_split(
            sample_dataframe
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert y_train is None
        assert y_test is None
    
    def test_get_train_test_split_with_target(self, sample_dataframe, sample_target):
        """Test train-test split with target variable."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.get_train_test_split(
            sample_dataframe,
            y=sample_target
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_get_train_test_split_temporal_ordering(self, sample_dataframe):
        """Test that train-test split maintains temporal ordering."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        X_train, X_test, _, _ = splitter.get_train_test_split(sample_dataframe)
        
        # Last training timestamp should be before first test timestamp
        assert X_train['timestamp'].max() < X_test['timestamp'].min()
    
    def test_get_train_test_split_custom_test_size(self, sample_dataframe):
        """Test train-test split with custom test size."""
        splitter = TimeSeriesSplitter(test_size=0.3)
        X_train, X_test, _, _ = splitter.get_train_test_split(sample_dataframe)
        
        assert len(X_train) == 70
        assert len(X_test) == 30
    
    def test_get_train_test_split_override_test_size(self, sample_dataframe):
        """Test overriding test_size in method call."""
        splitter = TimeSeriesSplitter(test_size=0.2)
        X_train, X_test, _, _ = splitter.get_train_test_split(
            sample_dataframe,
            test_size=0.3
        )
        
        assert len(X_train) == 70
        assert len(X_test) == 30
    
    def test_get_train_test_split_with_none_dataframe_raises_error(self):
        """Test that None DataFrame raises error."""
        splitter = TimeSeriesSplitter()
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.get_train_test_split(None)
    
    def test_get_train_test_split_with_invalid_test_size_raises_error(
        self,
        sample_dataframe
    ):
        """Test that invalid test_size raises error."""
        splitter = TimeSeriesSplitter()
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.get_train_test_split(sample_dataframe, test_size=0)
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.get_train_test_split(sample_dataframe, test_size=1)


class TestValidateNoLeakage:
    """Tests for validate_no_leakage method."""
    
    def test_validate_no_leakage_valid_split(self, sample_dataframe):
        """Test validation of valid split with no leakage."""
        splitter = TimeSeriesSplitter(n_splits=3)
        splits = splitter.split(sample_dataframe)
        
        train_idx, test_idx = splits[0]
        
        is_valid = splitter.validate_no_leakage(
            sample_dataframe,
            train_idx,
            test_idx
        )
        
        assert is_valid is True
    
    def test_validate_no_leakage_with_leakage(self, sample_dataframe):
        """Test detection of data leakage."""
        splitter = TimeSeriesSplitter()
        
        # Create indices with leakage (test before train)
        train_idx = np.arange(50, 100)
        test_idx = np.arange(0, 50)
        
        is_valid = splitter.validate_no_leakage(
            sample_dataframe,
            train_idx,
            test_idx
        )
        
        assert is_valid is False
    
    def test_validate_no_leakage_missing_timestamp_column_raises_error(self):
        """Test that missing timestamp column raises error."""
        splitter = TimeSeriesSplitter()
        
        df = pd.DataFrame({'value': [1, 2, 3]})
        train_idx = np.array([0, 1])
        test_idx = np.array([2])
        
        with pytest.raises(TimeSeriesSplitterError):
            splitter.validate_no_leakage(df, train_idx, test_idx)
    
    def test_validate_no_leakage_custom_timestamp_column(self, sample_dataframe):
        """Test validation with custom timestamp column name."""
        splitter = TimeSeriesSplitter()
        
        # Rename timestamp column
        df = sample_dataframe.rename(columns={'timestamp': 'time'})
        
        splits = splitter.split(df)
        train_idx, test_idx = splits[0]
        
        is_valid = splitter.validate_no_leakage(
            df,
            train_idx,
            test_idx,
            timestamp_col='time'
        )
        
        assert is_valid is True


class TestGetNSplits:
    """Tests for get_n_splits method."""
    
    def test_get_n_splits(self):
        """Test get_n_splits method."""
        splitter = TimeSeriesSplitter(n_splits=5)
        assert splitter.get_n_splits() == 5
    
    def test_get_n_splits_default(self):
        """Test get_n_splits with default value."""
        splitter = TimeSeriesSplitter()
        assert splitter.get_n_splits() == 3


class TestTimeSeriesSplitterIntegration:
    """Integration tests for TimeSeriesSplitter."""
    
    def test_multiple_splits_no_overlap(self, sample_dataframe):
        """Test that multiple splits don't overlap."""
        splitter = TimeSeriesSplitter(n_splits=5)
        splits = splitter.split(sample_dataframe)
        
        all_train_indices = set()
        all_test_indices = set()
        
        for train_idx, test_idx in splits:
            # Check no overlap within split
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Check no overlap with previous splits
            assert len(all_train_indices & set(test_idx)) == 0
            
            all_train_indices.update(train_idx)
            all_test_indices.update(test_idx)
    
    def test_split_consistency(self, sample_dataframe):
        """Test that split is consistent across calls."""
        splitter = TimeSeriesSplitter(n_splits=3)
        
        splits1 = splitter.split(sample_dataframe)
        splits2 = splitter.split(sample_dataframe)
        
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
