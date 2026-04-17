import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAnalyzerError(Exception):
    """Custom exception for FeatureAnalyzer errors."""
    pass


class FeatureAnalyzer:
    
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'aqi'):
        
        if df is None or len(df) == 0:
            raise FeatureAnalyzerError("DataFrame cannot be None or empty")
        
        if target_col not in df.columns:
            raise FeatureAnalyzerError(f"Target column '{target_col}' not found")
        
        self.df = df
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(
            f"FeatureAnalyzer initialized with {len(df)} records, "
            f"{len(self.numeric_cols)} numeric features"
        )
    
    def compute_feature_statistics(self) -> pd.DataFrame:
        
        logger.info("Computing feature statistics")
        
        stats = self.df[self.numeric_cols].describe().T
        
        # Add additional statistics
        stats['skewness'] = self.df[self.numeric_cols].skew()
        stats['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        stats['null_count'] = self.df[self.numeric_cols].isnull().sum()
        stats['null_percentage'] = (stats['null_count'] / len(self.df)) * 100
        
        logger.info(f"Computed statistics for {len(stats)} features")
        return stats
    
    def compute_correlation_matrix(
        self,
        include_target: bool = True
    ) -> pd.DataFrame:
        
        logger.info("Computing correlation matrix")
        
        if include_target:
            cols = self.numeric_cols
        else:
            cols = [c for c in self.numeric_cols if c != self.target_col]
        
        corr_matrix = self.df[cols].corr()
        
        logger.info(f"Computed correlation matrix ({len(corr_matrix)}x{len(corr_matrix)})")
        return corr_matrix
    
    def compute_target_correlations(self) -> pd.Series:
        
        
        logger.info(f"Computing correlations with target '{self.target_col}'")
        
        correlations = self.df[self.numeric_cols].corrwith(
            self.df[self.target_col]
        )
        
        # Sort by absolute correlation
        correlations = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )
        
        logger.info(f"Computed target correlations for {len(correlations)} features")
        return correlations
    
    def identify_feature_importance(self) -> Dict[str, float]:
        
        
        logger.info("Identifying feature importance indicators")
        
        correlations = self.compute_target_correlations()
        
        # Normalize to 0-1 range
        abs_corr = correlations.abs()
        importance = (abs_corr - abs_corr.min()) / (abs_corr.max() - abs_corr.min())
        
        importance_dict = importance.to_dict()
        
        logger.info(f"Identified importance for {len(importance_dict)} features")
        return importance_dict
    
    def get_feature_distributions(self) -> Dict[str, Dict]:
        
        
        logger.info("Computing feature distributions")
        
        distributions = {}
        
        for col in self.numeric_cols:
            distributions[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'q25': self.df[col].quantile(0.25),
                'q75': self.df[col].quantile(0.75),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis()
            }
        
        logger.info(f"Computed distributions for {len(distributions)} features")
        return distributions
    
    def identify_outliers(
        self,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, List[int]]:
        
        
        logger.info(f"Identifying outliers using {method} method")
        
        outliers = {}
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == 'zscore':
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = np.abs((self.df[col] - mean) / std)
                outlier_mask = z_scores > threshold
            
            else:
                raise FeatureAnalyzerError(f"Unknown method: {method}")
            
            outliers[col] = self.df[outlier_mask].index.tolist()
        
        total_outliers = sum(len(v) for v in outliers.values())
        logger.info(f"Identified {total_outliers} outliers across features")
        
        return outliers
    
    def check_feature_consistency_across_cities(
        self,
        city_col: str = 'city'
    ) -> Dict[str, Dict]:
        
        
        if city_col not in self.df.columns:
            raise FeatureAnalyzerError(f"City column '{city_col}' not found")
        
        logger.info("Checking feature consistency across cities")
        
        consistency = {}
        
        for col in self.numeric_cols:
            consistency[col] = {
                'global_mean': self.df[col].mean(),
                'global_std': self.df[col].std(),
                'by_city': {}
            }
            
            for city in self.df[city_col].unique():
                city_data = self.df[self.df[city_col] == city][col]
                consistency[col]['by_city'][city] = {
                    'mean': city_data.mean(),
                    'std': city_data.std(),
                    'count': len(city_data)
                }
        
        logger.info(f"Checked consistency for {len(consistency)} features")
        return consistency
    
    def generate_analysis_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict:
        
        
        logger.info("Generating comprehensive feature analysis report")
        
        report = {
            'summary': {
                'total_records': len(self.df),
                'total_features': len(self.numeric_cols),
                'target_column': self.target_col
            },
            'statistics': self.compute_feature_statistics().to_dict(),
            'correlations': {
                'target_correlations': self.compute_target_correlations().to_dict(),
                'correlation_matrix': self.compute_correlation_matrix().to_dict()
            },
            'importance': self.identify_feature_importance(),
            'distributions': self.get_feature_distributions(),
            'outliers': self.identify_outliers(),
        }
        
        if output_path:
            import json
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {output_path}")
        
        logger.info("Feature analysis report generated")
        return report
    
    def get_high_correlation_features(
        self,
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        
        
        logger.info(f"Identifying features with correlation > {threshold}")
        
        corr_matrix = self.compute_correlation_matrix(include_target=False)
        
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        logger.info(f"Found {len(high_corr_pairs)} high-correlation feature pairs")
        return high_corr_pairs
    
    def get_low_variance_features(
        self,
        threshold: float = 0.01
    ) -> List[str]:
        
        
        logger.info(f"Identifying features with variance < {threshold}")
        
        variances = self.df[self.numeric_cols].var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        logger.info(f"Found {len(low_var_features)} low-variance features")
        return low_var_features
