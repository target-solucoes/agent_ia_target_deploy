"""
Data Loader module for Phase 4 - Analytics Executor Agent.

This module provides functionality for loading data from various sources
(parquet, csv) with caching support for improved performance.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from functools import lru_cache
import pandas as pd

from src.shared_lib.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader with caching support for multiple file formats.
    
    Supports loading from:
    - Parquet files (.parquet)
    - CSV files (.csv)
    
    Features:
    - LRU caching for repeated loads
    - File existence validation
    - Comprehensive error handling
    - Detailed logging
    
    Example:
        >>> loader = DataLoader(cache_size=5)
        >>> df = loader.load("data/datasets/sales.parquet")
        >>> print(df.shape)
        (1000, 10)
    """
    
    def __init__(self, cache_size: int = 5):
        """
        Initialize the DataLoader.
        
        Args:
            cache_size: Maximum number of datasets to cache in memory.
                       Default is 5. Set to 0 to disable caching.
        """
        self.cache_size = cache_size
        self._setup_cache()
        logger.info(f"DataLoader initialized with cache_size={cache_size}")
    
    def _setup_cache(self) -> None:
        """Configure the LRU cache for the load method."""
        if self.cache_size > 0:
            self._cached_load = lru_cache(maxsize=self.cache_size)(self._load_file)
        else:
            self._cached_load = self._load_file
    
    def load(self, data_source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a file source with caching.
        
        This method validates the file exists, determines the file format,
        and loads the data into a pandas DataFrame. Results are cached
        for improved performance on repeated loads.
        
        Args:
            data_source: Path to the data file (parquet, csv)
        
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If the data source file does not exist
            ValueError: If the file format is not supported
            Exception: For any other errors during loading
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load("data/sales.parquet")
            >>> print(f"Loaded {len(df)} rows")
            Loaded 1000 rows
        """
        path = Path(data_source)
        
        # Validate file exists
        if not path.exists():
            error_msg = f"Data source not found: {data_source}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate file format
        if path.suffix not in ['.parquet', '.csv']:
            error_msg = f"Unsupported file format: {path.suffix}. Supported formats: .parquet, .csv"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load with caching
        logger.info(f"Loading data from: {data_source}")
        try:
            df = self._cached_load(str(path.resolve()))
            logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns from {data_source}")
            return df
        except Exception as e:
            error_msg = f"Failed to load data from {data_source}: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Internal method to load file (used by cache).

        Args:
            file_path: Absolute path to the file

        Returns:
            pd.DataFrame: Loaded data with temporal columns extracted

        Raises:
            ValueError: If file format is not supported
            Exception: For any errors during file reading
        """
        path = Path(file_path)

        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
                logger.debug(f"Loaded parquet file: {file_path}")
            elif path.suffix == '.csv':
                df = pd.read_csv(path)
                logger.debug(f"Loaded CSV file: {file_path}")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Validate DataFrame is not empty
            if df.empty:
                logger.warning(f"Loaded DataFrame is empty: {file_path}")

            # Extract temporal columns automatically
            df = self._extract_temporal_columns(df)

            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {file_path}")
            raise ValueError(f"Data file is empty: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def _extract_temporal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal columns (Ano, Mes, Dia, Trimestre) from Data column.

        This method automatically detects temporal columns (Data, Date, DateTime, etc.)
        and extracts derived columns for easier querying by Year, Month, Day, Quarter.

        Args:
            df: DataFrame to process

        Returns:
            pd.DataFrame: DataFrame with additional temporal columns

        Note:
            - Only extracts if temporal column is found
            - Preserves original Data column
            - Adds: Ano (year), Mes (month), Dia (day), Trimestre (quarter)
        """
        # Priority order for temporal column detection
        temporal_col_names = ['Data', 'data', 'DATE', 'date', 'DateTime', 'datetime', 'Timestamp', 'timestamp']

        temporal_col = None
        for col_name in temporal_col_names:
            if col_name in df.columns:
                temporal_col = col_name
                break

        # If no explicit temporal column found, scan for datetime dtypes
        if temporal_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    temporal_col = col
                    break

        if temporal_col is None:
            logger.debug("No temporal column found for extraction")
            return df

        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[temporal_col]):
                df[temporal_col] = pd.to_datetime(df[temporal_col])
                logger.debug(f"Converted {temporal_col} to datetime type")

            # Extract temporal components
            df['Ano'] = df[temporal_col].dt.year
            df['Mes'] = df[temporal_col].dt.month
            df['Dia'] = df[temporal_col].dt.day
            df['Trimestre'] = df[temporal_col].dt.quarter

            logger.info(
                f"Extracted temporal columns from '{temporal_col}': "
                f"Ano, Mes, Dia, Trimestre"
            )

        except Exception as e:
            logger.warning(f"Failed to extract temporal columns from {temporal_col}: {str(e)}")

        return df

    def clear_cache(self) -> None:
        """
        Clear the internal cache.
        
        This can be useful to free memory or force reload of updated files.
        
        Example:
            >>> loader = DataLoader()
            >>> df1 = loader.load("data.parquet")  # Loads from file
            >>> df2 = loader.load("data.parquet")  # Loads from cache
            >>> loader.clear_cache()
            >>> df3 = loader.load("data.parquet")  # Loads from file again
        """
        if hasattr(self._cached_load, 'cache_clear'):
            self._cached_load.cache_clear()
            logger.info("Cache cleared")
    
    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.
        
        Returns:
            dict: Cache statistics including hits, misses, size, and maxsize
        
        Example:
            >>> loader = DataLoader(cache_size=5)
            >>> df = loader.load("data.parquet")
            >>> info = loader.get_cache_info()
            >>> print(f"Cache hits: {info['hits']}")
            Cache hits: 0
        """
        if hasattr(self._cached_load, 'cache_info'):
            cache_info = self._cached_load.cache_info()
            return {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'size': cache_info.currsize,
                'maxsize': cache_info.maxsize
            }
        return {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'maxsize': 0
        }
    
    def load_multiple(self, data_sources: list[Union[str, Path]]) -> dict[str, pd.DataFrame]:
        """
        Load multiple data sources at once.
        
        Args:
            data_sources: List of paths to data files
        
        Returns:
            dict: Dictionary mapping file paths to DataFrames
        
        Raises:
            Exception: If any file fails to load, all successful loads are returned
                      and the error is logged
        
        Example:
            >>> loader = DataLoader()
            >>> datasets = loader.load_multiple([
            ...     "data/sales.parquet",
            ...     "data/customers.csv"
            ... ])
            >>> print(f"Loaded {len(datasets)} datasets")
            Loaded 2 datasets
        """
        results = {}
        errors = []
        
        for source in data_sources:
            try:
                results[str(source)] = self.load(source)
            except Exception as e:
                error_msg = f"Failed to load {source}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if errors:
            logger.warning(f"Failed to load {len(errors)}/{len(data_sources)} files")
        
        return results



