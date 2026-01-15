"""
DatasetMaxDateCache - Cache LRU para data máxima de datasets.

Este módulo fornece cache eficiente para informações de data máxima de datasets,
evitando múltiplas queries custosas ao banco de dados.

Estratégia:
- Usa DuckDB para queries diretas no Parquet (zero data loading)
- Cache LRU para até 10 datasets
- Extrai múltiplas granularidades em uma única query
"""

import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


@dataclass
class MaxDateInfo:
    """
    Informações sobre a data máxima de um dataset.

    Attributes:
        max_date: datetime da data máxima encontrada
        max_year: Ano máximo (int)
        max_month: Mês máximo (1-12)
        max_quarter: Trimestre máximo (1-4)
        max_semester: Semestre máximo (1-2)
        max_bimester: Bimestre máximo (1-6)
        dataset_path: Caminho do dataset consultado
    """
    max_date: datetime
    max_year: int
    max_month: int
    max_quarter: int
    max_semester: int
    max_bimester: int


class DatasetMaxDateCache:
    """
    Cache LRU para data máxima de datasets.

    Esta classe fornece acesso eficiente à data máxima de um dataset,
    utilizando DuckDB para queries rápidas sem carregar dados em memória.

    Features:
    - Cache LRU com até 10 datasets
    - Query direta via DuckDB (zero data loading)
    - Cálculo automático de granularidades (mês, trimestre, semestre)
    - Thread-safe para uso em aplicações multi-threaded

    Example:
        >>> cache = DatasetMaxDateCache()
        >>> info = cache.get_max_date("data/datasets/dataset.parquet")
        >>> print(info.max_date)  # datetime(2016, 6, 30)
        >>> print(info.max_month)  # 6 (junho)
    """

    def __init__(self):
        """Initialize cache."""
        self.logger = logging.getLogger(__name__)
        logger.info("[DatasetMaxDateCache] Initialized")

    @lru_cache(maxsize=10)
    def get_max_date(self, dataset_path: str) -> "MaxDateInfo":
        """
        Busca e cacheia informações sobre a data máxima no dataset.

        Args:
            dataset_path: Caminho para o arquivo parquet/csv do dataset

        Returns:
            MaxDateInfo com informações sobre data máxima

        Raises:
            ValueError: Se dataset não existir ou não tiver coluna Data
            RuntimeError: Se falhar ao carregar dataset
        """
        logger.info(f"[DatasetMaxDateCache] Fetching max date from {dataset_path}")

        try:
            # Approach 1: DuckDB (zero data loading, fastest)
            import duckdb

            # BUGFIX: Extrair ano/mês/trimestre da DATA MÁXIMA, não o máximo dos meses
            # MAX(MONTH(Data)) != MONTH(MAX(Data)) se dados não estão em ordem cronológica
            result = duckdb.query(
                f"""
                WITH max_date_row AS (
                    SELECT MAX(Data) as max_date FROM '{dataset_path}'
                )
                SELECT
                    max_date,
                    YEAR(max_date) as max_year,
                    MONTH(max_date) as max_month,
                    QUARTER(max_date) as max_quarter
                FROM max_date_row
                """
            ).to_df()

            max_date = pd.to_datetime(result.loc[0, "max_date"])
            max_year = int(result.loc[0, "max_year"])
            max_month = int(result.loc[0, "max_month"])
            max_quarter = int(result.loc[0, "max_quarter"])
            max_semester = 1 if max_month <= 6 else 2
            max_bimester = (max_month - 1) // 2 + 1

            logger.info(
                f"[DatasetMaxDateCache] Fetched max date from dataset: "
                f"{max_date.date()} (Year: {max_year}, Month: {max_month})"
            )

            return MaxDateInfo(
                max_date=max_date,
                max_year=max_year,
                max_month=max_month,
                max_quarter=max_quarter,
                max_semester=max_semester,
                max_bimester=max_bimester,
            )

        except Exception as e:
            logger.error(f"[DatasetMaxDateCache] Error fetching max date: {str(e)}")
            raise ValueError(f"Failed to fetch max date from dataset: {str(e)}")


# Global cache instance
_cache_instance = DatasetMaxDateCache()


def get_max_date_info(dataset_path: str) -> MaxDateInfo:
    """
    Função conveniente para obter informações de data máxima do dataset.

    Args:
        dataset_path: Caminho para o arquivo do dataset.

    Returns:
        MaxDateInfo com informações da data máxima.
    """
    return _cache_instance.get_max_date(dataset_path)
