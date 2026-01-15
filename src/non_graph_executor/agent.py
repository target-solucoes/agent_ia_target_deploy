"""
Main agent orchestrator for non-graph query processing.

This module implements the NonGraphExecutorAgent class, which provides
a high-level interface for processing non-graphical queries and
returning structured JSON responses.
"""

import logging
import time
from typing import Dict, Any, Optional

from src.non_graph_executor.models.schemas import NonGraphOutput
from src.non_graph_executor.core.settings import (
    DATA_PATH,
    ALIAS_PATH,
    validate_settings,
)
from src.non_graph_executor.models.llm_loader import load_llm
from src.non_graph_executor.tools.metadata_cache import MetadataCache
from src.non_graph_executor.tools.query_executor import QueryExecutor
from src.non_graph_executor.tools.query_classifier import QueryClassifier
from src.non_graph_executor.tools.conversational import ConversationalHandler
from src.non_graph_executor.utils.output_formatter import OutputFormatter
from src.shared_lib.utils.logger import setup_logging
from src.shared_lib.utils.performance_monitor import PerformanceMonitor
from src.graphic_classifier.tools.alias_mapper import AliasMapper

logger = logging.getLogger(__name__)


class NonGraphExecutorAgent:
    """
    Main agent for processing non-graphical queries.

    This agent orchestrates the entire workflow for non-graph queries:
    1. Accepts natural language queries
    2. Classifies query type (metadata, aggregation, lookup, etc.)
    3. Executes appropriate query operations
    4. Returns structured JSON output

    The agent is designed to be stateless and thread-safe, with all
    configuration loaded at initialization time.

    Responsabilidades:
    - Classificar tipo de query (metadata, aggregation, lookup, etc.)
    - Executar consultas otimizadas via DuckDB
    - Retornar JSON estruturado final compatível com frontend
    - Tracking de performance e estatísticas

    Attributes:
        data_source: Caminho para o dataset
        alias_path: Caminho para arquivo alias.yaml
        _query_count: Contador de queries processadas
        _error_count: Contador de erros

    Example:
        >>> agent = NonGraphExecutorAgent()
        >>> state = {
        ...     "query": "quantas linhas tem?",
        ...     "filter_final": {},
        ...     "data_source": "data/dataset.parquet"
        ... }
        >>> result = agent.execute(state)
        >>> print(result['non_graph_output']['status'])
        'success'
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        alias_path: Optional[str] = None,
        setup_logs: bool = True,
    ):
        """
        Initialize the NonGraphExecutorAgent.

        Args:
            data_path: Path to dataset file (uses default if None)
            alias_path: Path to alias.yaml file (uses default if None)
            setup_logs: Whether to configure logging system

        Raises:
            ValueError: If settings validation fails
            FileNotFoundError: If required files not found
            PermissionError: If files don't have read permissions
        """
        # Setup logging
        if setup_logs:
            setup_logging()

        logger.info("Initializing NonGraphExecutorAgent")

        # Validate environment settings
        try:
            validate_settings()
            logger.info("Settings validation successful")
        except Exception as e:
            logger.error(f"Settings validation failed: {str(e)}")
            raise

        # Store configuration
        self.data_source = data_path or DATA_PATH
        self.alias_path = alias_path or ALIAS_PATH

        # Initialize statistics counters
        self._query_count = 0
        self._error_count = 0

        # Initialize all components (Phases 2-4)
        logger.info("Initializing agent components...")

        try:
            # Phase 2.1: Metadata Cache with lazy loading
            self.metadata_cache = MetadataCache(self.data_source)
            logger.info("[OK] MetadataCache initialized")

            # NOVO: Obter nomes exatos das colunas do dataset
            dataset_columns = self.metadata_cache.get_exact_column_names()
            logger.info(
                f"[OK] Retrieved {len(dataset_columns)} exact column names from dataset"
            )

            # Phase 3: Query Classifier with AliasMapper (moved earlier for QueryExecutor)
            self.alias_mapper = AliasMapper(
                alias_path=self.alias_path, dataset_columns=dataset_columns
            )
            logger.info("[OK] AliasMapper initialized with case-insensitive matching")

            classifier_llm = load_llm(
                temperature=0.3,
                max_output_tokens=800,  # Classification + parameter extraction
            )
            self.query_classifier = QueryClassifier(
                llm=classifier_llm, alias_mapper=self.alias_mapper
            )
            logger.info("[OK] QueryClassifier initialized")

            # Phase 2.2: Query Executor (now with alias_mapper for type-aware filtering)
            self.query_executor = QueryExecutor(
                self.data_source, self.metadata_cache, self.alias_mapper
            )
            logger.info("[OK] QueryExecutor initialized")

            # Phase 2.3: Conversational Handler with LLM
            conversational_llm = load_llm(
                temperature=0.3,
                max_output_tokens=500,  # Short conversational responses
            )
            self.conversational_handler = ConversationalHandler(conversational_llm)
            logger.info("[OK] ConversationalHandler initialized")

            # Phase 4: Output Formatter with LLM for summaries
            formatter_llm = load_llm(
                temperature=0.3,
                max_output_tokens=800,  # Concise summaries
            )
            self.output_formatter = OutputFormatter(formatter_llm)
            logger.info("[OK] OutputFormatter initialized")

        except Exception as e:
            logger.error(f"Failed to initialize agent components: {str(e)}")
            raise

        logger.info(
            f"NonGraphExecutorAgent initialized successfully: "
            f"data_source={self.data_source}"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute non-graph query processing.

        Método principal de execução que orquestra todo o workflow:
        1. Extrai query e filtros do state
        2. Classifica tipo de query
        3. Delega para ferramenta apropriada
        4. Formata output final
        5. Atualiza estatísticas

        Args:
            state: Pipeline state contendo:
                - query: Query em linguagem natural
                - filter_final: Filtros aplicados (do filter_agent)
                - data_source: Caminho do dataset
                - available_columns: Colunas disponíveis (opcional)

        Returns:
            Dict contendo:
                - non_graph_output: Output estruturado (NonGraphOutput)
                - execution_time: Tempo total de execução

        Raises:
            Exception: Erros são capturados e retornados no output

        Example:
            >>> state = {
            ...     "query": "qual a média de vendas?",
            ...     "filter_final": {"Ano": 2015},
            ...     "data_source": "data/dataset.parquet"
            ... }
            >>> result = agent.execute(state)
        """
        # Extract query and context from state
        query = state.get("query", "")
        filters = state.get("filter_final", {})

        logger.info(f"[NonGraphExecutor] Processing query: '{query}'")
        if filters:
            logger.info(f"[NonGraphExecutor] Applied filters: {filters}")

        # Initialize performance monitor
        perf = PerformanceMonitor()

        # Initialize token accumulator
        from src.formatter_agent.utils.token_accumulator import TokenAccumulator

        token_accumulator = TokenAccumulator()
        state["_token_accumulator"] = token_accumulator

        try:
            # Increment query counter
            self._query_count += 1

            # Start total timing
            perf.start_times["total"] = time.perf_counter()

            # ========================================================================
            # STEP 1: Check for conversational queries first (highest priority)
            # ========================================================================
            if self.conversational_handler.is_conversational(query):
                logger.info("[NonGraphExecutor] Detected conversational query")

                perf.start_times["llm"] = time.perf_counter()
                response = self.conversational_handler.generate_response(
                    query, token_accumulator
                )
                perf.timings["llm"] = time.perf_counter() - perf.start_times["llm"]

                # Format conversational output
                perf.timings["total"] = time.perf_counter() - perf.start_times["total"]
                metrics = self._extract_performance_metrics(perf)

                formatted_output = self.output_formatter.format_conversational(response)
                formatted_output["performance_metrics"] = metrics
                formatted_output = self._attach_token_metadata(
                    formatted_output, token_accumulator
                )

                # Aggregate tokens from conversational handler
                if "agent_tokens" not in state:
                    state["agent_tokens"] = {}
                state["agent_tokens"]["non_graph_executor"] = (
                    token_accumulator.get_totals()
                )

                logger.info(
                    f"[NonGraphExecutor] Total tokens: {token_accumulator.get_totals()}"
                )

                return {
                    "non_graph_output": formatted_output,
                    "execution_time": perf.timings.get("total", 0.0),
                }

            # ========================================================================
            # STEP 2: Classify query type and extract parameters
            # ========================================================================
            logger.info("[NonGraphExecutor] Classifying query...")
            perf.start_times["classification"] = time.perf_counter()

            # Get available columns from metadata cache
            global_metadata = self.metadata_cache.get_global_metadata()
            available_columns = global_metadata.get("columns", [])

            # Create state for classifier with available columns
            classifier_state = {
                "query": query,
                "available_columns": available_columns,
                "data_source": self.data_source,
                "_token_accumulator": token_accumulator,  # Pass token accumulator
            }

            classification = self.query_classifier.classify(
                query=query, state=classifier_state
            )

            perf.timings["classification"] = (
                time.perf_counter() - perf.start_times["classification"]
            )
            logger.info(
                f"[NonGraphExecutor] Query classified as: {classification.query_type} "
                f"(confidence: {classification.confidence:.2f})"
            )

            # ========================================================================
            # STEP 3: Execute query based on type
            # ========================================================================
            logger.info("[NonGraphExecutor] Executing query...")
            perf.start_times["execution"] = time.perf_counter()

            # Track LLM usage if classification required it
            if classification.requires_llm:
                if "llm" not in perf.start_times:
                    perf.start_times["llm"] = perf.start_times["classification"]
                    perf.timings["llm"] = perf.timings["classification"]

            result_data, result_metadata = self._execute_by_type(
                classification=classification, filters=filters, perf=perf
            )

            # Enrich metadata with filtered_dataset_row_count
            result_metadata = self._enrich_metadata_with_dataset_row_count(
                result_metadata, filters
            )

            perf.timings["execution"] = (
                time.perf_counter() - perf.start_times["execution"]
            )
            logger.info(
                f"[NonGraphExecutor] Execution completed in "
                f"{perf.timings['execution']:.3f}s"
            )

            # ========================================================================
            # STEP 4: Format final output
            # ========================================================================
            perf.timings["total"] = time.perf_counter() - perf.start_times["total"]
            metrics = self._extract_performance_metrics(perf)

            formatted_output = self.output_formatter.format(
                query_type=classification.query_type,
                data=result_data,
                metadata=result_metadata,
                performance=metrics,
                query=query,
                filters=filters if filters else None,
                error=None,
                token_accumulator=token_accumulator,
            )
            formatted_output = self._attach_token_metadata(
                formatted_output, token_accumulator
            )

            logger.info(
                f"[NonGraphExecutor] Query completed successfully in "
                f"{perf.timings['total']:.3f}s"
            )

            # Aggregate tokens from all LLM calls
            if "agent_tokens" not in state:
                state["agent_tokens"] = {}
            state["agent_tokens"]["non_graph_executor"] = token_accumulator.get_totals()

            logger.info(
                f"[NonGraphExecutor] Total tokens: {token_accumulator.get_totals()}"
            )

            return {
                "non_graph_output": formatted_output,
                "execution_time": perf.timings.get("total", 0.0),
            }

        except Exception as e:
            logger.error(
                f"[NonGraphExecutor] Error executing query: {str(e)}", exc_info=True
            )
            self._error_count += 1

            # Track timing even on error
            if "total" in perf.start_times:
                perf.timings["total"] = time.perf_counter() - perf.start_times["total"]

            return self._handle_error(e, perf, query, filters)

    def _enrich_metadata_with_dataset_row_count(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add filtered_dataset_row_count to metadata.

        This represents the total number of rows in the filtered dataset,
        not the number of rows in the query result.

        Args:
            metadata: Original metadata dict
            filters: Filters applied to the dataset

        Returns:
            Enhanced metadata with filtered_dataset_row_count
        """
        try:
            if filters:
                meta = self.metadata_cache.get_filtered_metadata(filters)
                dataset_row_count = meta["shape"]["rows"]
            else:
                meta = self.metadata_cache.get_global_metadata()
                dataset_row_count = meta["shape"]["rows"]

            metadata["filtered_dataset_row_count"] = dataset_row_count
        except Exception as e:
            logger.warning(f"Failed to add filtered_dataset_row_count: {e}")

        return metadata

    def _execute_by_type(
        self, classification, filters: Dict[str, Any], perf: PerformanceMonitor
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute query based on classified type.

        Delega execução para ferramenta apropriada baseada no tipo:
        - metadata → MetadataCache + QueryExecutor
        - aggregation → QueryExecutor.compute_simple_aggregation
        - lookup → QueryExecutor.lookup_record
        - textual → QueryExecutor (text_search ou list_unique_values)
        - statistical → QueryExecutor.get_descriptive_stats
        - tabular → QueryExecutor.get_tabular_data

        Args:
            classification: QueryTypeClassification com tipo e parâmetros
            filters: Filtros a aplicar na query
            perf: PerformanceMonitor para tracking

        Returns:
            Tuple (data, metadata):
                - data: Lista de dicts com resultados
                - metadata: Dict com metadados da execução

        Raises:
            ValueError: Se tipo de query não suportado ou parâmetros inválidos
            Exception: Erros de execução das queries
        """
        query_type = classification.query_type
        params = classification.parameters

        logger.debug(f"[NonGraphExecutor] Executing {query_type} with params: {params}")

        # ========================================================================
        # METADATA
        # ========================================================================
        if query_type == "metadata":
            subtype = params.get("subtype", "unknown")

            if subtype == "row_count":
                # Row count (with or without filters)
                if filters:
                    meta = self.metadata_cache.get_filtered_metadata(filters)
                    count = meta["row_count"]
                else:
                    meta = self.metadata_cache.get_global_metadata()
                    count = meta["shape"]["rows"]

                data = [{"row_count": count}]
                metadata = {
                    "row_count": 1,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "cache_hit": True,
                    "filters_applied": filters if filters else {},
                }

            elif subtype == "column_list":
                # List all columns
                meta = self.metadata_cache.get_global_metadata()
                columns = meta.get("columns", [])
                data = [{"columns": columns}]
                metadata = {
                    "row_count": 1,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "cache_hit": True,
                }

            elif subtype == "sample_rows":
                # Get sample rows
                n = params.get("n", 5)
                rows = self.query_executor.get_sample_rows(n=n, filters=filters)
                data = rows
                metadata = {
                    "row_count": len(rows),
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "filters_applied": filters if filters else {},
                }

            elif subtype == "unique_count":
                # Count unique values in column
                column = params.get("column")
                if not column:
                    raise ValueError("Column name required for unique_count")

                count = self.query_executor.count_unique_values(column, filters)
                data = [{"column": column, "unique_count": count}]
                metadata = {
                    "row_count": 0,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "filters_applied": filters if filters else {},
                }

            else:
                # Default: return only essential info (row_count + column_count)
                meta = self.metadata_cache.get_global_metadata()
                data = [
                    {
                        "row_count": meta["shape"]["rows"],
                        "column_count": meta["shape"]["cols"],
                        "columns": meta.get("columns", []),
                    }
                ]
                metadata = {
                    "row_count": 1,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "cache_hit": True,
                }

        # ========================================================================
        # AGGREGATION
        # ========================================================================
        elif query_type == "aggregation":
            column = params.get("column")
            agg_type = params.get("aggregation", "sum")
            distinct = params.get("distinct", False)

            if not column:
                raise ValueError("Column name required for aggregation")

            result = self.query_executor.compute_simple_aggregation(
                column=column, aggregation=agg_type, filters=filters, distinct=distinct
            )

            data = [{f"{agg_type}_{column}": result}]
            metadata = {
                "row_count": 1,
                "execution_time": perf.timings.get("execution", 0.0),
                "engine": "DuckDB",
                "aggregation": agg_type,
                "column": column,
                "distinct": distinct,
                "filters_applied": filters if filters else {},
            }

        # ========================================================================
        # LOOKUP
        # ========================================================================
        elif query_type == "lookup":
            # Back-compat: older param names were id_column/id_value/columns.
            lookup_column = params.get("lookup_column") or params.get("id_column")
            lookup_value = (
                params.get("lookup_value")
                if "lookup_value" in params
                else params.get("id_value")
            )
            return_columns = params.get("return_columns") or params.get("columns")

            if not lookup_column or lookup_value is None:
                raise ValueError(
                    "Both lookup_column and lookup_value required for lookup"
                )

            record = self.query_executor.lookup_record(
                lookup_column=lookup_column,
                lookup_value=lookup_value,
                return_columns=return_columns,
                filters=filters,
            )

            if record:
                data = [record]
                metadata = {
                    "row_count": 1,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "filters_applied": filters if filters else {},
                }
            else:
                data = []
                metadata = {
                    "row_count": 0,
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "message": f"No record found with {lookup_column}={lookup_value}",
                    "filters_applied": filters if filters else {},
                }

        # ========================================================================
        # TEXTUAL
        # ========================================================================
        elif query_type == "textual":
            subtype = params.get("subtype", "search")

            if subtype == "list_unique":
                # List unique values in column
                column = params.get("column")
                if not column:
                    raise ValueError("Column name required for list_unique")

                values = self.query_executor.list_unique_values(column, filters)
                data = [{"column": column, "unique_values": values}]
                metadata = {
                    "row_count": len(values),
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "filters_applied": filters if filters else {},
                }

            else:
                # Text search
                column = params.get("column")
                search_term = params.get("search_term")
                case_sensitive = params.get("case_sensitive", False)

                if not column or not search_term:
                    raise ValueError("Column and search_term required for text search")

                results = self.query_executor.text_search(
                    column=column,
                    search_term=search_term,
                    case_sensitive=case_sensitive,
                    filters=filters,
                )

                data = results
                metadata = {
                    "row_count": len(results),
                    "execution_time": perf.timings.get("execution", 0.0),
                    "engine": "DuckDB",
                    "search_term": search_term,
                    "filters_applied": filters if filters else {},
                }

        # ========================================================================
        # STATISTICAL
        # ========================================================================
        elif query_type == "statistical":
            column = params.get("column")

            if not column:
                raise ValueError("Column name required for statistical analysis")

            stats = self.query_executor.get_descriptive_stats(column, filters)
            data = [stats]
            metadata = {
                "row_count": 1,
                "execution_time": perf.timings.get("execution", 0.0),
                "engine": "DuckDB",
                "column": column,
                "filters_applied": filters if filters else {},
            }

        # ========================================================================
        # TABULAR
        # ========================================================================
        elif query_type == "tabular":
            limit = params.get("limit", 100)
            columns = params.get("columns")  # Optional: specific columns

            rows = self.query_executor.get_tabular_data(
                limit=limit, columns=columns, filters=filters
            )

            data = rows
            metadata = {
                "row_count": len(rows),
                "execution_time": perf.timings.get("execution", 0.0),
                "engine": "DuckDB",
                "limit": limit,
                "filters_applied": filters if filters else {},
            }

        # ========================================================================
        # UNKNOWN/UNSUPPORTED
        # ========================================================================
        else:
            raise ValueError(f"Unsupported query type: {query_type}")

        return data, metadata

    def _attach_token_metadata(
        self, output: Dict[str, Any], token_accumulator
    ) -> Dict[str, Any]:
        """Embed aggregate token usage into the non-graph output payload."""
        if token_accumulator is None:
            token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        else:
            token_totals = token_accumulator.get_totals()

        # Always expose total token usage and per-agent breakdown
        output["total_tokens"] = token_totals
        output["agent_tokens"] = {"non_graph_executor": token_totals}
        return output

    def _extract_performance_metrics(
        self, perf: PerformanceMonitor
    ) -> Dict[str, float]:
        """
        Extrai métricas de performance do PerformanceMonitor.

        Este método converte os timings internos do monitor para
        o formato esperado pelo NonGraphOutput.

        Args:
            perf: PerformanceMonitor instance com timings registrados

        Returns:
            Dict com métricas de performance:
                - total_time: Tempo total de execução
                - classification_time: Tempo de classificação (se aplicável)
                - execution_time: Tempo de execução da query (se aplicável)
                - llm_time: Tempo de chamadas LLM (se aplicável)
                - cache_hit: Boolean indicando cache hit (se aplicável)

        Example:
            >>> perf = PerformanceMonitor()
            >>> perf.timings["total"] = 0.287
            >>> perf.timings["classification"] = 0.012
            >>> metrics = agent._extract_performance_metrics(perf)
            >>> print(metrics["total_time"])
            0.287
        """
        metrics = {}

        # Extract all available timings
        if "total" in perf.timings:
            metrics["total_time"] = perf.timings["total"]
        if "classification" in perf.timings:
            metrics["classification_time"] = perf.timings["classification"]
        if "execution" in perf.timings:
            metrics["execution_time"] = perf.timings["execution"]
        if "llm" in perf.timings:
            metrics["llm_time"] = perf.timings["llm"]

        # Cache hit is tracked separately (will be implemented in Phase 2)
        # metrics["cache_hit"] = cache_was_hit

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Retorna estatísticas de uso do agente:
        - Total de queries processadas
        - Total de erros
        - Taxa de sucesso

        Returns:
            Dict com estatísticas:
                - query_count: Total de queries
                - error_count: Total de erros
                - success_rate: Taxa de sucesso (0.0 a 1.0)

        Example:
            >>> agent = NonGraphExecutorAgent()
            >>> # ... execute queries ...
            >>> stats = agent.get_statistics()
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
        """
        success_count = self._query_count - self._error_count
        success_rate = (
            success_count / self._query_count if self._query_count > 0 else 0.0
        )

        return {
            "query_count": self._query_count,
            "total_queries": self._query_count,  # Alias for compatibility
            "error_count": self._error_count,
            "success_count": success_count,
            "success_rate": success_rate,
        }

    def _handle_error(
        self,
        error: Exception,
        perf: Optional[PerformanceMonitor] = None,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle error and return structured error output.

        Tratamento centralizado de erros:
        - Loga o erro
        - Retorna output estruturado com informações do erro
        - Mantém consistência com schema NonGraphOutput
        - Inclui métricas de performance quando disponíveis
        - Inclui contexto da query e filtros para debugging

        Args:
            error: Exception que ocorreu
            perf: PerformanceMonitor instance (opcional)
            query: Query original do usuário (opcional)
            filters: Filtros aplicados (opcional)

        Returns:
            Dict com output de erro estruturado
        """
        error_type = type(error).__name__
        error_message = str(error)

        logger.error(
            f"NonGraphExecutorAgent error: {error_type}: {error_message}", exc_info=True
        )

        # Log additional context if available
        if query:
            logger.error(f"Failed query: {query}")
        if filters:
            logger.error(f"Applied filters: {filters}")

        # Extract performance metrics if available
        performance_metrics = {}
        execution_time = 0.0
        if perf is not None:
            performance_metrics = self._extract_performance_metrics(perf)
            execution_time = performance_metrics.get("total_time", 0.0)

        # Build error metadata with context
        error_metadata = {
            "error_occurred_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if query:
            error_metadata["failed_query"] = query
        if filters:
            error_metadata["applied_filters"] = filters

        return {
            "non_graph_output": NonGraphOutput(
                status="error",
                query_type="metadata",  # Default for error cases
                error={"type": error_type, "message": error_message},
                metadata=error_metadata,
                performance_metrics=performance_metrics,
            ).model_dump(),
            "execution_time": execution_time,
        }
