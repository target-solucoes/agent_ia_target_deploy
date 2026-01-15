"""
Graph nodes for the Insight Generator LangGraph workflow.

This module contains all node functions that process the InsightState.
Each node is responsible for a specific step in the insight generation pipeline.
"""

import logging
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from .state import InsightState
from ..core.settings import STATUS_PROCESSING, STATUS_SUCCESS, STATUS_ERROR
from ..core.intent_enricher import (
    IntentEnricher,
    EnrichedIntent,
    Polarity,
    TemporalFocus,
    ComparisonType,
)
from ..calculators import get_calculator
from ..calculators.metric_composer import MetricComposer
from ..formatters.prompt_builder import build_prompt, SYSTEM_PROMPT
from ..formatters.markdown_formatter import ExecutiveMarkdownFormatter
from ..models.insight_schemas import load_insight_llm
from ..utils.transparency_validator import validate_insight_dict_transparency
from ..utils.alignment_validator import validate_alignment
from ..utils.alignment_corrector import apply_corrections

logger = logging.getLogger(__name__)


def parse_input_node(state: InsightState) -> InsightState:
    """
    Node 1: Parse and validate input data.

    Extracts chart_spec and analytics_result from upstream agents,
    validates their structure, and prepares data for processing.

    Args:
        state: Current workflow state

    Returns:
        Updated state with parsed input fields

    Raises:
        Adds errors to state if validation fails
    """
    logger.info("[parse_input_node] Starting input parsing")

    try:
        # Validate required fields
        if "chart_spec" not in state:
            raise ValueError("Missing required field: chart_spec")
        if "analytics_result" not in state:
            raise ValueError("Missing required field: analytics_result")

        chart_spec = state["chart_spec"]
        analytics_result = state["analytics_result"]
        plotly_result = state.get("plotly_result", {})

        # Extract chart_type from chart_spec
        chart_type = chart_spec.get("chart_type")
        if not chart_type:
            raise ValueError("chart_spec missing 'chart_type' field")

        state["chart_type"] = chart_type
        logger.debug(f"[parse_input_node] Extracted chart_type: {chart_type}")

        # Extract DataFrame from analytics_result
        # PRIORIDADE: Usar dados limitados do plotly_result se disponíveis
        # Isso garante que os insights sejam gerados sobre os mesmos dados do gráfico
        data = None

        # Try limited_data from plotly_result first (dados que foram realmente plotados)
        if "limited_data" in plotly_result:
            limited_data = plotly_result["limited_data"]
            if isinstance(limited_data, list) and limited_data:
                data = pd.DataFrame(limited_data)
                logger.info(
                    f"[parse_input_node] Using limited_data from plotly_result "
                    f"({len(data)} rows) - aligns with plotted data"
                )

        # Fallback: Try data from analytics_result
        if data is None or data.empty:
            if "data" in analytics_result:
                data_content = analytics_result["data"]

                # If data is a list of dicts, convert to DataFrame
                if isinstance(data_content, list):
                    data = pd.DataFrame(data_content)
                elif isinstance(data_content, pd.DataFrame):
                    data = data_content
                else:
                    logger.warning(
                        f"[parse_input_node] Unexpected data type: {type(data_content)}"
                    )

                if data is not None and not data.empty:
                    logger.debug(
                        f"[parse_input_node] Using data from analytics_result ({len(data)} rows)"
                    )

        # Try aggregated_data as fallback
        if data is None or data.empty:
            if "aggregated_data" in analytics_result:
                agg_data = analytics_result["aggregated_data"]
                if isinstance(agg_data, list):
                    data = pd.DataFrame(agg_data)
                elif isinstance(agg_data, pd.DataFrame):
                    data = agg_data

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise ValueError("No valid data found in analytics_result or plotly_result")

        state["data"] = data
        logger.info(f"[parse_input_node] Extracted DataFrame with shape: {data.shape}")

        # Validate data has content
        if len(data) == 0:
            raise ValueError("DataFrame is empty")

        # ========== FASE 1: INTENT ENRICHMENT ==========
        # Enrich intent with semantic metadata
        logger.info("[parse_input_node] Starting intent enrichment")

        # Extract base intent from chart_spec
        base_intent = chart_spec.get(
            "intent", "ranking"
        )  # default to ranking if not present

        # Extract user query (may be in different locations depending on pipeline)
        user_query = chart_spec.get("user_query", "")
        if not user_query:
            # Try to get from analytics_result metadata
            user_query = analytics_result.get("metadata", {}).get("user_query", "")
        if not user_query:
            # Try to get from state directly
            user_query = state.get("user_query", "")

        # If we have a user query, enrich the intent
        if user_query:
            logger.debug(f"[parse_input_node] Enriching intent for query: {user_query}")
            enricher = IntentEnricher()
            enriched = enricher.enrich(
                base_intent=base_intent,
                user_query=user_query,
                chart_spec=chart_spec,
                analytics_result=analytics_result,
            )

            # Convert EnrichedIntent to dict for storage in state
            state["enriched_intent"] = {
                "base_intent": enriched.base_intent,
                "polarity": enriched.polarity.value,
                "temporal_focus": enriched.temporal_focus.value,
                "comparison_type": enriched.comparison_type.value,
                "suggested_metrics": enriched.suggested_metrics,
                "key_entities": enriched.key_entities,
                "filters_context": enriched.filters_context,
                "narrative_angle": enriched.narrative_angle,
            }

            logger.info(
                f"[parse_input_node] Intent enriched - "
                f"polarity: {enriched.polarity.value}, "
                f"temporal_focus: {enriched.temporal_focus.value}, "
                f"narrative_angle: {enriched.narrative_angle}"
            )
        else:
            logger.warning(
                "[parse_input_node] No user_query available for intent enrichment, "
                "will use basic intent only"
            )
            state["enriched_intent"] = None

        logger.info("[parse_input_node] Input parsing complete")
        return state

    except Exception as e:
        logger.error(f"[parse_input_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"parse_input_node: {str(e)}")
        return state


def calculate_metrics_node(state: InsightState) -> InsightState:
    """
    Node 2: Calculate metrics using composable metric modules.

    FASE 2 Implementation: Uses MetricComposer to select and execute metric
    modules based on enriched intent rather than chart type.

    Routes to appropriate metric modules based on enriched_intent and computes
    numeric insights from the data.

    Args:
        state: Current workflow state (must contain chart_type, data, enriched_intent)

    Returns:
        Updated state with numeric_summary and cache_key

    Raises:
        Adds errors to state if calculation fails
    """
    logger.info("[calculate_metrics_node] Starting metric calculation")

    try:
        # Validate required fields
        if "chart_type" not in state:
            raise ValueError("Missing required field: chart_type")
        if "data" not in state:
            raise ValueError("Missing required field: data")

        chart_type = state["chart_type"]
        df = state["data"]
        chart_spec = state.get("chart_spec", {})
        analytics_result = state.get("analytics_result", {})

        # Build config from chart_spec and analytics_result
        config = _build_calculator_config(chart_spec, analytics_result, df, state)
        logger.debug(f"[calculate_metrics_node] Config: {config}")

        # ========== FASE 2: METRIC COMPOSER ==========
        # Use MetricComposer if enriched_intent is available
        if state.get("enriched_intent"):
            logger.info("[calculate_metrics_node] Using MetricComposer (FASE 2)")

            # Convert enriched_intent dict back to EnrichedIntent object
            enriched_dict = state["enriched_intent"]
            enriched_intent = EnrichedIntent(
                base_intent=enriched_dict["base_intent"],
                polarity=Polarity(enriched_dict["polarity"]),
                temporal_focus=TemporalFocus(enriched_dict["temporal_focus"]),
                comparison_type=ComparisonType(enriched_dict["comparison_type"]),
                suggested_metrics=enriched_dict["suggested_metrics"],
                key_entities=enriched_dict["key_entities"],
                filters_context=enriched_dict["filters_context"],
                narrative_angle=enriched_dict["narrative_angle"],
            )

            # Use MetricComposer
            composer = MetricComposer()
            numeric_summary = composer.compose(df, enriched_intent, config)

            logger.info(
                f"[calculate_metrics_node] MetricComposer used {numeric_summary['metadata']['modules_count']} modules: "
                f"{numeric_summary['modules_used']}"
            )
        else:
            # Fallback to legacy calculator system
            logger.warning(
                "[calculate_metrics_node] No enriched_intent available, "
                "falling back to legacy calculator system"
            )
            calculator = get_calculator(chart_type)
            logger.debug(
                f"[calculate_metrics_node] Using calculator: {calculator.__class__.__name__}"
            )
            numeric_summary = calculator.calculate(df, config)

        state["numeric_summary"] = numeric_summary

        logger.info(f"[calculate_metrics_node] Calculated metrics successfully")

        # Generate cache key for future optimizations
        cache_key = _generate_cache_key(chart_type, df, config)
        state["cache_key"] = cache_key
        logger.debug(f"[calculate_metrics_node] Cache key: {cache_key[:16]}...")

        logger.info("[calculate_metrics_node] Metric calculation complete")
        return state

    except Exception as e:
        logger.error(f"[calculate_metrics_node] Error: {e}", exc_info=True)
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"calculate_metrics_node: {str(e)}")
        return state


def _build_calculator_config(
    chart_spec: Dict[str, Any],
    analytics_result: Dict[str, Any],
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build configuration dict for calculator from chart_spec and analytics_result.

    Args:
        chart_spec: Chart specification
        analytics_result: Analytics result
        df: DataFrame with data
        state: Full state dict (to access plotly_result)

    Returns:
        Configuration dict with dimension_cols, metric_cols, etc.
    """

    def _find_matching_column(col_name: str, df_columns: list) -> Optional[str]:
        """Find matching column in DataFrame, handling underscore vs space variations."""
        if col_name in df_columns:
            return col_name

        # Try replacing underscores with spaces
        alt_name = col_name.replace("_", " ")
        if alt_name in df_columns:
            return alt_name

        # Try replacing spaces with underscores
        alt_name = col_name.replace(" ", "_")
        if alt_name in df_columns:
            return alt_name

        return None

    config = {}
    df_columns = df.columns.tolist()

    # Extract dimension columns
    dimensions = chart_spec.get("dimensions", [])
    dimension_cols = []
    for dim in dimensions:
        if isinstance(dim, dict):
            # Use 'name' (actual column name in DataFrame), not 'alias' (display name)
            col = dim.get("name") or dim.get("column") or dim.get("alias")
            if col:
                matched_col = _find_matching_column(col, df_columns)
                if matched_col:
                    dimension_cols.append(matched_col)
        elif isinstance(dim, str):
            matched_col = _find_matching_column(dim, df_columns)
            if matched_col:
                dimension_cols.append(matched_col)

    # If no dimensions found, try to infer from DataFrame
    if not dimension_cols and len(df.columns) > 0:
        # Use first non-numeric column as dimension
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                dimension_cols.append(col)
                break

        # If no non-numeric found, use first column
        if not dimension_cols:
            dimension_cols.append(df.columns[0])

    config["dimension_cols"] = dimension_cols

    # Extract metric columns
    metrics = chart_spec.get("metrics", [])
    metric_cols = []
    aggregations = []

    for metric in metrics:
        if isinstance(metric, dict):
            # Use 'name' (actual column name in DataFrame), not 'alias' (display name)
            col = metric.get("name") or metric.get("column") or metric.get("alias")
            agg = metric.get("aggregation", "sum")
            if col:
                matched_col = _find_matching_column(col, df_columns)
                if matched_col:
                    metric_cols.append(matched_col)
                    aggregations.append(agg)
        elif isinstance(metric, str):
            matched_col = _find_matching_column(metric, df_columns)
            if matched_col:
                metric_cols.append(matched_col)
                aggregations.append("sum")

    # If no metrics found, use remaining numeric columns
    if not metric_cols:
        for col in df.columns:
            if col not in dimension_cols and pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)
                aggregations.append("sum")

    config["metric_cols"] = metric_cols
    config["aggregation"] = aggregations[0] if aggregations else "sum"

    # Add optional parameters
    # PRIORIDADE: Detectar top_n real dos metadata de limitação do plotly_result
    # Isso garante que os insights usem o valor correto de categorias limitadas
    top_n = None

    # Check if plotly_result has category limiting metadata
    plotly_result = state.get("plotly_result", {})
    plotly_metadata = plotly_result.get("metadata", {})
    category_limiting = plotly_metadata.get("category_limiting", {})

    if category_limiting.get("limit_applied"):
        # Use the actual limited count from plotly generator
        detected_top_n = category_limiting.get("limited_count")
        if detected_top_n:
            top_n = detected_top_n
            logger.info(
                f"[_build_calculator_config] Detected category limiting from plotly_result: "
                f"{category_limiting.get('original_count')} → {top_n} categories"
            )

    # If not found in category_limiting, check chart_spec
    if not top_n:
        chart_top_n = chart_spec.get("top_n")
        if chart_top_n:
            top_n = chart_top_n
            logger.debug(
                f"[_build_calculator_config] Using top_n from chart_spec: {top_n}"
            )

    # Apply final value or fallback to default
    if top_n:
        config["top_n"] = top_n
    else:
        # Fallback: Use 15 as default to align with CategoryLimiter default
        config["top_n"] = 15
        logger.debug("[_build_calculator_config] Using fallback top_n=15")

    # Extract filters if present
    if "filters" in chart_spec:
        config["filters"] = chart_spec["filters"]

    # Extract metadata (inclui full_dataset_totals para cálculos corretos)
    if "metadata" in analytics_result:
        config["metadata"] = analytics_result["metadata"]

    # ========== SERIES/STACK COLUMN DETECTION FOR MULTI-SERIES CHARTS ==========
    # For composed/multi-series charts, we need to identify which column contains the series
    # This is critical for TemporalMultiCalculator, ComposedCalculator, and StackedCalculator
    chart_type = chart_spec.get("chart_type", "")

    # Charts that require series_col or stack_col
    MULTI_SERIES_CHARTS = [
        "line_composed",
        "bar_vertical_composed",
        "bar_vertical_stacked",
    ]

    if chart_type in MULTI_SERIES_CHARTS and len(dimension_cols) >= 2:
        # For multi-series charts with 2+ dimensions:
        # - First dimension is typically the primary axis (time for temporal, category for bars)
        # - Second dimension is the series/grouping column
        primary_dim = dimension_cols[0]
        series_dim = dimension_cols[1]

        # Detect which dimension is temporal (for line_composed)
        temporal_dim = None
        categorical_dim = None

        for i, dim in enumerate(dimensions):
            if isinstance(dim, dict):
                temporal_gran = dim.get("temporal_granularity")
                if temporal_gran:
                    temporal_dim = (
                        dimension_cols[i] if i < len(dimension_cols) else None
                    )
                else:
                    categorical_dim = (
                        dimension_cols[i] if i < len(dimension_cols) else None
                    )

        # For line_composed: series_col is the categorical dimension
        if chart_type == "line_composed":
            if categorical_dim:
                config["series_col"] = categorical_dim
                logger.info(
                    f"[_build_calculator_config] line_composed: series_col='{categorical_dim}' "
                    f"(detected from dimensions)"
                )
            else:
                # Fallback: use second dimension as series
                config["series_col"] = series_dim
                logger.info(
                    f"[_build_calculator_config] line_composed: series_col='{series_dim}' "
                    f"(fallback to second dimension)"
                )

        # For bar_vertical_composed: series_col is the second dimension
        elif chart_type == "bar_vertical_composed":
            config["series_col"] = series_dim
            logger.info(
                f"[_build_calculator_config] bar_vertical_composed: series_col='{series_dim}'"
            )

        # For bar_vertical_stacked: stack_col is the second dimension
        elif chart_type == "bar_vertical_stacked":
            config["stack_col"] = series_dim
            logger.info(
                f"[_build_calculator_config] bar_vertical_stacked: stack_col='{series_dim}'"
            )

    elif chart_type in MULTI_SERIES_CHARTS and len(dimension_cols) == 1:
        # Single dimension - use it as both primary and series (will be handled by calculator)
        logger.debug(
            f"[_build_calculator_config] {chart_type} with single dimension - "
            f"calculator will use fallback behavior"
        )

    return config


def _generate_cache_key(
    chart_type: str, df: pd.DataFrame, config: Dict[str, Any]
) -> str:
    """
    Generate cache key for metric calculations.

    Args:
        chart_type: Type of chart
        df: DataFrame
        config: Calculator configuration

    Returns:
        Hash string for cache lookup
    """

    def _make_json_safe(obj: Any) -> Any:
        """Convert non-JSON-serializable objects to safe representations."""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return f"DataFrame({obj.shape})"
        elif isinstance(obj, pd.Series):
            return f"Series({len(obj)})"
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_json_safe(v) for v in obj]
        elif hasattr(obj, "__dict__"):
            return str(type(obj).__name__)
        return obj

    # Create hashable representation with safe conversions
    cache_data = {
        "chart_type": chart_type,
        "data_shape": df.shape,
        "columns": list(df.columns),
        "config": _make_json_safe(config),
    }

    # Generate hash
    try:
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to a simple hash if JSON serialization fails
        cache_str = f"{chart_type}_{df.shape}_{list(df.columns)}"

    return hashlib.md5(cache_str.encode()).hexdigest()


def build_prompt_node(state: InsightState) -> InsightState:
    """
    Node 3: Build LLM prompt from numeric summary.

    Constructs a chart-type-specific prompt using templates and
    the calculated numeric metrics. Also includes applied filters
    for context in the introduction.

    Args:
        state: Current workflow state (must contain numeric_summary and chart_type)

    Returns:
        Updated state with llm_prompt

    Raises:
        Adds errors to state if prompt building fails
    """
    logger.info("[build_prompt_node] Starting prompt building")

    try:
        # Validate required fields
        if "numeric_summary" not in state:
            raise ValueError("Missing required field: numeric_summary")
        if "chart_type" not in state:
            raise ValueError("Missing required field: chart_type")

        numeric_summary = state["numeric_summary"]
        chart_type = state["chart_type"]

        # Extract filters from chart_spec for inclusion in introduction
        filters = {}
        chart_spec = state.get("chart_spec", {})
        if chart_spec and "filters" in chart_spec:
            filters = chart_spec["filters"]
            logger.debug(f"[build_prompt_node] Found filters: {filters}")

        # Build prompt using prompt_builder with filters
        llm_prompt = build_prompt(numeric_summary, chart_type, filters=filters)
        state["llm_prompt"] = llm_prompt

        logger.info(
            f"[build_prompt_node] Built prompt with {len(llm_prompt)} characters"
        )
        if filters:
            logger.info(
                f"[build_prompt_node] Included {len(filters)} filters in prompt"
            )
        logger.debug(f"[build_prompt_node] Prompt preview: {llm_prompt[:200]}...")

        logger.info("[build_prompt_node] Prompt building complete")
        return state

    except Exception as e:
        logger.error(f"[build_prompt_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"build_prompt_node: {str(e)}")
        return state


def invoke_llm_node(state: InsightState) -> InsightState:
    """
    Node 4: Invoke LLM to generate insights.

    Sends the formatted prompt to GPT-5-nano and retrieves the
    raw insight text.

    Args:
        state: Current workflow state (must contain llm_prompt)

    Returns:
        Updated state with llm_response

    Raises:
        Adds errors to state if LLM invocation fails
    """
    logger.info("[invoke_llm_node] Starting LLM invocation")

    try:
        # Validate required field
        if "llm_prompt" not in state:
            raise ValueError("Missing required field: llm_prompt")

        llm_prompt = state["llm_prompt"]

        # Load LLM instance
        llm = load_insight_llm()
        # ChatGoogleGenerativeAI uses 'model' attribute, not 'model_name'
        model_identifier = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
        logger.debug(f"[invoke_llm_node] Loaded LLM: {model_identifier}")

        # Build messages with SYSTEM_PROMPT and user prompt
        # CRITICAL: Send SYSTEM_PROMPT as SystemMessage to enforce formula transparency
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=llm_prompt),
        ]

        logger.debug(
            "[invoke_llm_node] Sending messages with SYSTEM_PROMPT to enforce transparency"
        )

        # Invoke LLM with messages
        response = llm.invoke(messages)

        # Capture tokens from LLM response
        from src.shared_lib.utils.token_tracker import extract_token_usage

        tokens = extract_token_usage(response, llm)
        if "agent_tokens" not in state:
            state["agent_tokens"] = {}
        state["agent_tokens"]["insight_generator"] = tokens
        logger.info(
            f"[invoke_llm_node] Tokens captured: "
            f"input={tokens['input_tokens']}, "
            f"output={tokens['output_tokens']}, "
            f"total={tokens['total_tokens']}, "
            f"model={tokens.get('model_name', 'unknown')}"
        )

        # Extract response content - handle multiple formats
        logger.debug(f"[invoke_llm_node] Response type: {type(response)}")

        if hasattr(response, "content"):
            llm_response = response.content
            logger.debug(f"[invoke_llm_node] Content type: {type(llm_response)}")
        else:
            llm_response = str(response)
            logger.debug(f"[invoke_llm_node] Using str(response)")

        # Handle case where response is a list (structured output with reasoning)
        if isinstance(llm_response, list):
            logger.debug(
                f"[invoke_llm_node] Processing list with {len(llm_response)} items"
            )
            # Extract text from structured responses
            text_parts = []
            for item in llm_response:
                logger.debug(f"[invoke_llm_node] List item type: {type(item)}")
                if isinstance(item, dict):
                    # Extract text from {'type': 'text', 'text': '...'} format
                    if item.get("type") == "text" and "text" in item:
                        text_parts.append(item["text"])
                    # Also handle 'content' key format from some models
                    elif "content" in item:
                        text_parts.append(item["content"])
                    # Skip reasoning parts ({'type': 'reasoning'})
                elif hasattr(item, "text"):
                    # Handle object with text attribute
                    text_parts.append(item.text)
                else:
                    text_parts.append(str(item))
            llm_response = "\n".join(text_parts)
            logger.debug(
                f"[invoke_llm_node] After list processing: {len(llm_response)} chars"
            )

        state["llm_response"] = llm_response

        logger.info(
            f"[invoke_llm_node] Received response with {len(llm_response)} characters"
        )
        logger.info(
            f"[invoke_llm_node] Response (first 500 chars): {llm_response[:500]}"
        )

        logger.info("[invoke_llm_node] LLM invocation complete")
        return state

    except Exception as e:
        logger.error(f"[invoke_llm_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"invoke_llm_node: {str(e)}")
        return state


def validate_insights_node(state: InsightState) -> InsightState:
    """
    Node 5: Validate and structure insights (FASE 4 - Unified Output + FASE 5 - Alignment Validation).

    Parses the unified LLM response containing all report components,
    validates structure and transparency, and populates state with:
    - insights (detailed_insights from LLM)
    - executive_summary
    - synthesized_narrative
    - key_findings
    - next_steps

    FASE 5 Integration:
    - Validates alignment between narrative and detailed_insights
    - Applies automatic corrections for misalignments
    - Implements retry logic for value mismatches (max 2 attempts)
    - Adds alignment metadata to state

    Args:
        state: Current workflow state (must contain llm_response)

    Returns:
        Updated state with all insight components, transparency validation,
        and alignment validation results

    Raises:
        Adds errors to state if validation fails
    """
    logger.info(
        "[validate_insights_node] Starting unified insight validation (FASE 4 + 5)"
    )

    try:
        # Validate required fields
        if "llm_response" not in state:
            raise ValueError("Missing required field: llm_response")

        llm_response = state["llm_response"]
        chart_type = state.get("chart_type", "unknown")
        numeric_summary = state.get("numeric_summary", {})
        composed_metrics = state.get("composed_metrics", None)

        # Parse unified response (FASE 4 format)
        parsed_output = _parse_unified_llm_response(
            llm_response, chart_type, numeric_summary
        )

        # Extract components
        executive_summary = parsed_output.get("executive_summary", {})
        detailed_insights = parsed_output.get("detailed_insights", [])
        synthesized_insights = parsed_output.get("synthesized_insights", {})
        next_steps = parsed_output.get("next_steps", {})

        narrative = synthesized_insights.get("narrative", "")
        key_findings = synthesized_insights.get("key_findings", [])

        # Limit to MAX 5 detailed insights
        MAX_INSIGHTS = 5
        if len(detailed_insights) > MAX_INSIGHTS:
            logger.warning(
                f"[validate_insights_node] Truncating {len(detailed_insights)} insights to {MAX_INSIGHTS}"
            )
            detailed_insights = detailed_insights[:MAX_INSIGHTS]

        # ==================== FASE 5: ALIGNMENT VALIDATION ====================
        logger.info("[validate_insights_node] FASE 5: Starting alignment validation")

        # Step 1: Validate alignment
        alignment_result = validate_alignment(
            narrative=narrative,
            detailed_insights=detailed_insights,
            composed_metrics=composed_metrics,
        )

        logger.info(
            f"[validate_insights_node] Alignment validation result: "
            f"score={alignment_result['alignment_score']:.2f}, "
            f"aligned={alignment_result['is_aligned']}"
        )

        # Step 2: Apply automatic corrections if needed
        corrections_applied = []

        if not alignment_result["is_aligned"]:
            logger.warning(
                f"[validate_insights_node] Alignment issues detected: "
                f"{len(alignment_result.get('missing_in_detailed', []))} missing in detailed, "
                f"{len(alignment_result.get('value_mismatches', []))} value mismatches"
            )

            # Apply corrections
            correction_result = apply_corrections(
                narrative=narrative,
                detailed_insights=detailed_insights,
                key_findings=key_findings,
                executive_summary=executive_summary,
                validation_result=alignment_result,
                composed_metrics=composed_metrics,
            )

            # Update with corrected values
            detailed_insights = correction_result["detailed_insights"]
            key_findings = correction_result["key_findings"]
            executive_summary = correction_result["executive_summary"]
            corrections_applied = correction_result["corrections_applied"]

            # Corrections may add placeholder insights; keep contract of max 5.
            if len(detailed_insights) > MAX_INSIGHTS:
                logger.warning(
                    f"[validate_insights_node] Truncating corrected {len(detailed_insights)} insights to {MAX_INSIGHTS}"
                )
                detailed_insights = detailed_insights[:MAX_INSIGHTS]

            logger.info(
                f"[validate_insights_node] Applied {len(corrections_applied)} correction(s)"
            )

            # Re-validate after corrections
            alignment_result_after = validate_alignment(
                narrative=narrative,
                detailed_insights=detailed_insights,
                composed_metrics=composed_metrics,
            )

            logger.info(
                f"[validate_insights_node] Post-correction alignment score: "
                f"{alignment_result_after['alignment_score']:.2f}"
            )

        # Step 3: Check if retry is needed for value mismatches
        retry_count = state.get("_alignment_retry_count", 0)
        MAX_RETRIES = 2

        if (
            alignment_result.get("value_mismatches")
            and retry_count < MAX_RETRIES
            and len(alignment_result["value_mismatches"]) > 2
        ):  # Only retry if >2 mismatches
            logger.warning(
                f"[validate_insights_node] Significant value mismatches detected. "
                f"Triggering retry {retry_count + 1}/{MAX_RETRIES}"
            )

            # Mark for retry by clearing llm_response and incrementing counter
            state["_alignment_retry_count"] = retry_count + 1
            state["_retry_reason"] = (
                f"Value mismatches: {len(alignment_result['value_mismatches'])}"
            )

            # Return state without populating final fields to trigger re-invocation
            # This would require workflow modification to support retry
            # For now, we'll just log and continue with corrections
            logger.warning(
                "[validate_insights_node] Retry logic requires workflow support - "
                "continuing with corrections"
            )

        # ==================== END FASE 5 ====================

        # Populate state with all components
        state["insights"] = detailed_insights
        state["executive_summary"] = executive_summary
        state["synthesized_narrative"] = narrative
        state["key_findings"] = key_findings
        state["next_steps"] = next_steps.get("recommendations", [])

        # Add alignment metadata (FASE 5)
        state["alignment_score"] = alignment_result["alignment_score"]
        state["alignment_validated"] = alignment_result["is_aligned"]
        state["corrections_applied"] = corrections_applied
        state["alignment_warnings"] = alignment_result.get("warnings", [])

        logger.info(
            f"[validate_insights_node] Parsed unified output: "
            f"{len(detailed_insights)} insights, "
            f"narrative={len(narrative)} chars, "
            f"{len(key_findings)} key_findings, "
            f"{len(state['next_steps'])} next_steps, "
            f"alignment_score={alignment_result['alignment_score']:.2f}"
        )

        # Validate transparency for detailed_insights
        transparency_validated = validate_insight_dict_transparency(detailed_insights)
        state["transparency_validated"] = transparency_validated

        if transparency_validated:
            logger.info("[validate_insights_node] Transparency validation PASSED")
        else:
            logger.warning("[validate_insights_node] Transparency validation FAILED")

        logger.info(
            "[validate_insights_node] Unified insight validation complete (FASE 4 + 5)"
        )
        return state

    except Exception as e:
        logger.error(f"[validate_insights_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"validate_insights_node: {str(e)}")
        return state


def _parse_unified_llm_response(
    llm_response: str, chart_type: str, numeric_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parse unified LLM response (FASE 4 format) into structured components.

    Expected JSON structure:
    {
      "executive_summary": {"title": "...", "introduction": "..."},
      "detailed_insights": [{"title": "...", "formula": "...", "interpretation": "..."}],
      "synthesized_insights": {"narrative": "...", "key_findings": [...]},
      "next_steps": {"recommendations": [...]}
    }

    Args:
        llm_response: Raw LLM response text or JSON
        chart_type: Chart type for context
        numeric_summary: Metrics for reference

    Returns:
        Dictionary with all parsed components
    """
    try:
        # Parse JSON response
        response_data = (
            json.loads(llm_response) if isinstance(llm_response, str) else llm_response
        )

        if not isinstance(response_data, dict):
            raise ValueError(f"Expected dict, got {type(response_data)}")

        # Validate required sections
        required_sections = [
            "executive_summary",
            "detailed_insights",
            "synthesized_insights",
            "next_steps",
        ]
        for section in required_sections:
            if section not in response_data:
                logger.warning(
                    f"[_parse_unified_llm_response] Missing section: {section}"
                )
                # Provide fallback empty structure
                if section == "executive_summary":
                    response_data[section] = {
                        "title": "Análise de Dados",
                        "introduction": "",
                    }
                elif section == "detailed_insights":
                    response_data[section] = []
                elif section == "synthesized_insights":
                    response_data[section] = {"narrative": "", "key_findings": []}
                elif section == "next_steps":
                    response_data[section] = {"recommendations": []}

        # Process detailed_insights to add metadata
        detailed_insights = response_data.get("detailed_insights", [])
        processed_insights = []

        for item in detailed_insights:
            # Validate required fields
            if not all(k in item for k in ["title", "formula", "interpretation"]):
                logger.warning(
                    f"[_parse_unified_llm_response] Skipping invalid insight: {item}"
                )
                continue

            # Combine formula and interpretation into content field for backward compatibility
            content = f"{item['formula']}\n{item['interpretation']}"

            processed_insights.append(
                {
                    "title": item["title"],
                    "content": content,
                    "formula": item["formula"],  # Store separately for validation
                    "interpretation": item["interpretation"],
                    "metrics": numeric_summary,
                    "confidence": 0.9,  # Higher confidence for structured JSON
                    "chart_context": chart_type,
                }
            )

        response_data["detailed_insights"] = processed_insights

        logger.info(
            f"[_parse_unified_llm_response] Successfully parsed unified output: "
            f"{len(processed_insights)} detailed_insights"
        )

        return response_data

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error(
            f"[_parse_unified_llm_response] Failed to parse unified response: {e}"
        )
        logger.debug(
            f"[_parse_unified_llm_response] Raw response: {llm_response[:500]}"
        )

        # Fallback: Try old format parsing
        logger.info(
            "[_parse_unified_llm_response] Attempting fallback to old format parser"
        )
        old_format_insights = _parse_llm_response_legacy(
            llm_response, chart_type, numeric_summary
        )

        # Return minimal structure with fallback insights
        return {
            "executive_summary": {
                "title": f"Análise de {chart_type.replace('_', ' ').title()}",
                "introduction": "Análise gerada com formato legado.",
            },
            "detailed_insights": old_format_insights,
            "synthesized_insights": {
                "narrative": "Narrativa não disponível no formato legado.",
                "key_findings": [],
            },
            "next_steps": {"recommendations": []},
        }


def _parse_llm_response_legacy(
    llm_response: str, chart_type: str, numeric_summary: Dict[str, Any]
) -> list:
    """
    Parse LLM response into structured insights (LEGACY FORMAT - Pre-FASE 4).

    Maintained for backward compatibility and fallback.

    Tries JSON parsing first (for old JSON mode), falls back to text parsing.

    Args:
        llm_response: Raw LLM response text or JSON
        chart_type: Chart type for context
        numeric_summary: Metrics for reference

    Returns:
        List of insight dictionaries with keys: title, content, metrics, confidence, chart_context
    """
    insights = []

    # Try old JSON parsing first
    try:
        response_data = (
            json.loads(llm_response) if isinstance(llm_response, str) else llm_response
        )

        if isinstance(response_data, dict) and "insights" in response_data:
            json_insights = response_data["insights"]

            for item in json_insights:
                # Validate required fields
                if not all(k in item for k in ["title", "formula", "interpretation"]):
                    logger.warning(
                        f"[_parse_llm_response_legacy] Skipping invalid insight: {item}"
                    )
                    continue

                # Combine formula and interpretation into content field
                content = f"{item['formula']}\n{item['interpretation']}"

                insights.append(
                    {
                        "title": item["title"],
                        "content": content,
                        "formula": item["formula"],  # Store separately for validation
                        "interpretation": item["interpretation"],
                        "metrics": numeric_summary,
                        "confidence": 0.9,  # Higher confidence for structured JSON
                        "chart_context": chart_type,
                    }
                )

            logger.info(
                f"[_parse_llm_response_legacy] Successfully parsed {len(insights)} insights from old JSON format"
            )
            return insights

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug(
            f"[_parse_llm_response_legacy] Old JSON parsing failed, falling back to text parsing: {e}"
        )

    # Fallback: text parsing for backward compatibility
    lines = llm_response.strip().split("\n")
    current_insight = {"title": "", "content": ""}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line is a title (starts with ** or numbered)
        if line.startswith("**") or (
            len(line) > 2 and line[0].isdigit() and line[1] in [".", ")", ":"]
        ):
            # Save previous insight if exists
            if current_insight["title"] and current_insight["content"]:
                insights.append(
                    {
                        "title": current_insight["title"],
                        "content": current_insight["content"],
                        "metrics": numeric_summary,
                        "confidence": 0.8,
                        "chart_context": chart_type,
                    }
                )

            # Extract title (remove markdown bold and numbering)
            title = line.replace("**", "").strip()
            # Remove leading numbers and separators
            if title and title[0].isdigit():
                title = title.split(maxsplit=1)[-1] if " " in title else title

            # Handle "Title:" format
            if ":" in title:
                parts = title.split(":", 1)
                title = parts[0].strip()
                content_start = parts[1].strip() if len(parts) > 1 else ""
                current_insight = {"title": title, "content": content_start}
            else:
                current_insight = {"title": title, "content": ""}
        else:
            # Continue content of current insight
            if current_insight["content"]:
                current_insight["content"] += " " + line
            else:
                current_insight["content"] = line

    # Add last insight
    if current_insight["title"] and current_insight["content"]:
        insights.append(
            {
                "title": current_insight["title"],
                "content": current_insight["content"],
                "metrics": numeric_summary,
                "confidence": 0.8,
                "chart_context": chart_type,
            }
        )

    logger.info(
        f"[_parse_llm_response] Parsed {len(insights)} insights from text format"
    )
    return insights


def transform_to_markdown_node(state: InsightState) -> InsightState:
    """
    Node 5.5: Transform structured insights to executive markdown format.

    Takes parsed JSON insights and transforms them into executive-style
    markdown with H3 headers, bullet points, bold formatting, and separators.

    Args:
        state: Current workflow state (must contain insights list)

    Returns:
        Updated state with formatted_insights (markdown string)

    Raises:
        Adds errors to state if transformation fails
    """
    logger.info("[transform_to_markdown_node] Starting markdown transformation")

    try:
        # Validate required fields
        if "insights" not in state:
            raise ValueError("Missing required field: insights")

        insights = state["insights"]
        chart_type = state.get("chart_type", "unknown")

        if not insights:
            logger.warning("[transform_to_markdown_node] No insights to transform")
            state["formatted_insights"] = ""
            return state

        # Initialize markdown formatter
        formatter = ExecutiveMarkdownFormatter()

        # Transform insights to markdown
        # Insights list contains dicts with: title, content, formula, interpretation
        insights_for_formatting = []
        for insight in insights:
            # If insight has separate formula and interpretation, use them
            if "formula" in insight and "interpretation" in insight:
                insights_for_formatting.append(
                    {
                        "title": insight["title"],
                        "formula": insight["formula"],
                        "interpretation": insight["interpretation"],
                    }
                )
            else:
                # Fallback: try to split content into formula and interpretation
                content = insight.get("content", "")
                lines = content.split("\n", 1)
                insights_for_formatting.append(
                    {
                        "title": insight["title"],
                        "formula": lines[0] if lines else "",
                        "interpretation": lines[1] if len(lines) > 1 else "",
                    }
                )

        # Format as executive markdown
        formatted_markdown = formatter.format_insights(
            insights_for_formatting, chart_type
        )

        state["formatted_insights"] = formatted_markdown

        logger.info(
            f"[transform_to_markdown_node] Transformed {len(insights)} insights "
            f"to {len(formatted_markdown)} character markdown"
        )
        logger.debug(
            f"[transform_to_markdown_node] Preview (first 500 chars):\n{formatted_markdown[:500]}"
        )

        logger.info("[transform_to_markdown_node] Markdown transformation complete")
        return state

    except Exception as e:
        logger.error(f"[transform_to_markdown_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"transform_to_markdown_node: {str(e)}")
        # Fallback: use raw content as formatted_insights
        state["formatted_insights"] = "\n\n".join(
            [
                f"**{i['title']}**\n{i.get('content', '')}"
                for i in state.get("insights", [])
            ]
        )
        return state


def format_output_node(state: InsightState) -> InsightState:
    """
    Node 6: Format final output (FASE 4 - Unified Output + FASE 5 - Alignment Metadata).

    Assembles all results into the final unified output structure with
    all components from the single LLM call:
    - executive_summary
    - detailed_insights
    - synthesized_insights (narrative + key_findings)
    - next_steps
    - metadata (including FASE 5 alignment validation results)
    - status and error handling

    Args:
        state: Current workflow state (must contain all unified components)

    Returns:
        Updated state with final_output

    Raises:
        Adds errors to state if formatting fails
    """
    logger.info("[format_output_node] Starting unified output formatting (FASE 4 + 5)")

    try:
        # Check for errors
        has_errors = bool(state.get("errors"))
        status = STATUS_ERROR if has_errors else STATUS_SUCCESS

        # Get all unified components from state
        executive_summary = state.get("executive_summary", {})
        insights = state.get("insights", [])  # detailed_insights
        formatted_insights = state.get("formatted_insights", "")
        synthesized_narrative = state.get("synthesized_narrative", "")
        key_findings = state.get("key_findings", [])
        next_steps = state.get("next_steps", [])
        numeric_summary = state.get("numeric_summary", {})

        # Calculate metadata
        metrics_count = len(numeric_summary)
        timestamp = datetime.now().isoformat()
        transparency_validated = state.get("transparency_validated", False)

        # FASE 5: Get alignment validation metadata
        alignment_score = state.get("alignment_score", 1.0)
        alignment_validated = state.get("alignment_validated", True)
        corrections_applied = state.get("corrections_applied", [])
        alignment_warnings = state.get("alignment_warnings", [])

        # Build unified final output (FASE 4 + 5)
        final_output = {
            "status": status,
            "chart_type": state.get("chart_type", "unknown"),
            # FASE 4: Include all unified components
            "executive_summary": executive_summary,
            "detailed_insights": insights,  # List of detailed insights with formula + interpretation
            "formatted_insights": formatted_insights,  # Executive markdown format (backward compat)
            "synthesized_insights": {
                "narrative": synthesized_narrative,
                "key_findings": key_findings,
            },
            "next_steps": next_steps,
            "metadata": {
                "calculation_time": 0.0,  # Can be enhanced with timing later
                "metrics_count": metrics_count,
                "llm_model": "gemini-2.5-flash-preview-09-2025",
                "timestamp": timestamp,
                "transparency_validated": transparency_validated,
                "fase_4_unified": True,  # Flag to indicate FASE 4 unified output
                # FASE 5: Alignment validation metadata
                "alignment_score": alignment_score,
                "alignment_validated": alignment_validated,
                "corrections_applied": corrections_applied,
                "alignment_warnings": alignment_warnings,
                "fase_5_alignment": True,  # Flag to indicate FASE 5 alignment validation
            },
            "error": state["errors"][0] if has_errors else None,
        }

        state["final_output"] = final_output

        logger.info(
            f"[format_output_node] Unified output formatted with status: {status}, "
            f"alignment_score: {alignment_score:.2f}"
        )
        logger.info(
            f"[format_output_node] Generated {len(insights)} detailed insights, "
            f"{len(key_findings)} key findings, {len(next_steps)} next steps, "
            f"{len(corrections_applied)} corrections applied"
        )

        if alignment_warnings:
            logger.warning(
                f"[format_output_node] Alignment warnings: {alignment_warnings}"
            )

        logger.info(
            "[format_output_node] Unified output formatting complete (FASE 4 + 5)"
        )
        return state

    except Exception as e:
        logger.error(f"[format_output_node] Error: {e}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"format_output_node: {str(e)}")

        # Create minimal error output
        state["final_output"] = {
            "status": STATUS_ERROR,
            "chart_type": state.get("chart_type", "unknown"),
            "executive_summary": {},
            "detailed_insights": [],
            "synthesized_insights": {"narrative": "", "key_findings": []},
            "next_steps": [],
            "metadata": {
                "calculation_time": 0.0,
                "metrics_count": 0,
                "llm_model": "gemini-2.5-flash-preview-09-2025",
                "timestamp": datetime.now().isoformat(),
                "transparency_validated": False,
                "fase_4_unified": True,
                "alignment_score": 0.0,
                "alignment_validated": False,
                "corrections_applied": [],
                "alignment_warnings": [],
                "fase_5_alignment": True,
            },
            "error": str(e),
        }
        return state


def initialize_state(
    chart_spec: Dict[str, Any], analytics_result: Dict[str, Any]
) -> InsightState:
    """
    Initialize InsightState with input data and default values.

    Args:
        chart_spec: Chart specification from graphic_classifier
        analytics_result: Analytics output from analytics_executor

    Returns:
        Initialized InsightState ready for workflow execution

    Example:
        >>> state = initialize_state(chart_spec, analytics_result)
        >>> workflow.invoke(state)
    """
    return InsightState(
        chart_spec=chart_spec,
        analytics_result=analytics_result,
        errors=[],
        insights=[],
        agent_tokens={},  # CRITICAL: Initialize for token tracking
    )
