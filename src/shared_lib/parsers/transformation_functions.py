"""
Individual Transformation Functions for Chart Spec Pipeline

Este módulo contém funções de transformação independentes que podem
ser compostas em um pipeline para transformar ChartOutput specs.

Cada função é autônoma, testável e documentada.

Conforme especificado em planning_graphical_correction.md - Fase 3.3:
- Funções standalone (não métodos de classe)
- Assinaturas uniformes: (spec: Dict) -> Dict
- Sem dependências entre funções
- Facilmente testáveis isoladamente

Referência: planning_graphical_correction.md - Fase 3.3
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ========== Chart Type Requirements ==========

CHART_TYPE_REQUIREMENTS = {
    "bar_horizontal": {
        "min_metrics": 1,
        "min_dimensions": 1,
        "max_dimensions": 1,
        "requires_temporal": False,
        "default_aggregation": "sum",
        "description": "Rankings and top-N comparisons",
    },
    "bar_vertical": {
        "min_metrics": 1,
        "min_dimensions": 1,
        "max_dimensions": 1,
        "requires_temporal": False,
        "default_aggregation": "sum",
        "description": "Direct comparisons between categories",
    },
    "bar_vertical_composed": {
        "min_metrics": 1,
        "min_dimensions": 2,
        "max_dimensions": 2,
        "requires_temporal": False,
        "default_aggregation": "sum",
        "description": "Grouped comparisons across periods or conditions",
    },
    "bar_vertical_stacked": {
        "min_metrics": 1,
        "min_dimensions": 2,
        "max_dimensions": 2,
        "requires_temporal": False,
        "default_aggregation": "sum",
        "description": "Composition of subcategories within main categories",
    },
    "line": {
        "min_metrics": 1,
        "min_dimensions": 1,
        "max_dimensions": 1,
        "requires_temporal": True,
        "default_aggregation": "sum",
        "description": "Temporal trends and series",
    },
    "line_composed": {
        "min_metrics": 1,
        "min_dimensions": 2,
        "max_dimensions": 2,
        "requires_temporal": True,
        "default_aggregation": "sum",
        "description": "Multiple category trends over time",
    },
    "pie": {
        "min_metrics": 1,
        "min_dimensions": 1,
        "max_dimensions": 1,
        "requires_temporal": False,
        "default_aggregation": "sum",
        "description": "Proportional composition and participation",
    },
    "histogram": {
        "min_metrics": 1,
        "min_dimensions": 0,
        "max_dimensions": 0,
        "requires_temporal": False,
        "default_aggregation": "count",
        "description": "Distribution of numeric values",
    },
}


# ========== Column Classifications ==========

TEMPORAL_COLUMNS = [
    "Mes",
    "Ano",
    "Data",
    "Data_Venda",
    "Data_Pedido",
    "Trimestre",
    "Semestre",
    "Dia",
    "Semana",
]

DIMENSION_COLUMNS = [
    "UF_Cliente",
    "Des_Regiao_Vendedor",
    "Cod_Cliente",
    "Cod_Vendedor",
    "Des_Linha_Produto",
    "Des_Grupo_Produto",
    "Des_Familia_Produto",
    "Des_Segmento_Cliente",
    "Des_Linha_Produto",
]

METRIC_COLUMNS = [
    "Valor_Vendido",
    "Qtd_Vendida",
    "Peso_Vendido",
    "Preco_Unitario",
    "Custo_Produto",
]


# ========== Transformation 1: Infer Missing Metrics ==========


def infer_missing_metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garante que o spec tenha métricas suficientes para o tipo de gráfico.

    Se métricas estão faltando, tenta inferir de:
    1. parsed_entities (metric_hints)
    2. Palavras-chave da query
    3. Defaults do chart type

    Args:
        spec: Chart specification

    Returns:
        Spec com métricas inferidas (se necessário)
    """
    chart_type = spec.get("chart_type")

    if chart_type is None or chart_type == "null":
        return spec

    if chart_type not in CHART_TYPE_REQUIREMENTS:
        logger.warning(f"[infer_missing_metrics] Unknown chart type '{chart_type}'")
        return spec

    metrics = spec.get("metrics", [])
    requirements = CHART_TYPE_REQUIREMENTS[chart_type]
    min_metrics = requirements["min_metrics"]

    if len(metrics) >= min_metrics:
        logger.debug(
            f"[infer_missing_metrics] Sufficient metrics: {len(metrics)}/{min_metrics}"
        )
        return spec

    logger.warning(
        f"[infer_missing_metrics] Insufficient metrics for {chart_type}: "
        f"found {len(metrics)}, need {min_metrics}"
    )

    # Tentar inferir métrica
    inferred_metric = _infer_metric_from_context(spec)

    if inferred_metric:
        if not metrics:
            metrics = []

        metric_spec = {
            "name": inferred_metric,
            "aggregation": requirements["default_aggregation"],
            "alias": _prettify_label(inferred_metric),
        }

        metrics.append(metric_spec)
        spec["metrics"] = metrics

        logger.info(f"[infer_missing_metrics] Inferred metric: {inferred_metric}")
    else:
        logger.error(f"[infer_missing_metrics] Could not infer metric for {chart_type}")

    return spec


def _infer_metric_from_context(spec: Dict[str, Any]) -> Optional[str]:
    """Inferir métrica de parsed_entities, query, ou defaults."""
    parsed_entities = (
        spec.get("parsed_entities", {})
        if isinstance(spec.get("parsed_entities"), dict)
        else {}
    )

    # Prioridade 1: metric_hints
    metric_hints = parsed_entities.get("metric_hints", [])
    for hint in metric_hints:
        if hint in METRIC_COLUMNS:
            return hint

        # Tentar variações
        variations = [
            hint.capitalize(),
            hint.upper(),
            f"{hint.capitalize()}_Vendido",
            f"Valor_{hint.capitalize()}",
        ]
        for var in variations:
            if var in METRIC_COLUMNS:
                return var

    # Prioridade 2: Palavras-chave da query
    query = spec.get("query", "").lower()

    keyword_mappings = {
        "vendas": "Valor_Vendido",
        "venda": "Valor_Vendido",
        "faturamento": "Valor_Vendido",
        "receita": "Valor_Vendido",
        "quantidade": "Qtd_Vendida",
        "qtd": "Qtd_Vendida",
        "peso": "Peso_Vendido",
    }

    for keyword, metric in keyword_mappings.items():
        if keyword in query:
            return metric

    # Prioridade 3: Defaults por chart type
    chart_type = spec.get("chart_type")
    defaults = {
        "bar_horizontal": "Valor_Vendido",
        "bar_vertical": "Valor_Vendido",
        "pie": "Valor_Vendido",
        "line": "Valor_Vendido",
        "histogram": "Qtd_Vendida",
    }

    return defaults.get(chart_type, "Valor_Vendido")


# ========== Transformation 2: Infer Temporal Dimensions ==========


def infer_temporal_dimensions(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida e infere dimensões temporais para line charts.

    Garante que line/line_composed tenham pelo menos uma dimensão temporal.

    Args:
        spec: Chart specification

    Returns:
        Spec com dimensão temporal inferida (se necessário)
    """
    chart_type = spec.get("chart_type")

    if chart_type is None or chart_type == "null":
        return spec

    if chart_type not in CHART_TYPE_REQUIREMENTS:
        return spec

    requirements = CHART_TYPE_REQUIREMENTS[chart_type]

    if not requirements.get("requires_temporal"):
        logger.debug(
            "[infer_temporal_dimensions] Chart type does not require temporal dimension"
        )
        return spec

    dimensions = spec.get("dimensions", [])

    # Verificar se já tem dimensão temporal
    has_temporal = any(
        _is_temporal_column(d.get("name") if isinstance(d, dict) else d)
        for d in dimensions
    )

    if has_temporal:
        logger.debug("[infer_temporal_dimensions] Temporal dimension already present")
        return spec

    logger.warning(
        f"[infer_temporal_dimensions] {chart_type} requires temporal dimension but none found"
    )

    # Tentar inferir dimensão temporal
    temporal_dim = _infer_temporal_dimension_from_context(spec)

    if temporal_dim:
        dim_spec = {"name": temporal_dim, "alias": _prettify_label(temporal_dim)}

        # Inserir como primeira dimensão
        if not dimensions:
            dimensions = []
        dimensions.insert(0, dim_spec)

        spec["dimensions"] = dimensions
        logger.info(
            f"[infer_temporal_dimensions] Added temporal dimension: {temporal_dim}"
        )
    else:
        logger.error(
            f"[infer_temporal_dimensions] Could not infer temporal dimension for {chart_type}"
        )

    return spec


def _infer_temporal_dimension_from_context(spec: Dict[str, Any]) -> Optional[str]:
    """
    Inferir dimensão temporal ADEQUADA (não apenas detectar temporal em filter).

    IMPORTANTE: Diferenciar entre:
    - Filter temporal (range): ex: Data entre 2015-01-01 e 2015-12-31
    - Dimension temporal (agregação): ex: Mes (para mostrar evolução mensal)

    Esta função escolhe a dimensão de agregação apropriada baseada no contexto,
    não apenas retorna a primeira coluna temporal encontrada.

    Args:
        spec: Chart specification com filters, query, chart_type

    Returns:
        Nome da coluna temporal apropriada para usar como dimensão
    """
    filters = spec.get("filters", {})
    chart_type = spec.get("chart_type")
    query = spec.get("query", "").lower()

    # 1. Analisar filters para detectar range filters vs single value filters
    for filter_col, filter_value in filters.items():
        if not _is_temporal_column(filter_col):
            continue

        # Detectar se é range filter
        is_range = False
        if isinstance(filter_value, list) and len(filter_value) == 2:
            # Pode ser range filter - verificar se são datas/valores sequenciais
            is_range = True

        # Se é range filter em Data → dimensão deve ser Mes
        if is_range and filter_col == "Data":
            logger.info(
                "[_infer_temporal_dimension] Detected range filter on 'Data' → "
                "using 'Mes' as aggregation dimension"
            )
            return "Mes"

        # Se é range filter em Ano → dimensão deve ser Mes
        if is_range and filter_col == "Ano":
            logger.info(
                "[_infer_temporal_dimension] Detected range filter on 'Ano' → "
                "using 'Mes' as aggregation dimension"
            )
            return "Mes"

        # Se é valor único (não range):
        # Ano específico → mostrar por Mes
        if filter_col == "Ano" and not is_range:
            logger.info(
                "[_infer_temporal_dimension] Single year filter → "
                "using 'Mes' as aggregation dimension"
            )
            return "Mes"

        # Mes específico → mostrar por Data
        if filter_col == "Mes" and not is_range:
            logger.info(
                "[_infer_temporal_dimension] Single month filter → "
                "using 'Data' as aggregation dimension"
            )
            return "Data"

    # 2. Verificar query por keywords temporais
    temporal_keywords = {
        "mensal": "Mes",
        "mensalmente": "Mes",
        "mes": "Mes",
        "mês": "Mes",
        "mês a mês": "Mes",
        "mes a mes": "Mes",
        "por mês": "Mes",
        "por mes": "Mes",
        "histórico": "Mes",  # ADICIONAR: histórico geralmente implica mensal
        "historico": "Mes",
        "evolução": "Mes",  # ADICIONAR: evolução geralmente é mensal
        "evolucao": "Mes",
        "tendência": "Mes",  # ADICIONAR: tendência geralmente é mensal
        "tendencia": "Mes",
        "ano": "Ano",
        "anual": "Ano",
        "anualmente": "Ano",
        "ano a ano": "Ano",
        "por ano": "Ano",
        "data": "Data",
        "dia": "Data",
        "diário": "Data",
        "diariamente": "Data",
        "trimestre": "Trimestre",
        "trimestral": "Trimestre",
    }

    for keyword, column in temporal_keywords.items():
        if keyword in query:
            logger.info(
                f"[_infer_temporal_dimension] Detected keyword '{keyword}' in query → "
                f"using '{column}' as dimension"
            )
            return column

    # 3. Default para line charts: Mes (granularidade mais comum)
    if chart_type in ["line", "line_composed"]:
        logger.debug(
            "[_infer_temporal_dimension] Line chart without specific context → "
            "defaulting to 'Mes'"
        )
        return "Mes"

    # 4. Fallback geral: primeira coluna temporal disponível
    if TEMPORAL_COLUMNS:
        logger.debug(
            f"[_infer_temporal_dimension] No specific context → "
            f"defaulting to first temporal column: {TEMPORAL_COLUMNS[0]}"
        )
        return TEMPORAL_COLUMNS[0]

    return None


def _is_temporal_column(col_name: Optional[str]) -> bool:
    """Verifica se coluna é temporal."""
    if not col_name:
        return False

    return col_name in TEMPORAL_COLUMNS or any(
        keyword in col_name.lower()
        for keyword in ["data", "mes", "ano", "dia", "trimestre", "semestre"]
    )


# ========== Transformation 3: Normalize Aggregations ==========


def normalize_aggregations(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza funções de agregação baseadas no chart type.

    Garante que métricas tenham agregações apropriadas.

    Args:
        spec: Chart specification

    Returns:
        Spec com agregações normalizadas
    """
    chart_type = spec.get("chart_type")

    if chart_type is None or chart_type == "null":
        return spec

    if chart_type not in CHART_TYPE_REQUIREMENTS:
        return spec

    requirements = CHART_TYPE_REQUIREMENTS[chart_type]
    default_agg = requirements["default_aggregation"]

    metrics = spec.get("metrics", [])

    for metric in metrics:
        if isinstance(metric, dict):
            if not metric.get("aggregation"):
                metric["aggregation"] = default_agg
                logger.debug(
                    f"[normalize_aggregations] Set default aggregation '{default_agg}' "
                    f"for metric '{metric.get('name')}'"
                )

    return spec


# ========== Transformation 4: Adjust Dimensions by Chart Type ==========


def adjust_dimensions_by_chart_type(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajusta número de dimensões para os requisitos do chart type.

    Adiciona ou remove dimensões conforme necessário.

    Args:
        spec: Chart specification

    Returns:
        Spec com dimensões ajustadas
    """
    chart_type = spec.get("chart_type")

    if chart_type is None or chart_type == "null":
        return spec

    if chart_type not in CHART_TYPE_REQUIREMENTS:
        return spec

    dimensions = spec.get("dimensions", [])
    requirements = CHART_TYPE_REQUIREMENTS[chart_type]

    min_dims = requirements["min_dimensions"]
    max_dims = requirements["max_dimensions"]
    requires_temporal = requirements["requires_temporal"]

    # LAYER 6: Check intent_config for single_line variant
    # When dimension_structure.series=None, line_composed uses only 1 dimension (temporal)
    intent_config = spec.get("_intent_config") or spec.get("intent_config")
    if chart_type == "line_composed" and intent_config:
        dim_structure = intent_config.get("dimension_structure", {})
        if isinstance(dim_structure, dict) and dim_structure.get("series") is None:
            # single_line variant: only temporal dimension needed
            min_dims = 1
            max_dims = 1
            logger.info(
                f"[adjust_dimensions_by_chart_type] LAYER 6: line_composed single_line variant "
                f"detected (series=None). Adjusted min_dims=1, max_dims=1"
            )

    # Caso 1: Poucas dimensões
    if len(dimensions) < min_dims:
        logger.warning(
            f"[adjust_dimensions_by_chart_type] Insufficient dimensions for {chart_type}: "
            f"found {len(dimensions)}, need {min_dims}"
        )

        inferred_dims = _infer_dimensions_from_context(
            spec, count=min_dims - len(dimensions), requires_temporal=requires_temporal
        )

        if inferred_dims:
            if not dimensions:
                dimensions = []
            dimensions.extend(inferred_dims)
            spec["dimensions"] = dimensions
            logger.info(
                f"[adjust_dimensions_by_chart_type] Inferred {len(inferred_dims)} dimensions: "
                f"{[d['name'] for d in inferred_dims]}"
            )

    # Caso 2: Muitas dimensões
    elif len(dimensions) > max_dims:
        logger.warning(
            f"[adjust_dimensions_by_chart_type] Too many dimensions for {chart_type}: "
            f"found {len(dimensions)}, max {max_dims}"
        )

        dimensions = _select_best_dimensions(dimensions, max_dims, requires_temporal)
        spec["dimensions"] = dimensions
        logger.info(
            f"[adjust_dimensions_by_chart_type] Reduced to {len(dimensions)} dimensions"
        )

    return spec


def _infer_dimensions_from_context(
    spec: Dict[str, Any], count: int, requires_temporal: bool
) -> List[Dict[str, Any]]:
    """Inferir dimensões faltantes de parsed_entities ou defaults."""
    inferred = []
    parsed_entities = (
        spec.get("parsed_entities", {})
        if isinstance(spec.get("parsed_entities"), dict)
        else {}
    )
    potential_columns = parsed_entities.get("potential_columns", [])

    # Se requer temporal, priorizar
    if requires_temporal:
        temporal_dim = _infer_temporal_dimension_from_context(spec)
        if temporal_dim:
            inferred.append(
                {"name": temporal_dim, "alias": _prettify_label(temporal_dim)}
            )
            count -= 1

    # Tentar usar potential_columns
    for col_ref in potential_columns:
        if len(inferred) >= count:
            break

        if col_ref in DIMENSION_COLUMNS and col_ref not in [
            d["name"] for d in inferred
        ]:
            inferred.append({"name": col_ref, "alias": _prettify_label(col_ref)})

    # Se ainda precisa mais, usar defaults
    if len(inferred) < count:
        for dim_col in DIMENSION_COLUMNS:
            if len(inferred) >= count:
                break

            if dim_col not in [d["name"] for d in inferred]:
                inferred.append({"name": dim_col, "alias": _prettify_label(dim_col)})

    return inferred


def _select_best_dimensions(
    dimensions: List[Dict[str, Any]], max_count: int, requires_temporal: bool
) -> List[Dict[str, Any]]:
    """Seleciona as dimensões mais relevantes quando há muitas."""
    if len(dimensions) <= max_count:
        return dimensions

    selected = []

    # Prioridade 1: Dimensões temporais (se requeridas)
    if requires_temporal:
        for dim in dimensions:
            dim_name = dim.get("name") if isinstance(dim, dict) else dim
            if _is_temporal_column(dim_name):
                selected.append(dim)
                if len(selected) >= max_count:
                    return selected

    # Prioridade 2: Primeiras dimensões (mais relevantes)
    for dim in dimensions:
        if dim not in selected:
            selected.append(dim)
            if len(selected) >= max_count:
                return selected

    return selected


# ========== Transformation 5: Apply Chart-Specific Fixes ==========


def apply_chart_specific_fixes(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica fixes específicos por chart type.

    Args:
        spec: Chart specification

    Returns:
        Spec com fixes aplicados
    """
    chart_type = spec.get("chart_type")

    if chart_type is None or chart_type == "null":
        return spec

    # Fix 1: Histogram
    if chart_type == "histogram":
        # Histogramas NÃO devem ter dimensions (binning é feito na métrica)
        if spec.get("dimensions"):
            logger.warning(
                "[apply_chart_specific_fixes] Histogram should not have dimensions, removing"
            )
            spec["dimensions"] = []

        # Garantir agregação count ou raw
        metrics = spec.get("metrics", [])
        if metrics and isinstance(metrics[0], dict):
            if metrics[0].get("aggregation") not in ["count", None]:
                logger.info(
                    "[apply_chart_specific_fixes] Changing histogram aggregation to 'count'"
                )
                metrics[0]["aggregation"] = "count"

    # Fix 2: Pie chart
    elif chart_type == "pie":
        # Pie charts devem ter exatamente 1 dimensão
        dimensions = spec.get("dimensions", [])
        if len(dimensions) > 1:
            logger.warning(
                f"[apply_chart_specific_fixes] Pie chart should have 1 dimension, "
                f"found {len(dimensions)}, keeping first"
            )
            spec["dimensions"] = dimensions[:1]

    # Fix 3: Composed/Stacked charts
    elif chart_type in [
        "bar_vertical_composed",
        "bar_vertical_stacked",
        "line_composed",
    ]:
        dimensions = spec.get("dimensions", [])

        # LAYER 6: Check for single_line variant in line_composed
        intent_config = spec.get("_intent_config") or spec.get("intent_config")
        is_single_line = False
        if chart_type == "line_composed" and intent_config:
            dim_structure = intent_config.get("dimension_structure", {})
            if isinstance(dim_structure, dict) and dim_structure.get("series") is None:
                is_single_line = True
                logger.info(
                    "[apply_chart_specific_fixes] LAYER 6: line_composed single_line variant "
                    "(series=None) - 1 dimension is valid"
                )

        if len(dimensions) < 2 and not is_single_line:
            logger.warning(
                f"[apply_chart_specific_fixes] {chart_type} needs 2 dimensions "
                f"but has {len(dimensions)}. May need to infer secondary dimension."
            )

    return spec


# ========== Utility Functions ==========


def _prettify_label(value: Optional[str]) -> Optional[str]:
    """Converte nome de coluna para label amigável."""
    if not value:
        return None

    label = value.replace("_", " ").strip()
    if not label:
        return None

    return label[:1].upper() + label[1:]
