"""
Metric Detector - FASE 1.2 (Etapas 1.2.1 e 1.2.2)

Este modulo implementa deteccao semantica de metricas com contexto,
resolvendo ambiguidades em keywords como "vendas" que podem significar
tanto quantidade quanto valor monetario.

Problema Original:
- Query: "maior aumento de vendas"
- Metrica atual: Valor_Vendido (incorreto)
- Metrica esperada: Quantidade_Vendida (correto - contexto de "aumento")

Referencia: planning_graph_classifier_diagnosis.md - FASE 1, Etapa 1.2
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricDetectionResult:
    """Resultado da deteccao de metrica com contexto."""
    metric_name: str
    confidence: float
    reasoning: str
    context_keywords: List[str]
    ambiguity_resolved: bool = False


class MetricDetector:
    """
    Detector contextual de metricas que resolve ambiguidades baseado
    no contexto semantico da query.

    Esta classe implementa as Etapas 1.2.1 e 1.2.2 do planejamento:
    - Mapeamento de keywords para metricas
    - Deteccao semantica com contexto
    """

    def __init__(self, alias_mapper=None):
        """
        Inicializa o detector com mapeamentos de keywords.

        Args:
            alias_mapper: AliasMapper opcional para resolucao de aliases
        """
        self.alias_mapper = alias_mapper

        # =====================================================================
        # Etapa 1.2.1: Mapeamento de Keywords para Metricas
        # =====================================================================

        # Keywords que indicam QUANTIDADE (unidades, itens, volume)
        self.quantity_keywords = [
            "quantidade",
            "qtd",
            "volume",
            "unidades",
            "itens",
            "produtos vendidos",
            "numero de vendas",
            "volume de vendas",
            "aumento",  # "aumento de vendas" = aumento de quantidade
            "crescimento",
            "reducao",
            "queda",
            "variacao",
            "diferenca",
        ]

        # Keywords que indicam VALOR MONETARIO (R$, faturamento, receita)
        self.monetary_keywords = [
            "faturamento",
            "receita",
            "valor",
            "r$",
            "reais",
            "dinheiro",
            "financeiro",
            "lucro",
            "ganho",
            "rendimento",
        ]

        # Keywords especificas para outras metricas
        self.weight_keywords = [
            "peso",
            "kg",
            "tonelada",
            "quilos",
        ]

        # Mapeamento direto: keyword → metrica
        # IMPORTANTE: Este mapeamento é usado APÓS análise de contexto
        self.direct_metric_mapping = {
            # Quantidade
            "quantidade_vendida": "Qtd_Vendida",
            "qtd_vendida": "Qtd_Vendida",
            "quantidade": "Qtd_Vendida",
            "qtd": "Qtd_Vendida",

            # Valor monetário
            "valor_vendido": "Valor_Vendido",
            "faturamento": "Valor_Vendido",
            "receita": "Valor_Vendido",

            # Peso
            "peso_vendido": "Peso_Vendido",
            "peso": "Peso_Vendido",
        }

    def detect_metric(
        self,
        query: str,
        parsed_entities: Optional[Dict[str, Any]] = None
    ) -> MetricDetectionResult:
        """
        Detecta a metrica apropriada baseada na query e contexto.

        Esta funcao implementa a Etapa 1.2.2: Deteccao semantica de metricas.

        Args:
            query: Query do usuario
            parsed_entities: Entidades parseadas (opcional)

        Returns:
            MetricDetectionResult com metrica detectada e confianca

        Examples:
            >>> detector = MetricDetector()
            >>> result = detector.detect_metric("maior aumento de vendas")
            >>> result.metric_name
            'Qtd_Vendida'
            >>> result.confidence
            0.90
        """
        query_lower = query.lower()
        parsed_entities = parsed_entities or {}

        # Inicializar scores
        quantity_score = 0.0
        monetary_score = 0.0
        weight_score = 0.0

        # Contextos detectados
        quantity_contexts = []
        monetary_contexts = []
        weight_contexts = []

        # =====================================================================
        # Análise de Contexto: Detectar keywords que indicam tipo de métrica
        # =====================================================================

        # 1. Verificar keywords de QUANTIDADE
        for keyword in self.quantity_keywords:
            if keyword in query_lower:
                quantity_score += 1.0
                quantity_contexts.append(keyword)
                logger.debug(f"[MetricDetector] Quantity context: '{keyword}'")

        # 2. Verificar keywords de VALOR MONETÁRIO
        for keyword in self.monetary_keywords:
            if keyword in query_lower:
                monetary_score += 1.0
                monetary_contexts.append(keyword)
                logger.debug(f"[MetricDetector] Monetary context: '{keyword}'")

        # 3. Verificar keywords de PESO
        for keyword in self.weight_keywords:
            if keyword in query_lower:
                weight_score += 1.0
                weight_contexts.append(keyword)
                logger.debug(f"[MetricDetector] Weight context: '{keyword}'")

        # =====================================================================
        # Regra Especial: "vendas" é AMBÍGUO
        # =====================================================================
        # "vendas" pode significar:
        # - Quantidade_Vendida (contexto: "aumento de vendas", "volume")
        # - Valor_Vendido (contexto: "faturamento de vendas", "receita")
        #
        # Regra: Se "vendas" aparece SEM contexto monetário explícito,
        # assumir QUANTIDADE (uso mais comum)
        # =====================================================================

        has_vendas_keyword = any(
            keyword in query_lower
            for keyword in ["vendas", "venda", "vendidos", "vendido"]
        )

        if has_vendas_keyword:
            if monetary_score == 0:
                # "vendas" sem contexto monetário → QUANTIDADE
                quantity_score += 0.5  # Boost moderado
                quantity_contexts.append("vendas (implícito)")
                logger.info(
                    "[MetricDetector] 'vendas' detected WITHOUT monetary context "
                    "→ interpreting as QUANTITY (Qtd_Vendida)"
                )
            else:
                # "vendas" COM contexto monetário → VALOR
                monetary_score += 0.5
                monetary_contexts.append("vendas (monetário)")
                logger.info(
                    "[MetricDetector] 'vendas' detected WITH monetary context "
                    "→ interpreting as VALUE (Valor_Vendido)"
                )

        # =====================================================================
        # Determinar Métrica com Maior Score
        # =====================================================================

        total_score = quantity_score + monetary_score + weight_score

        if total_score == 0:
            # Fallback: Nenhum contexto detectado
            logger.warning("[MetricDetector] No metric context detected, using fallback")
            return self._fallback_metric_detection(query, parsed_entities)

        # Normalizar scores para 0-1
        quantity_confidence = quantity_score / total_score if total_score > 0 else 0.0
        monetary_confidence = monetary_score / total_score if total_score > 0 else 0.0
        weight_confidence = weight_score / total_score if total_score > 0 else 0.0

        # Selecionar métrica com maior confiança
        if quantity_confidence > monetary_confidence and quantity_confidence > weight_confidence:
            metric_name = "Qtd_Vendida"
            confidence = quantity_confidence
            context_keywords = quantity_contexts
            reasoning = (
                f"Contexto de QUANTIDADE detectado: {', '.join(quantity_contexts)}. "
                f"Confidence: {confidence:.2f}"
            )
        elif monetary_confidence > weight_confidence:
            metric_name = "Valor_Vendido"
            confidence = monetary_confidence
            context_keywords = monetary_contexts
            reasoning = (
                f"Contexto MONETÁRIO detectado: {', '.join(monetary_contexts)}. "
                f"Confidence: {confidence:.2f}"
            )
        else:
            metric_name = "Peso_Vendido"
            confidence = weight_confidence
            context_keywords = weight_contexts
            reasoning = (
                f"Contexto de PESO detectado: {', '.join(weight_contexts)}. "
                f"Confidence: {confidence:.2f}"
            )

        # Verificar se houve resolução de ambiguidade
        ambiguity_resolved = has_vendas_keyword and (quantity_score > 0 or monetary_score > 0)

        logger.info(
            f"[MetricDetector] Detected metric: {metric_name} "
            f"(confidence={confidence:.2f}, ambiguity_resolved={ambiguity_resolved})"
        )

        return MetricDetectionResult(
            metric_name=metric_name,
            confidence=confidence,
            reasoning=reasoning,
            context_keywords=context_keywords,
            ambiguity_resolved=ambiguity_resolved
        )

    def _fallback_metric_detection(
        self,
        query: str,
        parsed_entities: Dict[str, Any]
    ) -> MetricDetectionResult:
        """
        Fallback para quando nenhum contexto claro é detectado.

        Implementa a Etapa 1.2.3: Fallback com score de confiança.

        Args:
            query: Query original
            parsed_entities: Entidades parseadas

        Returns:
            MetricDetectionResult com métrica de fallback
        """
        query_lower = query.lower()

        # Fallback 1: Verificar keywords diretas (sem contexto)
        for keyword, metric in self.direct_metric_mapping.items():
            if keyword in query_lower:
                logger.info(
                    f"[MetricDetector] Fallback: Direct keyword match '{keyword}' → {metric}"
                )
                return MetricDetectionResult(
                    metric_name=metric,
                    confidence=0.70,  # Confiança moderada
                    reasoning=f"Fallback: Direct keyword match '{keyword}'",
                    context_keywords=[keyword],
                    ambiguity_resolved=False
                )

        # Fallback 2: Usar agregação detectada
        aggregation = parsed_entities.get("aggregation")
        if aggregation == "count":
            logger.info("[MetricDetector] Fallback: aggregation=count → Qtd_Vendida")
            return MetricDetectionResult(
                metric_name="Qtd_Vendida",
                confidence=0.60,
                reasoning="Fallback: COUNT aggregation suggests quantity metric",
                context_keywords=["count"],
                ambiguity_resolved=False
            )

        # Fallback 3: Default universal (Valor_Vendido é a métrica mais comum)
        logger.info("[MetricDetector] Fallback: No context → defaulting to Valor_Vendido")
        return MetricDetectionResult(
            metric_name="Valor_Vendido",
            confidence=0.50,  # Confiança baixa
            reasoning="Fallback: No clear context, using default metric (Valor_Vendido)",
            context_keywords=[],
            ambiguity_resolved=False
        )

    def get_all_supported_metrics(self) -> List[str]:
        """
        Retorna lista de todas as métricas suportadas.

        Returns:
            Lista de nomes de métricas
        """
        return ["Qtd_Vendida", "Valor_Vendido", "Peso_Vendido"]

    def validate_metric_compatibility(
        self,
        metric_name: str,
        aggregation: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida se a métrica é compatível com a agregação.

        Args:
            metric_name: Nome da métrica
            aggregation: Tipo de agregação (sum, avg, max, min, count)

        Returns:
            Tupla (is_valid, error_message)
        """
        # Regras de compatibilidade
        incompatible_combinations = [
            # (metric, aggregation, reason)
            ("Qtd_Vendida", "max", "MAX é inapropriado para vendas totais"),
            ("Valor_Vendido", "max", "MAX é inapropriado para vendas totais"),
            ("Peso_Vendido", "max", "MAX é inapropriado para vendas totais"),
        ]

        for incompat_metric, incompat_agg, reason in incompatible_combinations:
            if metric_name == incompat_metric and aggregation == incompat_agg:
                return False, reason

        return True, None


# =============================================================================
# Função Helper para Integração no Workflow
# =============================================================================

def detect_metric_from_query(
    query: str,
    parsed_entities: Optional[Dict[str, Any]] = None,
    alias_mapper=None
) -> Dict[str, Any]:
    """
    Funcao helper para detectar metrica a partir da query.

    Esta funcao pode ser integrada no workflow do graphic_classifier.

    Args:
        query: Query do usuario
        parsed_entities: Entidades parseadas (opcional)
        alias_mapper: AliasMapper (opcional)

    Returns:
        Dict com metrica detectada e metadados:
        {
            "metric_name": str,
            "confidence": float,
            "reasoning": str,
            "ambiguity_resolved": bool
        }

    Examples:
        >>> result = detect_metric_from_query("maior aumento de vendas")
        >>> result["metric_name"]
        'Qtd_Vendida'
    """
    detector = MetricDetector(alias_mapper=alias_mapper)
    detection_result = detector.detect_metric(query, parsed_entities)

    return {
        "metric_name": detection_result.metric_name,
        "confidence": detection_result.confidence,
        "reasoning": detection_result.reasoning,
        "ambiguity_resolved": detection_result.ambiguity_resolved,
        "context_keywords": detection_result.context_keywords,
    }
