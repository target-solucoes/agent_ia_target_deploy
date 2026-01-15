"""
Query Analyzer - Heurísticas para otimização de pipeline.

Este módulo implementa heurísticas de análise de queries para determinar
quando certos agentes do pipeline podem ser pulados sem comprometer a acurácia.

Autor: Sistema de Otimização de Performance
Data: 2025-11-24
Fase: 2 - Estrutural (Problema #6: Filter Condicional)
"""

import re
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Resultado da análise de uma query."""

    needs_filter: bool
    confidence: float  # 0.0 a 1.0
    detected_keywords: List[str]
    detected_entities: List[str]
    reason: str


class QueryAnalyzer:
    """
    Analisador de queries para otimização de pipeline.

    Implementa heurísticas para determinar se uma query necessita
    do agente filter_classifier, permitindo skip quando apropriado.
    """

    # Palavras-chave que indicam necessidade de filtros
    FILTER_KEYWORDS_PT = {
        # Preposições de localização/tempo (removido "de" pois é muito genérico)
        "em",
        "do",
        "da",
        "dos",
        "das",
        "para",
        "no",
        "na",
        "nos",
        "nas",
        # Verbos de filtragem
        "filtrar",
        "filtrado",
        "filtre",
        "apenas",
        "somente",
        "só",
        "exclusivamente",
        # Comparativos
        "maior",
        "menor",
        "acima",
        "abaixo",
        "superior",
        "inferior",
        "entre",
        # Contextualizadores
        "onde",
        "que",
        "qual",
        "quais",
        "quando",
    }

    # Padrões de comparação temporal (FASE 2, Etapa 2.1)
    COMPARISON_KEYWORDS = {
        "versus",      # "maio versus junho"
        "vs",          # "maio vs junho"
        "comparar",    # "comparar maio com junho"
        "comparação",  # "comparação entre"
        "entre",       # "entre maio e junho"
        "variação",    # "variação de maio"
        "aumento",     # "aumento de vendas"
        "incremento",  # "incremento de maio"
        "crescimento", # "crescimento de maio para junho"
        "queda",       # "queda de maio para junho"
        "redução",     # "redução de maio para junho"
        "diferença",   # "diferença entre maio e junho"
    }

    # Estados brasileiros (UF)
    ESTADOS_BR = {
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RS",
        "RO",
        "RR",
        "SC",
        "SP",
        "SE",
        "TO",
    }

    # Cidades importantes (subset - pode ser expandido)
    CIDADES_IMPORTANTES = {
        "joinville",
        "florianópolis",
        "florianopolis",
        "curitiba",
        "porto alegre",
        "são paulo",
        "sao paulo",
        "rio de janeiro",
        "belo horizonte",
        "brasília",
        "brasilia",
        "salvador",
        "fortaleza",
        "recife",
        "manaus",
        "belém",
        "belem",
        "goiânia",
        "goiania",
        "campinas",
        "vitória",
        "vitoria",
    }

    # Padrões de anos (2015, 2016 - escopo permitido do dataset)
    ANOS_VALIDOS = {"2015", "2016"}

    # Padrões de meses
    MESES_PT = {
        "janeiro",
        "fevereiro",
        "março",
        "marco",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
        "jan",
        "fev",
        "mar",
        "abr",
        "mai",
        "jun",
        "jul",
        "ago",
        "set",
        "out",
        "nov",
        "dez",
    }

    # Queries genéricas que NÃO precisam de filtros
    GENERIC_PATTERNS = [
        r"^(mostre|exiba|faça|faz|gere|crie|criar)\s+(um\s+)?(gráfico|grafico|chart|visualização|visualizacao)",
        r"^(qual|quais)\s+(o|os|a|as)\s+(total|soma|média|media)",
        r"^(distribuição|distribuicao|análise|analise)\s+de",
        r"^(histórico|historico|evolução|evolucao)\s+de",
    ]

    def __init__(self):
        """Inicializa o analisador de queries."""
        self.generic_patterns_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.GENERIC_PATTERNS
        ]

    def _detect_temporal_comparison(self, query: str) -> bool:
        """
        Detecta padrões de comparação entre períodos temporais.

        Esta função identifica queries que comparam entidades entre dois períodos:
        - "de maio de 2016 para junho de 2016"
        - "maio vs junho"
        - "comparar maio com junho"
        - "variação de maio para junho"

        Args:
            query: Query em linguagem natural

        Returns:
            True se detecta padrão de comparação temporal, False caso contrário

        Note:
            Implementado na FASE 2, Etapa 2.1 para corrigir detecção de filtros
            em queries de comparação temporal.
        """
        patterns = [
            # Padrão: "de [mês] de [ano] para/a/até [mês] de [ano]"
            r"de\s+(\w+)\s+(de\s+)?\d{4}\s+(para|a|até)\s+(\w+)\s+(de\s+)?\d{4}",
            # Padrão: "[mês] vs/versus [mês]"
            r"(\w+)\s+(vs\.?|versus)\s+(\w+)",
            # Padrão: "comparar [período] com/e [período]"
            r"comparar\s+(\w+)\s+(com|e)\s+(\w+)",
            # Padrão: "entre [período] e [período]"
            r"entre\s+(\w+)\s+e\s+(\w+)",
            # Padrão: "variação/aumento/incremento/crescimento/queda/redução de [período] para [período]"
            r"(variação|aumento|incremento|crescimento|queda|redução|diferença)\s+de\s+(\w+)\s+(para|a)\s+(\w+)",
        ]

        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"Detected temporal comparison pattern: {pattern}")
                return True

        return False

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analisa uma query e determina se precisa do filter_classifier.

        Args:
            query: Query em linguagem natural

        Returns:
            QueryAnalysis com resultado da análise
        """
        query_lower = query.lower().strip()
        detected_keywords = []
        detected_entities = []
        confidence = 0.0

        # 1. Check anos (alta prioridade)
        anos_encontrados = [ano for ano in self.ANOS_VALIDOS if ano in query]
        if anos_encontrados:
            detected_entities.extend(anos_encontrados)
            confidence += 0.4
            logger.debug(f"Detected years: {anos_encontrados}")

        # 2. Check estados (UF)
        estados_encontrados = []
        for uf in self.ESTADOS_BR:
            # Match palavra completa ou em padrões específicos
            if re.search(rf"\b{uf}\b", query, re.IGNORECASE):
                estados_encontrados.append(uf)
        if estados_encontrados:
            detected_entities.extend(estados_encontrados)
            confidence += 0.4
            logger.debug(f"Detected states: {estados_encontrados}")

        # 3. Check cidades
        cidades_encontradas = [
            cidade for cidade in self.CIDADES_IMPORTANTES if cidade in query_lower
        ]
        if cidades_encontradas:
            detected_entities.extend(cidades_encontradas)
            confidence += 0.4
            logger.debug(f"Detected cities: {cidades_encontradas}")

        # 4. Check meses
        meses_encontrados = [mes for mes in self.MESES_PT if mes in query_lower]
        if meses_encontrados:
            detected_entities.extend(meses_encontrados)
            confidence += 0.3
            logger.debug(f"Detected months: {meses_encontrados}")

        # 5. Check keywords de filtro
        keywords_encontradas = [
            kw for kw in self.FILTER_KEYWORDS_PT if kw in query_lower.split()
        ]
        if keywords_encontradas:
            detected_keywords.extend(keywords_encontradas)
            confidence += 0.2 * len(keywords_encontradas)
            logger.debug(f"Detected filter keywords: {keywords_encontradas}")

        # 5.1. Check comparison keywords (FASE 2, Etapa 2.1)
        comparison_keywords_encontradas = [
            kw for kw in self.COMPARISON_KEYWORDS if kw in query_lower
        ]
        if comparison_keywords_encontradas:
            detected_keywords.extend(comparison_keywords_encontradas)
            confidence += 0.3  # Higher weight for comparison patterns
            logger.debug(
                f"Detected comparison keywords: {comparison_keywords_encontradas}"
            )

        # 5.2. Check temporal comparison patterns (FASE 2, Etapa 2.1)
        if self._detect_temporal_comparison(query):
            confidence += 0.4  # Strong indicator of filter need
            detected_keywords.append("temporal_comparison_pattern")
            logger.debug("Detected temporal comparison pattern")

        # 6. Check padrões genéricos APENAS se não há entidades detectadas
        if not detected_entities:
            for pattern in self.generic_patterns_compiled:
                if pattern.search(query_lower):
                    logger.debug(f"Query matched generic pattern: {pattern.pattern}")
                    return QueryAnalysis(
                        needs_filter=False,
                        confidence=0.95,
                        detected_keywords=[],
                        detected_entities=[],
                        reason="Query genérica sem especificação de filtros",
                    )

        # 7. Limitar confidence a 1.0
        confidence = min(confidence, 1.0)

        # 8. Decisão final (threshold: 0.3)
        needs_filter = confidence >= 0.3

        # Construir razão
        if not needs_filter:
            reason = "Query não possui indicadores de filtros específicos"
        else:
            components = []
            if detected_entities:
                components.append(f"entidades: {', '.join(detected_entities[:3])}")
            if detected_keywords:
                components.append(f"keywords: {', '.join(detected_keywords[:3])}")
            reason = f"Query indica necessidade de filtros ({'; '.join(components)})"

        logger.info(
            f"Query analysis: needs_filter={needs_filter}, "
            f"confidence={confidence:.2f}, reason='{reason}'"
        )

        return QueryAnalysis(
            needs_filter=needs_filter,
            confidence=confidence,
            detected_keywords=detected_keywords,
            detected_entities=detected_entities,
            reason=reason,
        )

    def should_skip_filter_classifier(self, query: str) -> bool:
        """
        Determina se o filter_classifier deve ser pulado para esta query.

        Este é o método principal usado pelo orchestrator.

        Args:
            query: Query em linguagem natural

        Returns:
            True se deve PULAR o filter_classifier, False caso contrário
        """
        analysis = self.analyze(query)
        return not analysis.needs_filter


# ============================================================================
# CONVENIENCE FUNCTIONS (para uso direto no orchestrator)
# ============================================================================


def needs_filter_classification(query: str) -> bool:
    """
    Determina se uma query necessita do filter_classifier.

    Esta é uma função de conveniência que pode ser usada diretamente
    sem instanciar o QueryAnalyzer.

    Args:
        query: Query em linguagem natural

    Returns:
        True se precisa do filter_classifier, False caso contrário

    Example:
        >>> needs_filter_classification("top 3 clientes de SP em 2015")
        True
        >>> needs_filter_classification("mostre um gráfico de vendas")
        False
    """
    analyzer = QueryAnalyzer()
    return not analyzer.should_skip_filter_classifier(query)


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analisa uma query e retorna informações detalhadas.

    Args:
        query: Query em linguagem natural

    Returns:
        QueryAnalysis com resultado completo da análise

    Example:
        >>> analysis = analyze_query("vendas de SP em 2015")
        >>> print(analysis.needs_filter)  # True
        >>> print(analysis.confidence)  # 0.8
        >>> print(analysis.detected_entities)  # ['SP', '2015']
    """
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)
