"""
Query classifier for non_graph_executor.

This module implements query classification and parameter extraction
for non-graph queries using keyword-based detection and LLM fallback.
"""

import logging
import json
import re
from typing import Dict, Optional, Any

from src.non_graph_executor.models.schemas import QueryTypeClassification
from src.non_graph_executor.tools.query_classifier_params import ParameterExtractor

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classificador de queries não-gráficas.

    Estratégia de classificação:
    1. Detecção rápida via keywords (prioridade alta)
    2. Uso de LLM apenas para casos ambíguos
    3. Extração de parâmetros específicos por tipo

    Ordem de prioridade:
    tabular > conversational > metadata > aggregation >
    statistical > lookup > textual > LLM fallback
    """

    # ========================================================================
    # KEYWORDS POR CATEGORIA (Estratégia escalável e contextual)
    # ========================================================================

    # TABULAR: Solicitação explícita de visualização de dados em formato tabela
    TABULAR_KEYWORDS = [
        "mostrar tabela",
        "mostre a tabela",
        "mostre tabela",
        "exibir tabela",
        "exiba tabela",
        "ver tabela",
        "veja tabela",
        "dados brutos",
        "tabela completa",
        "tabela de dados",
        "todos os dados",
        "ver todos os dados",
        "mostrar registros",
        "mostre registros",
        "exibir registros",
        "ver registros",
        "listar registros",
        "mostre os dados",
    ]

    # METADATA: Perguntas sobre estrutura e meta-informações do dataset
    # (Excluindo queries com termos de negócio que indicam agregação)
    METADATA_KEYWORDS = [
        "quantas linhas",
        "quantas colunas",
        "quantos registros",
        "número de registros",
        "total de linhas",
        "total de registros",
        "total de colunas",
        "quais colunas",
        "quais são as colunas",
        "quais os tipos",
        "quais campos",
        "liste as colunas",
        "lista de colunas",
        "tipos de dados",
        "tipo das colunas",
        "mostre os tipos",
        "valores únicos de",
        "valores únicos tem",
        "valores unicos de",
        "valores unicos tem",
        "quantos valores únicos",
        "quantos valores unicos",
        "valores nulos",
        "primeiras linhas",
        "últimas linhas",
        "mostre linhas",
        "mostre 5 linhas",
        "mostre 10 linhas",
        "mostre algumas linhas",
        "mostrar linhas",
        "mostrar 5 linhas",
        "mostrar 10 linhas",
        "mostre os campos",
        "amostra",
        "sample",
        "shape",
        "estrutura",
        "schema",
        "número de linhas",
        "número de colunas",
        "valores distintos de",
        "distinct count de",
        "distinct count",
        "linhas de exemplo",
        "exemplos de dados",
        "exemplos",
        "preview dos dados",
        "preview",
        "quantos registros tem",
    ]

    # AGGREGATION: Operações de agregação (soma, média, contagem, min, max, etc)
    # Palavras específicas que indicam cálculos agregados
    AGGREGATION_PATTERNS = [
        # AVG patterns
        ("média", None),
        ("media", None),
        ("average", None),
        ("valor médio", None),
        ("valor medio", None),
        # SUM patterns
        ("soma", None),
        ("somatório", None),
        ("somatorio", None),
        (
            "total de",
            ["vendas", "valor", "quantidade", "pedidos", "clientes"],
        ),  # "total de [business]" is aggregation
        ("qual o total", None),  # "qual o total de vendas"
        ("qual a total", None),
        # COUNT patterns
        (
            "quantos",
            ["clientes", "produtos", "pedidos", "vendas"],
        ),  # Must have business context
        ("quantas", ["vendas", "compras", "transações", "transacoes"]),
        ("número de", ["clientes", "produtos", "pedidos", "vendas"]),
        ("numero de", ["clientes", "produtos", "pedidos", "vendas"]),
        ("count de", None),
        ("count", None),
        ("contagem de", None),
        # MIN/MAX patterns
        ("menor", None),
        ("mínimo", None),
        ("minimo", None),
        ("qual o menor", None),
        ("qual a menor", None),
        ("maior", None),
        ("máximo", None),
        ("maximo", None),
        ("qual o maior", None),
        ("qual a maior", None),
        # MEDIAN patterns
        ("mediana", None),
        ("median", None),
        ("valor mediano", None),
    ]

    # STATISTICAL: Análises estatísticas avançadas (desvio, variância, quartis, etc)
    # Diferente de aggregation simples
    STATISTICAL_KEYWORDS = [
        "estatísticas",
        "estatisticas",
        "resumo estatístico",
        "resumo estatistico",
        "análise estatística",
        "analise estatistica",
        "quartis",
        "quartil",
        "variância",
        "variancia",
        "desvio padrão",
        "desvio padrao",
        "desvio-padrão",
        "iqr",
        "percentil",
        "percentis",
        "distribuição",
        "distribuicao",
        "q1",
        "q3",
        "std de",
        "variance",
    ]

    # TEXTUAL: Listagens e buscas textuais
    # Diferente de TABULAR (que mostra tudo) e LOOKUP (que busca registro específico)
    TEXTUAL_PATTERNS = [
        ("liste todos", None),  # "liste todos os X"
        ("listar todos", None),
        ("liste os", None),  # "liste os X"
        ("listar os", None),
        ("mostre todos os", None),  # "mostre todos os X" (quando não é tabela)
        ("mostrar todos os", None),
        ("contém", None),
        ("contem", None),
        ("que contém", None),
        ("que contem", None),
        ("buscar texto", None),
        ("procurar por", ["texto", "palavra", "nome"]),  # Busca textual específica
    ]

    # LOOKUP: Busca de registro específico por ID/código
    # Diferente de AGGREGATION (min/max)
    LOOKUP_PATTERNS = [
        ("cliente", ["123", "abc", "xyz"]),  # Indica ID específico
        ("pedido", ["123", "abc", "xyz"]),
        ("produto", ["123", "abc", "xyz"]),
        ("detalhes do", None),
        ("dados do", ["cliente", "pedido", "produto"]),
        ("informações do", None),
        ("informacoes do", None),
        ("registro", ["123", "abc", "xyz"]),
    ]

    # Business keywords para excluir conversationais
    # Qualquer termo de negócio/domínio indica que não é conversacional
    BUSINESS_KEYWORDS = [
        "vendas",
        "clientes",
        "produtos",
        "produto",
        "cliente",
        "valor",
        "valores",
        "quantidade",
        "quantidades",
        "tabela",
        "dados",
        "pedidos",
        "pedido",
        "faturamento",
        "receita",
        "preço",
        "preco",
        "preços",
        "precos",
        "peso",
        "qtd",
        "empresa",
        "filial",
        "matriz",
        "estado",
        "uf",
        "cidade",
        "data",
        "ano",
        "mes",
        "mês",
    ]

    # Saudações simples
    GREETINGS = ["oi", "olá", "ola", "hello", "hi", "bom dia", "boa tarde", "boa noite"]

    def __init__(self, alias_mapper, llm):
        """
        Initialize query classifier.

        Args:
            alias_mapper: AliasMapper instance for column resolution
            llm: LLM instance for ambiguous cases
        """
        self.alias_mapper = alias_mapper
        self.llm = llm
        logger.info("QueryClassifier initialized with keyword-based classification")

    def classify(self, query: str, state: Dict) -> QueryTypeClassification:
        """
        Classifica query usando estratégia inteligente e contextual.

        Ordem de prioridade (refatorada para evitar conflitos):
        1. TABULAR - Keywords explícitas de tabela
        2. CONVERSATIONAL - Saudações e ajuda
        3. METADATA - Estrutura do dataset (exceto com contexto de negócio)
        4. STATISTICAL - Análises avançadas (separado de aggregation)
        5. AGGREGATION - Cálculos simples (verificação contextual)
        6. TEXTUAL - Listagens (verificação contextual)
        7. LOOKUP - Busca específica (verificação contextual)
        8. LLM fallback

        Args:
            query: Query do usuário
            state: State do pipeline

        Returns:
            QueryTypeClassification com tipo, confidence e parâmetros
        """
        query_lower = query.lower()

        # ====================================================================
        # 1. METADATA com número de linhas - Prioridade sobre TABULAR
        # "mostre 5 linhas", "primeiras 10 registros" = sample_rows (metadata)
        # ====================================================================
        # Check if query has a specific small number (indicating sample, not full table)
        sample_match = re.search(
            r"(mostre|mostrar|exibir|ver|primeiras?|últimas?)\s+(\d+)\s*(linhas|registros|rows)",
            query_lower,
        )
        if sample_match:
            n = int(sample_match.group(2))
            if n <= 100:  # Small numbers indicate sample request
                logger.debug(f"Query classified as METADATA (sample_rows): {query}")
                return QueryTypeClassification(
                    query_type="metadata",
                    subtype="sample_rows",
                    confidence=0.95,
                    requires_llm=False,
                    parameters={"metadata_type": "sample_rows", "n": n},
                )

        # ====================================================================
        # 2. TABULAR - Prioridade para visualização completa de dados
        # ====================================================================
        if any(kw in query_lower for kw in self.TABULAR_KEYWORDS):
            logger.debug(f"Query classified as TABULAR: {query}")
            # Extract limit if present
            limit_match = re.search(r"(\d+)\s*(linhas|registros|rows)", query_lower)
            limit = int(limit_match.group(1)) if limit_match else 100
            return QueryTypeClassification(
                query_type="tabular",
                confidence=0.95,
                requires_llm=False,
                parameters={"limit": limit},
            )

        # ====================================================================
        # 2. CONVERSATIONAL - Saudações e ajuda
        # ====================================================================
        if self._is_conversational(query, query_lower):
            logger.debug(f"Query classified as CONVERSATIONAL: {query}")
            return QueryTypeClassification(
                query_type="conversational",
                confidence=0.98,
                requires_llm=True,
                parameters={},
            )

        # ====================================================================
        # 3. METADATA - Estrutura do dataset
        # Tem prioridade sobre aggregation para perguntas sobre estrutura
        # ====================================================================
        # Check for metadata keywords
        if any(kw in query_lower for kw in self.METADATA_KEYWORDS):
            # EXCEPT: "quantos/quantas" with specific business entity (clientes, produtos, etc)
            # But "quantas linhas", "quantos registros" are still metadata
            has_business_context = any(
                kw in query_lower for kw in self.BUSINESS_KEYWORDS
            )

            # Specific metadata terms that override business context
            # Including "valores únicos" for unique values, "tipos" for data types
            # But NOT "total de [business]" which is aggregation
            metadata_terms = [
                "linhas",
                "registros",
                "colunas",
                "campos",
                "rows",
                "columns",
                "valores únicos",
                "valores unicos",
                "distinct count",
                "tipos de dados",
                "tipo das",  # For dtypes
            ]
            has_metadata_terms = any(term in query_lower for term in metadata_terms)

            # If has metadata terms, it's always metadata (even with business keywords)
            # Otherwise, check if it's "quantos clientes" (aggregation) vs "quantas linhas" (metadata)
            if has_metadata_terms:
                logger.debug(f"Query classified as METADATA: {query}")
                params = self._extract_metadata_params(query, query_lower)
                return QueryTypeClassification(
                    query_type="metadata",
                    confidence=0.90,
                    requires_llm=False,
                    parameters=params,
                )
            elif not has_business_context:
                # No business context, definitely metadata
                logger.debug(f"Query classified as METADATA: {query}")
                params = self._extract_metadata_params(query, query_lower)
                return QueryTypeClassification(
                    query_type="metadata",
                    confidence=0.90,
                    requires_llm=False,
                    parameters=params,
                )
            # else: has business but no metadata terms → fall through to AGGREGATION

        # ====================================================================
        # 4. STATISTICAL - Análises avançadas (antes de aggregation)
        # ====================================================================
        if any(kw in query_lower for kw in self.STATISTICAL_KEYWORDS):
            logger.debug(f"Query classified as STATISTICAL: {query}")
            params = self._extract_statistical_params(query, state)
            return QueryTypeClassification(
                query_type="statistical",
                confidence=0.85,
                requires_llm=True,
                parameters=params,
            )

        # ====================================================================
        # 5. AGGREGATION - Cálculos simples com verificação contextual
        # ====================================================================
        if self._is_aggregation(query_lower):
            logger.debug(f"Query classified as AGGREGATION: {query}")
            params = self._extract_aggregation_params(query, state)
            return QueryTypeClassification(
                query_type="aggregation",
                confidence=0.85,
                requires_llm=True,
                parameters=params,
            )

        # ====================================================================
        # 6. TEXTUAL - Listagens com verificação contextual
        # ====================================================================
        if self._is_textual(query_lower):
            logger.debug(f"Query classified as TEXTUAL: {query}")
            params = self._extract_textual_params(query, state)
            return QueryTypeClassification(
                query_type="textual",
                confidence=0.80,
                requires_llm=True,
                parameters=params,
            )

        # ====================================================================
        # 7. LOOKUP - Busca específica com verificação contextual
        # ====================================================================
        if self._is_lookup(query_lower):
            logger.debug(f"Query classified as LOOKUP: {query}")
            params = self._extract_lookup_params(query, state)
            return QueryTypeClassification(
                query_type="lookup",
                confidence=0.80,
                requires_llm=True,
                parameters=params,
            )

        # ====================================================================
        # 8. LLM Fallback
        # ====================================================================
        logger.debug(f"No keyword match, using LLM fallback for: {query}")
        return self._llm_classify(query, state)

    def _is_aggregation(self, query_lower: str) -> bool:
        """
        Verifica se query é uma agregação usando padrões contextuais.

        Evita falsos positivos verificando contexto ao redor das keywords.

        Args:
            query_lower: Query em lowercase

        Returns:
            True se agregação, False caso contrário
        """
        for pattern, context_words in self.AGGREGATION_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's aggregation
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_textual(self, query_lower: str) -> bool:
        """
        Verifica se query é textual usando padrões contextuais.

        Args:
            query_lower: Query em lowercase

        Returns:
            True se textual, False caso contrário
        """
        for pattern, context_words in self.TEXTUAL_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's textual
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_lookup(self, query_lower: str) -> bool:
        """
        Verifica se query é lookup usando padrões contextuais.

        Lookup requer indicação de registro específico (ID, código, etc).

        Args:
            query_lower: Query em lowercase

        Returns:
            True se lookup, False caso contrário
        """
        for pattern, context_words in self.LOOKUP_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's lookup
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_conversational(self, query: str, query_lower: str) -> bool:
        """
        Detecta se query é conversacional.

        Critérios:
        - Query é saudação simples (GREETINGS)
        - Query tem <= 3 palavras E não contém keywords de negócio
        - NÃO deve classificar queries que contenham keywords de metadata
        - Queries de ajuda ("como funciona?", "o que você faz?", etc)

        Args:
            query: Query original
            query_lower: Query em lowercase

        Returns:
            True se conversacional, False caso contrário
        """
        # Check if it's a greeting
        if query_lower.strip() in self.GREETINGS:
            return True

        # Check for help/question patterns
        help_patterns = [
            "como funciona",
            "o que você faz",
            "o que voce faz",
            "pode me ajudar",
            "ajuda",
            "help",
            "como usar",
            "o que é isso",
            "o que e isso",
        ]
        if any(pattern in query_lower for pattern in help_patterns):
            return True

        # Check if query has metadata keywords (not conversational)
        if any(kw in query_lower for kw in self.METADATA_KEYWORDS):
            return False

        # Check if short and generic (no business/data keywords)
        # Must be TRULY conversational (greeting-like)
        words = query.split()
        if len(words) <= 3:
            has_business = any(kw in query_lower for kw in self.BUSINESS_KEYWORDS)
            # Also check for data-related terms
            data_terms = ["linhas", "registros", "rows", "dados", "tabela", "colunas"]
            has_data_terms = any(term in query_lower for term in data_terms)

            # Only conversational if no business keywords AND no data terms
            if not has_business and not has_data_terms:
                return True

        return False

    def _extract_metadata_params(self, query: str, query_lower: str) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de metadata.

        Args:
            query: Query original
            query_lower: Query em lowercase

        Returns:
            Dict com metadata_type e parâmetros adicionais
        """
        return ParameterExtractor.extract_metadata_params(
            query, query_lower, self.alias_mapper
        )

    def _extract_aggregation_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de agregação.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com aggregation, column e filters
        """
        return ParameterExtractor.extract_aggregation_params(
            query, state, self.alias_mapper
        )

    def _extract_lookup_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de lookup usando LLM.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com lookup_column e lookup_value
        """
        token_accumulator = state.get("_token_accumulator")
        return ParameterExtractor.extract_lookup_params(
            query,
            state,
            self.llm,
            self.alias_mapper,
            token_accumulator=token_accumulator,
        )

    def _extract_textual_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries textuais.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com column e search_term
        """
        return ParameterExtractor.extract_textual_params(
            query, state, self.alias_mapper
        )

    def _extract_statistical_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries estatísticas.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com column e filters
        """
        return ParameterExtractor.extract_statistical_params(
            query, state, self.alias_mapper
        )

    def _llm_classify(self, query: str, state: Dict) -> QueryTypeClassification:
        """
        Fallback para classificação via LLM quando keywords não funcionam.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            QueryTypeClassification com resultado da classificação
        """
        try:
            prompt = f"""Classifique a seguinte query em UMA das categorias abaixo:

Categorias:
- metadata: Perguntas sobre estrutura dos dados (linhas, colunas, tipos)
- aggregation: Agregações simples (média, soma, total, min, max)
- lookup: Busca de registros específicos
- textual: Buscas textuais ou listagens
- statistical: Estatísticas descritivas completas
- tabular: Solicitação de dados em formato tabela

Query: "{query}"

Retorne APENAS um JSON válido no formato:
{{"query_type": "tipo", "confidence": 0.0-1.0}}

Use confidence alto (0.8-0.9) se tiver certeza, médio (0.5-0.7) se ambíguo."""

            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Capture and accumulate tokens
            from src.shared_lib.utils.token_tracker import extract_token_usage

            tokens = extract_token_usage(response, self.llm)
            if "_token_accumulator" in state:
                state["_token_accumulator"].add(tokens)
                logger.debug(f"[QueryClassifier] Tokens accumulated: {tokens}")

            # Try to parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    # Try to find any JSON object
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(0))
                    else:
                        raise ValueError("No JSON found in response")

            query_type = result.get("query_type", "tabular")
            confidence = float(result.get("confidence", 0.5))

            logger.info(
                f"LLM classified query as: {query_type} (confidence: {confidence})"
            )

            # Extract parameters based on type
            parameters = {}
            if query_type == "aggregation":
                parameters = self._extract_aggregation_params(query, state)
            elif query_type == "lookup":
                parameters = self._extract_lookup_params(query, state)
            elif query_type == "textual":
                parameters = self._extract_textual_params(query, state)
            elif query_type == "statistical":
                parameters = self._extract_statistical_params(query, state)
            elif query_type == "metadata":
                parameters = self._extract_metadata_params(query, query.lower())
            elif query_type == "tabular":
                parameters = {"limit": 100}

            return QueryTypeClassification(
                query_type=query_type,
                confidence=confidence,
                requires_llm=True,
                parameters=parameters,
            )

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Fallback to tabular with low confidence
            return QueryTypeClassification(
                query_type="tabular",
                confidence=0.5,
                requires_llm=False,
                parameters={"limit": 100},
            )
