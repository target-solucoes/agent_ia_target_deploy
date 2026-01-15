"""
Prompt builder for generating LLM prompts based on chart type and numeric summaries.

This module contains the system prompt and template functions for building
context-specific prompts for insight generation.
"""

from typing import Dict, Any


SYSTEM_PROMPT = """Você é um analista de dados especializado em gerar insights estratégicos executivos completos.

═══════════════════════════════════════════════════════════════════════════════
FASE 4 - LLM CALL UNIFICATION: GERAÇÃO COMPLETA DE INSIGHTS
═══════════════════════════════════════════════════════════════════════════════

Você deve gerar UM ÚNICO JSON contendo TODOS os componentes do relatório analítico:
1. Executive Summary (título + introdução contextual)
2. Detailed Insights (5 insights estruturados com fórmulas transparentes)
3. Synthesized Insights (narrative + key_findings)
4. Next Steps (3 recomendações estratégicas)

═══════════════════════════════════════════════════════════════════════════════
REGRA 1: FORMATO DE RESPOSTA - JSON COMPLETO ESTRUTURADO
═══════════════════════════════════════════════════════════════════════════════

Retorne APENAS um JSON válido com a seguinte estrutura EXATA:

{
  "executive_summary": {
    "title": "Título Profissional da Análise (max 80 caracteres)",
    "introduction": "Parágrafo introdutório contextual de 2-3 frases (50-300 caracteres)"
  },
  "detailed_insights": [
    {
      "title": "Título Executivo do Insight",
      "formula": "Fórmula completa com valores numéricos",
      "interpretation": "Implicação estratégica concisa (max 150 caracteres)"
    }
  ],
  "synthesized_insights": {
    "narrative": "Narrativa coesa conectando os insights principais (400-800 caracteres)",
    "key_findings": [
      "Bullet point 1 conciso (max 140 caracteres)",
      "Bullet point 2 conciso (max 140 caracteres)",
      "Bullet point 3 conciso (max 140 caracteres)"
    ]
  },
  "next_steps": {
    "recommendations": [
      "Recomendação estratégica 1 acionável (max 200 caracteres)",
      "Recomendação estratégica 2 acionável (max 200 caracteres)",
      "Recomendação estratégica 3 acionável (max 200 caracteres)"
    ]
  }
}

═══════════════════════════════════════════════════════════════════════════════
REGRA 2: EXECUTIVE SUMMARY - Contextualização Profissional
═══════════════════════════════════════════════════════════════════════════════

**title**: 
- Máximo 80 caracteres
- Capture a essência da análise
- Linguagem profissional e direta
- Mencione a dimensão analisada e contexto

**introduction**:
- 50-300 caracteres (2-3 frases)
- OBRIGATÓRIO: Mencione TODOS os filtros aplicados em **negrito** (markdown bold)
- Se há filtros no contexto, eles DEVEM aparecer explicitamente na introduction
- Formato: "Esta análise examina ... para **Ano: 2016**, **Região: Sul**, **Produto: X** ..."
- Tom executivo e objetivo
- Prepare o leitor para os insights

EXEMPLOS:
✓ title: "Análise de Ranking de Produtos por Faturamento em SP - 2016"
✓ introduction: "Esta análise examina o desempenho de vendas para **Região: São Paulo** no **Período: 2016**, identificando concentrações críticas e oportunidades estratégicas."

ANTI-EXEMPLOS (NUNCA FAÇA para introduction):
✗ "Esta análise examina o desempenho..." (sem mencionar filtros)
✗ "Análise para São Paulo em 2016" (filtros sem bold)

═══════════════════════════════════════════════════════════════════════════════
REGRA 3: DETAILED INSIGHTS - Transparência Total com Fórmulas
═══════════════════════════════════════════════════════════════════════════════

Gere EXATAMENTE 5 insights estruturados.

Cada insight DEVE conter:
- **title**: Título executivo claro
- **formula**: Fórmula COMPLETA com valores numéricos e operadores
- **interpretation**: 1-2 frases sobre implicação estratégica (max 150 caracteres)

EXEMPLOS CORRETOS de fórmulas:
✓ "Top 3 = R$ 8,66M / Total R$ 12,68M → 68,3%"
✓ "Gap = Líder - Segundo = R$ 3,4M - R$ 2,1M = R$ 1,3M (62% maior)"
✓ "Variação = (Final - Inicial) / Inicial = (450 - 300) / 300 = +50%"
✓ "Amplitude = Max - Min = 500 - 100 = 400"

ANTI-EXEMPLOS (NUNCA FAÇA):
✗ "Top 3 representa 68,3%" (sem valores base)
✗ "O líder tem 62% a mais" (sem fórmula completa)
✗ "Crescimento de 50%" (sem cálculo explícito)

═══════════════════════════════════════════════════════════════════════════════
REGRA 4: SYNTHESIZED INSIGHTS - Narrativa Coesa e Key Findings
═══════════════════════════════════════════════════════════════════════════════

**narrative**:
- 400-800 caracteres (parágrafo executivo)
- Conecte os insights principais em uma narrativa fluida
- Use linguagem natural, não telegráfica
- TODA métrica mencionada DEVE ter correspondência em detailed_insights
- Sem emojis

**key_findings**:
- Exatamente 3-5 bullet points
- Máximo 140 caracteres cada
- Conciso, acionável e com valores concretos
- Priorize insights mais estratégicos

EXEMPLO de narrative:
"A análise revela concentração extrema nos principais produtos, com o Top 3 representando 68,3% do faturamento total. O líder mantém vantagem de 62% sobre o segundo colocado, criando barreira competitiva significativa. Produtos fora do Top 10 apresentam oportunidades de crescimento inexploradas."

EXEMPLO de key_findings:
[
  "Top 3 produtos concentram 68,3% da receita → dependência crítica",
  "Líder com vantagem de 62% sobre segundo → posição defensável",
  "Cauda longa com potencial inexplorado → oportunidade de diversificação"
]

═══════════════════════════════════════════════════════════════════════════════
REGRA 5: NEXT STEPS - Recomendações Estratégicas Acionáveis
═══════════════════════════════════════════════════════════════════════════════

**recommendations**:
- Exatamente 3 recomendações estratégicas
- Máximo 200 caracteres cada
- Diretas, acionáveis e contextualizadas aos insights
- Foco em ação executiva (investigar, desenvolver, estabelecer)

EXEMPLOS:
[
  "Investigar causas da concentração nos Top 3 e desenvolver estratégias de retenção para mitigar risco de dependência",
  "Avaliar oportunidades de crescimento nos produtos de menor performance para reduzir concentração",
  "Estabelecer monitoramento contínuo dos top performers para identificar mudanças de padrão rapidamente"
]

═══════════════════════════════════════════════════════════════════════════════
REGRA 6: ALINHAMENTO E CONSISTÊNCIA
═══════════════════════════════════════════════════════════════════════════════

INVARIANTES OBRIGATÓRIAS:
- [ ] Toda métrica citada em narrative APARECE em detailed_insights
- [ ] Valores numéricos são consistentes entre seções
- [ ] key_findings derivam dos detailed_insights
- [ ] next_steps são contextualizados aos insights identificados
- [ ] Zero emojis em todo o output
- [ ] JSON válido e bem formatado

═══════════════════════════════════════════════════════════════════════════════
CHECKLIST FINAL DE VALIDAÇÃO
═══════════════════════════════════════════════════════════════════════════════

Antes de retornar, valide:
- [ ] JSON válido com todas as 4 seções principais
- [ ] executive_summary tem title (≤80) e introduction (50-300)
- [ ] detailed_insights tem EXATAMENTE 5 itens com title, formula, interpretation
- [ ] Todas as fórmulas contêm operadores (=, /, -, +, →) e valores numéricos
- [ ] synthesized_insights tem narrative (400-800) e key_findings (3-5 itens)
- [ ] Cada key_finding tem ≤140 caracteres
- [ ] next_steps tem exatamente 3 recommendations (≤200 cada)
- [ ] Alinhamento: métricas em narrative ↔ detailed_insights

═══════════════════════════════════════════════════════════════════════════════
"""


# Templates por chart type com few-shot examples
CHART_TYPE_TEMPLATES = {
    "bar_horizontal": """
═══════════════════════════════════════════════════════════════════════════════
ANÁLISE DE RANKING
═══════════════════════════════════════════════════════════════════════════════

DADOS DISPONÍVEIS:
{dados}

CONTEXTO DA ANÁLISE:
- Tipo: Ranking de desempenho
- Foco: Concentração, gap competitivo, distribuição de poder
- Prioridade: Identificar dependências críticas e oportunidades

INSTRUÇÕES ESPECÍFICAS PARA DETAILED_INSIGHTS:
Gere 5 insights focados em:
1. Concentração de poder (Top N vs universo total com fórmula explícita)
2. Gap competitivo entre líder e demais (cálculo absoluto e relativo)
3. Oportunidades na cauda (itens fora do Top N)
4. Riscos de dependência crítica (análise de exposição)
5. Dinâmica competitiva e projeções estratégicas

EXEMPLO COMPLETO DE OUTPUT:
{{
  "executive_summary": {{
    "title": "Análise de Ranking: Concentração e Oportunidades Estratégicas",
    "introduction": "Esta análise examina a distribuição de desempenho entre os principais itens do ranking, identificando concentrações críticas que representam riscos operacionais e oportunidades de diversificação."
  }},
  "detailed_insights": [
    {{
      "title": "Concentração Extrema no Top 3",
      "formula": "Top 3 = R$ 8,66M / Total R$ 12,68M → 68,3%",
      "interpretation": "Dependência crítica. Risco de perda de 68% da receita se Top 3 apresentar retração."
    }},
    {{
      "title": "Gap Competitivo Insuperável",
      "formula": "Gap = Líder - Segundo = R$ 3,4M - R$ 2,1M = R$ 1,3M (62% maior)",
      "interpretation": "Vantagem competitiva robusta do líder. Difícil de reverter no curto prazo."
    }},
    {{
      "title": "Cauda Longa Subutilizada",
      "formula": "Itens fora Top 10 = R$ 2,5M / Total R$ 12,68M → 19,7%",
      "interpretation": "Oportunidade de crescimento inexplorada. Diversificação reduziria dependência."
    }},
    {{
      "title": "Risco de Exposição Elevado",
      "formula": "Líder = R$ 3,4M / Total R$ 12,68M → 26,8% de dependência",
      "interpretation": "Exposição individual crítica. Perda do líder impactaria mais de 1/4 do total."
    }},
    {{
      "title": "Dinâmica Competitiva Estável",
      "formula": "CV (Top 5) = Desvio / Média = 0,45 (moderado)",
      "interpretation": "Dispersão moderada indica estabilidade. Posições consolidadas com baixa volatilidade."
    }}
  ],
  "synthesized_insights": {{
    "narrative": "A análise revela concentração extrema, com os Top 3 itens representando 68,3% do total. O líder mantém vantagem significativa de 62% sobre o segundo colocado, criando barreira competitiva robusta. Os itens fora do Top 10 representam apenas 19,7% do total, indicando oportunidades de crescimento inexploradas que poderiam reduzir a dependência crítica dos principais performers.",
    "key_findings": [
      "Top 3 concentram 68,3% do total → dependência crítica com risco operacional elevado",
      "Líder com vantagem de 62% sobre segundo → posição defensável e estável",
      "Cauda longa com 19,7% → oportunidade de diversificação subutilizada"
    ]
  }},
  "next_steps": {{
    "recommendations": [
      "Investigar causas da concentração no Top 3 e desenvolver estratégias de retenção para mitigar risco de dependência crítica",
      "Avaliar oportunidades de crescimento nos itens de menor performance para reduzir concentração e aumentar resiliência",
      "Estabelecer monitoramento contínuo dos top performers para identificar mudanças de padrão e antecipar riscos"
    ]
  }}
}}
""",
    "bar_vertical": """
DADOS DE COMPARAÇÃO:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Amplitude Significativa",
      "formula": "Amplitude = Max - Min = R$ 500K - R$ 100K = R$ 400K",
      "interpretation": "Dispersão de 80% indica heterogeneidade. Oportunidade de equalização."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Amplitude de variação entre categorias
2. Análise de extremos (máximo vs mínimo)
3. Dispersão relativa à média
4. Padrões de distribuição identificados
5. Implicações estratégicas
""",
    "bar_vertical_composed": """
DADOS DE COMPARAÇÃO MULTI-SÉRIES:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Série Dominante Identificada",
      "formula": "Série A = R$ 5,2M / Total R$ 8,0M → 65% de dominância",
      "interpretation": "Série A lidera com folga. Outras séries têm potencial subutilizado."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Série dominante e seu peso relativo
2. Variabilidade entre séries (CV, desvio)
3. Correlações identificadas entre séries
4. Dinâmica competitiva multi-série
5. Oportunidades de balanceamento
""",
    "bar_vertical_stacked": """
DADOS DE COMPOSIÇÃO EMPILHADA:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Composição Desequilibrada",
      "formula": "Componente X = R$ 3,5M / Total Stack R$ 5,0M → 70%",
      "interpretation": "Componente X domina a composição. Risco de concentração elevado."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Composição total e contribuição por componente
2. Contribuição percentual de cada stack
3. Padrões de empilhamento identificados
4. Análise de participação relativa
5. Recomendações estratégicas de rebalanceamento
""",
    "line": """
DADOS TEMPORAIS:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Crescimento Acelerado",
      "formula": "Variação = (Final - Inicial) / Inicial = (450 - 300) / 300 = +50%",
      "interpretation": "Crescimento robusto. Tendência indica potencial para continuidade."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Evolução temporal e variação total
2. Tendência identificada (crescente/decrescente/estável)
3. Volatilidade e consistência da série
4. Pontos de inflexão ou mudanças críticas
5. Projeção futura e implicações estratégicas
""",
    "line_composed": """
DADOS TEMPORAIS MULTI-SÉRIES:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Divergência Entre Séries",
      "formula": "Série A: +40% vs Série B: -15% no período",
      "interpretation": "Séries divergem fortemente. Série A ganha enquanto B perde market share."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Evolução comparativa entre séries
2. Séries divergentes ou convergentes
3. Correlações temporais identificadas
4. Liderança ao longo do tempo (mudanças de posição)
5. Dinâmica competitiva multi-temporal
""",
    "pie": """
DADOS DE DISTRIBUIÇÃO:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Concentração Alta",
      "formula": "Top 3 = R$ 6,5M / Total R$ 10,0M → 65%",
      "interpretation": "Alta concentração indica dependência de poucas categorias. Risco moderado."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Concentração geral (HHI, Top N)
2. Categoria dominante e seu peso
3. Índice de diversificação e equilíbrio
4. Fragmentação identificada (categorias <5%)
5. Recomendações de portfolio e diversificação
""",
    "histogram": """
DADOS DE DISTRIBUIÇÃO DE FREQUÊNCIA:
{dados}

EXEMPLO DE OUTPUT ESPERADO:
{{
  "insights": [
    {{
      "title": "Distribuição Assimétrica",
      "formula": "Moda = 150-200 (40% das observações) vs Média = 220",
      "interpretation": "Assimetria positiva. Maioria concentrada abaixo da média."
    }}
  ]
}}

GERE 5 INSIGHTS JSON SOBRE:
1. Distribuição de frequências e forma
2. Concentração modal e picos identificados
3. Assimetria da distribuição (skewness)
4. Outliers e valores extremos
5. Implicações operacionais do padrão de dispersão
""",
}


def _format_number(value: float, is_percentage: bool = False) -> str:
    """Formata numero com separadores de milhares."""
    if is_percentage:
        return f"{value:.2f}%"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"


def _format_ranking_metrics(metrics: Dict[str, Any]) -> str:
    """Formata metricas de ranking (bar_horizontal) com formulas explicitas."""
    lines = []

    # Extrai valores principais
    total = metrics.get("total", 0)
    top_n = metrics.get("top_n", 0)
    sum_top_n = metrics.get("sum_top_n", 0)
    concentracao_top_n = metrics.get("concentracao_top_n_pct", 0)
    top3_sum = metrics.get("top3_sum", 0)
    concentracao_top3 = metrics.get("concentracao_top3_pct", 0)
    total_items = metrics.get("total_items", 0)

    lider_valor = metrics.get("lider_valor", 0)
    lider_label = metrics.get("lider_label", "")
    peso_lider = metrics.get("peso_lider_total_pct", 0)

    segundo_valor = metrics.get("segundo_valor", 0)
    segundo_label = metrics.get("segundo_label", "")
    gap_absoluto = metrics.get("gap_absoluto", 0)
    gap_percentual = metrics.get("gap_percentual", 0)

    # Formata com formulas explicitas
    if top_n > 0 and total > 0:
        lines.append(
            f"Top {top_n} = {_format_number(sum_top_n)} / Total {_format_number(total)} "
            f"→ {concentracao_top_n:.2f}% ({top_n} itens de {total_items})"
        )

    if top3_sum > 0 and total > 0:
        lines.append(
            f"Top 3 = {_format_number(top3_sum)} / Total {_format_number(total)} "
            f"→ {concentracao_top3:.2f}%"
        )

    if lider_valor > 0 and total > 0:
        lines.append(
            f"Lider ({lider_label}) = {_format_number(lider_valor)} "
            f"({peso_lider:.2f}% do total)"
        )

    if segundo_valor > 0 and gap_absoluto > 0:
        lines.append(
            f"Gap competitivo = Lider - Segundo = {_format_number(lider_valor)} - {_format_number(segundo_valor)} "
            f"= {_format_number(gap_absoluto)} ({gap_percentual:.2f}% maior)"
        )

    # Adiciona metricas adicionais relevantes
    tail_sum = metrics.get("tail_sum")
    tail_pct = metrics.get("tail_pct")
    if tail_sum is not None and tail_pct is not None:
        lines.append(
            f"Cauda (fora Top {top_n}) = {_format_number(tail_sum)} ({tail_pct:.2f}% do total)"
        )

    # Se nenhuma metrica especifica foi formatada, usa fallback generico
    if not lines:
        return _format_generic_metrics(metrics)

    return "\n".join(lines)


def _format_temporal_metrics(metrics: Dict[str, Any]) -> str:
    """Formata metricas temporais (line) com formulas explicitas."""
    lines = []

    valor_inicial = metrics.get("valor_inicial", 0)
    valor_final = metrics.get("valor_final", 0)
    variacao_absoluta = metrics.get("variacao_absoluta", 0)
    variacao_percentual = metrics.get("variacao_percentual", 0)

    if valor_inicial > 0 and valor_final > 0:
        lines.append(
            f"Variacao = (Final - Inicial) / Inicial = "
            f"({_format_number(valor_final)} - {_format_number(valor_inicial)}) / {_format_number(valor_inicial)} "
            f"= {_format_number(variacao_absoluta)} ({variacao_percentual:+.2f}%)"
        )

    tendencia = metrics.get("tendencia")
    if tendencia:
        lines.append(f"Tendencia detectada: {tendencia}")

    aceleracao = metrics.get("aceleracao_pct")
    if aceleracao is not None:
        lines.append(f"Aceleracao: {aceleracao:+.2f}%")

    max_valor = metrics.get("max_valor")
    min_valor = metrics.get("min_valor")
    if max_valor is not None and min_valor is not None:
        amplitude = max_valor - min_valor
        lines.append(
            f"Amplitude = Max - Min = {_format_number(max_valor)} - {_format_number(min_valor)} "
            f"= {_format_number(amplitude)}"
        )

    # Adiciona outras metricas numericas
    for key, value in metrics.items():
        if key not in [
            "valor_inicial",
            "valor_final",
            "variacao_absoluta",
            "variacao_percentual",
            "tendencia",
            "aceleracao_pct",
            "max_valor",
            "min_valor",
        ]:
            if (
                isinstance(value, (int, float))
                and not key.endswith("_col")
                and not key.endswith("_label")
            ):
                if key.endswith("_pct") or "percentual" in key.lower():
                    lines.append(f"{key}: {value:.2f}%")
                else:
                    lines.append(f"{key}: {_format_number(value)}")

    # Se nenhuma metrica especifica foi formatada, usa fallback generico
    if not lines:
        return _format_generic_metrics(metrics)

    return "\n".join(lines)


def _format_comparison_metrics(metrics: Dict[str, Any]) -> str:
    """Formata metricas de comparacao (bar_vertical) com formulas explicitas."""
    lines = []

    max_valor = metrics.get("max_valor", 0)
    max_label = metrics.get("max_label", "")
    min_valor = metrics.get("min_valor", 0)
    min_label = metrics.get("min_label", "")

    if max_valor > 0 and min_valor >= 0:
        amplitude = max_valor - min_valor
        lines.append(
            f"Amplitude = Max ({max_label}) - Min ({min_label}) = "
            f"{_format_number(max_valor)} - {_format_number(min_valor)} = {_format_number(amplitude)}"
        )

    media = metrics.get("media", 0)
    dispersao_pct = metrics.get("dispersao_pct", 0)
    if media > 0:
        lines.append(f"Media: {_format_number(media)}")
        if dispersao_pct > 0:
            lines.append(f"Dispersao: {dispersao_pct:.2f}% da media")

    # Adiciona outras metricas
    for key, value in metrics.items():
        if key not in [
            "max_valor",
            "max_label",
            "min_valor",
            "min_label",
            "media",
            "dispersao_pct",
        ]:
            if (
                isinstance(value, (int, float))
                and not key.endswith("_col")
                and not key.endswith("_label")
            ):
                if key.endswith("_pct") or "percentual" in key.lower():
                    lines.append(f"{key}: {value:.2f}%")
                else:
                    lines.append(f"{key}: {_format_number(value)}")

    # Se nenhuma metrica especifica foi formatada, usa fallback generico
    if not lines:
        return _format_generic_metrics(metrics)

    return "\n".join(lines)


def _format_distribution_metrics(metrics: Dict[str, Any]) -> str:
    """Formata metricas de distribuicao (pie) com formulas explicitas."""
    lines = []

    total = metrics.get("total", 0)
    top_n = metrics.get("top_n", 0)
    sum_top_n = metrics.get("sum_top_n", 0)
    concentracao_top_n = metrics.get("concentracao_top_n_pct", 0)

    if top_n > 0 and total > 0:
        lines.append(
            f"Top {top_n} = {_format_number(sum_top_n)} / Total {_format_number(total)} "
            f"→ {concentracao_top_n:.2f}%"
        )

    hhi = metrics.get("hhi")
    if hhi is not None:
        lines.append(f"Indice HHI (concentracao): {hhi:.2f}")

    diversidade = metrics.get("diversidade_pct")
    if diversidade is not None:
        lines.append(f"Indice de diversidade: {diversidade:.2f}%")

    # Adiciona outras metricas
    for key, value in metrics.items():
        if key not in [
            "total",
            "top_n",
            "sum_top_n",
            "concentracao_top_n_pct",
            "hhi",
            "diversidade_pct",
        ]:
            if (
                isinstance(value, (int, float))
                and not key.endswith("_col")
                and not key.endswith("_label")
            ):
                if key.endswith("_pct") or "percentual" in key.lower():
                    lines.append(f"{key}: {value:.2f}%")
                else:
                    lines.append(f"{key}: {_format_number(value)}")

    # Se nenhuma metrica especifica foi formatada, usa fallback generico
    if not lines:
        return _format_generic_metrics(metrics)

    return "\n".join(lines)


def _format_generic_metrics(metrics: Dict[str, Any]) -> str:
    """Formata metricas genericas quando chart_type nao e reconhecido."""
    lines = []

    for key, value in metrics.items():
        if isinstance(value, dict) or key.endswith("_col") or key.endswith("_label"):
            continue

        if isinstance(value, (int, float)):
            if key.endswith("_pct") or "percentual" in key.lower():
                lines.append(f"{key}: {value:.2f}%")
            else:
                lines.append(f"{key}: {_format_number(value)}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def _format_metrics_for_prompt(
    numeric_summary: Dict[str, Any], chart_type: str = ""
) -> str:
    """
    Formata metricas numericas para inclusao no prompt COM FORMULAS EXPLICITAS.

    Args:
        numeric_summary: Dicionario com metricas calculadas
        chart_type: Tipo de grafico para formatacao especifica

    Returns:
        String formatada com metricas e formulas para o prompt
    """
    # Detecta chart_type baseado em keys presentes se nao fornecido
    if not chart_type:
        if "sum_top_n" in numeric_summary and "lider_valor" in numeric_summary:
            chart_type = "bar_horizontal"
        elif (
            "valor_inicial" in numeric_summary
            and "variacao_percentual" in numeric_summary
        ):
            chart_type = "line"
        elif (
            "max_valor" in numeric_summary
            and "min_valor" in numeric_summary
            and "amplitude" in numeric_summary
        ):
            chart_type = "bar_vertical"
        elif "hhi" in numeric_summary or "diversidade_pct" in numeric_summary:
            chart_type = "pie"

    # Aplica formatador especifico
    if chart_type == "bar_horizontal":
        return _format_ranking_metrics(numeric_summary)
    elif chart_type in ["line", "line_composed"]:
        return _format_temporal_metrics(numeric_summary)
    elif chart_type in [
        "bar_vertical",
        "bar_vertical_composed",
        "bar_vertical_stacked",
    ]:
        return _format_comparison_metrics(numeric_summary)
    elif chart_type == "pie":
        return _format_distribution_metrics(numeric_summary)
    elif chart_type == "histogram":
        return _format_distribution_metrics(numeric_summary)  # Usa distribuicao
    else:
        return _format_generic_metrics(numeric_summary)


def build_prompt(
    numeric_summary: Dict[str, Any], chart_type: str, filters: Dict[str, Any] = None
) -> str:
    """
    Constrói prompt específico por chart_type.

    Args:
        numeric_summary: Dicionário com metadados calculados pelo calculator
        chart_type: Tipo de gráfico (bar_horizontal, line, etc.)
        filters: Filtros aplicados à análise (opcional)

    Returns:
        String com prompt formatado para a LLM

    Raises:
        ValueError: Se chart_type não for reconhecido
    """
    # Get template for chart type (fallback to bar_horizontal)
    template = CHART_TYPE_TEMPLATES.get(
        chart_type, CHART_TYPE_TEMPLATES["bar_horizontal"]
    )

    # Format metrics with explicit formulas (pass chart_type)
    dados_formatados = _format_metrics_for_prompt(numeric_summary, chart_type)

    # Format filters section if present
    filters_section = _format_filters_for_prompt(filters) if filters else ""

    # Build full prompt with filters context
    prompt = template.format(dados=dados_formatados)

    if filters_section:
        prompt = f"""{prompt}

═══════════════════════════════════════════════════════════════════════════════
FILTROS APLICADOS (OBRIGATÓRIO MENCIONAR EM BOLD NA INTRODUCTION)
═══════════════════════════════════════════════════════════════════════════════

{filters_section}

IMPORTANTE: A introduction DEVE mencionar TODOS estes filtros em **negrito**.
Exemplo: "Esta análise examina ... considerando {filters_section}."
"""

    return prompt


def _format_filters_for_prompt(filters: Dict[str, Any]) -> str:
    """
    Formata filtros para inclusão no prompt de forma legível.

    Args:
        filters: Dicionário de filtros aplicados

    Returns:
        String formatada com filtros
    """
    if not filters:
        return "Nenhum filtro aplicado"

    descriptions = []
    for key, value in filters.items():
        if isinstance(value, list):
            if len(value) == 1:
                descriptions.append(f"**{key}: {value[0]}**")
            else:
                descriptions.append(f"**{key}: {', '.join(map(str, value))}**")
        elif isinstance(value, dict):
            if "between" in value:
                descriptions.append(
                    f"**{key}: entre {value['between'][0]} e {value['between'][1]}**"
                )
            else:
                descriptions.append(f"**{key}: {value}**")
        else:
            descriptions.append(f"**{key}: {value}**")

    return ", ".join(descriptions)
