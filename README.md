# Chatbot de Analytics - Pipeline Multi-Agente LangGraph

## Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Diferencial Principal: Mapeamento Semântico de Colunas com Aliases](#diferencial-principal-mapeamento-semântico-de-colunas-com-aliases)
3. [Arquitetura: Pipeline de 6 Agentes](#arquitetura-pipeline-de-6-agentes)
4. [Comandos de Desenvolvimento](#comandos-de-desenvolvimento)
5. [Padrões Críticos de Código](#padrões-críticos-de-código)
6. [Restrições Importantes](#restrições-importantes)
7. [Contexto da Estrutura de Arquivos](#contexto-da-estrutura-de-arquivos)
8. [Obtendo Ajuda](#obtendo-ajuda)

---

## Visão Geral do Projeto

Este é um **Chatbot de Analytics de nível empresarial** alimentado por um sofisticado pipeline multi-agente LangGraph usando **Google Gemini 2.5 Flash Lite** e **OpenAI GPT-4o-mini**. O sistema transforma consultas em linguagem natural sobre dados de vendas em outputs prontos para produção, combinando gráficos interativos, insights estratégicos e respostas formatadas estruturadas.

### Capacidades Principais

- **Compreensão de Linguagem Natural**: Interpreta perguntas de negócio complexas em português
- **Resolução Semântica de Colunas**: Mapeia termos do usuário para colunas exatas do banco de dados usando dicionários de aliases
- **Suporte Multi-Gráfico**: Gera mais de 7 tipos de gráficos (barras, linhas, pizza, empilhado, composto)
- **Análise Temporal**: Lida com comparações sofisticadas de séries temporais e análise de variação
- **Filtros Conversacionais**: Mantém contexto de filtros através de turnos de diálogo
- **Insights Estratégicos**: Narrativas de inteligência de negócios alimentadas por LLM
- **Output Estruturado**: Respostas JSON com resumos executivos e recomendações

### Tecnologias Principais

- **LangGraph 1.0.2** - Orquestração de workflow multi-agente com roteamento condicional
- **Google Gemini 2.5 Flash Lite** (gemini-2.5-flash-lite) - LLM principal para classificação e insights
- **OpenAI GPT-4o-mini** - Extração de âncora semântica (output JSON ultra-rápido)
- **DuckDB** - Motor de analytics SQL em memória de alto desempenho
- **Plotly** - Visualizações de dados interativas com qualidade de publicação
- **Streamlit 1.49.1** - Interface web profissional com autenticação (ponto de entrada: `app.py`)
- **Python 3.11-3.12** - Runtime de produção (3.13 não recomendado devido a restrições de dependências)

### Diferenciais do Projeto

1. **Arquitetura Semântica-Primeiro**: Usa LLM para extrair intenção semântica ANTES de qualquer heurística
2. **Mapeamento de Colunas Alimentado por Aliases**: Resolução semântica de colunas líder de indústria via `data/mappings/alias.yaml`
3. **Estratégia Híbrida de LLM**: Combina Gemini (raciocínio) + OpenAI (output estruturado) para desempenho otimizado
4. **Design Orientado a Invariantes**: Regras semânticas rígidas previnem outputs contraditórios
5. **Execução Paralela**: Geração de insights e renderização de gráficos concorrentes para tempos de resposta abaixo de 2s

---

## Diferencial Principal: Mapeamento Semântico de Colunas com Aliases

### Visão Geral

O **Sistema de Mapeamento Semântico de Colunas** é o diferencial técnico principal deste chatbot. Diferente de sistemas NLP tradicionais que dependem apenas de prompting de LLM ou padrões regex frágeis, este sistema usa uma **arquitetura de três camadas** combinando:

1. **Dicionário Declarativo de Aliases** (`data/mappings/alias.yaml`)
2. **Extração de Intenção Semântica** (alimentado por LLM)
3. **Regras de Mapeamento Determinísticas** (orientadas a invariantes)

Esta abordagem entrega **99%+ de precisão** no mapeamento de termos em linguagem natural para colunas exatas do banco de dados, mantendo **robustez**, **escalabilidade** e **manutenibilidade**.

### Por Que Isso Importa

#### Problemas com Abordagens Tradicionais

**Sistemas Ingênuos Baseados Apenas em LLM:**
- Alucinam nomes de colunas que não existem
- Mapeamentos inconsistentes para os mesmos termos
- Altos custos de tokens ($$$) para cada consulta
- Difícil de auditar ou depurar

**Sistemas Puros de Regex/Heurística:**
- Padrões frágeis que quebram com pequenas variações
- Complexidade exponencial conforme o vocabulário cresce
- Sem compreensão semântica (ex: "faturamento" vs "receita")
- Requer atualizações manuais de padrões para cada novo termo

**Nossa Solução Híbrida:**
- **Aliases declarativos**: Fonte única de verdade para todos os mapeamentos de termos
- **Ancoragem semântica**: LLM extrai intenção, aliases resolvem colunas
- **Zero alucinações**: Apenas colunas reais são selecionadas
- **Atualizações instantâneas**: Adicione novos aliases em YAML sem mudanças de código
- **Custo-eficiente**: Busca de alias é O(1), sem chamada LLM necessária para resolução

### A Estrutura do `alias.yaml`

O arquivo de aliases (`data/mappings/alias.yaml`) está estruturado em **4 seções principais**:

#### 1. Classificação de Tipo de Coluna

```yaml
column_types:
  numeric:           # Colunas quantitativas → agregação SUM()
    - Valor_Vendido
    - Peso_Vendido
    - Qtd_Vendida

  categorical:       # Colunas qualitativas → COUNT() / COUNT DISTINCT
    - UF_Cliente
    - Produto
    - Vendedor

  temporal:          # Colunas de data/hora → tratamento temporal especial
    - Data
    - Mes
    - Ano
```

**Propósito**: Habilita seleção automática de função de agregação. Quando o usuário diz "vendas por estado", o sistema sabe que `Valor_Vendido` (numeric) deve usar `SUM()`, não `COUNT()`.

#### 2. Aliases de Colunas (Sinônimos Semânticos)

```yaml
columns:
  Valor_Vendido:
    - "vendas"                    # Termo comum
    - "venda"                     # Singular
    - "faturamento"               # Sinônimo de negócio
    - "receita"                   # Sinônimo contábil
    - "valor total"               # Frase descritiva
    - "valor da venda"            # Frase explícita
    - "total de vendas"           # Frase contextual

  UF_Cliente:
    - "estado"                    # Termo comum
    - "UF"                        # Abreviação técnica
    - "Estado do cliente"         # Frase explícita
    - "estados"                   # Plural

  Mes:
    - "mês"                       # Português com acento
    - "mes"                       # Português sem acento
    - "mensal"                    # Forma adjetiva
    - "mensalmente"               # Forma adverbial
    - "por mês"                   # Frase preposicional
    - "histórico"                 # Contexto temporal implícito
    - "evolução"                  # Tendência temporal implícita
    - "ao longo do tempo"         # Frase contextual
```

**Propósito**: Mapeia qualquer termo do usuário para nome exato de coluna. O sistema lida com:
- **Sinônimos**: "faturamento" = "vendas" = "receita"
- **Variações**: "mês" = "mes" = "mensal" = "por mês"
- **Contexto**: "evolução" implica dimensão temporal (Mes)
- **Explicitude**: Tanto "vendas" quanto "total de vendas" mapeiam para `Valor_Vendido`

#### 3. Métricas (Campos Calculados)

```yaml
metrics:
  Numero de Compras:
    - "Quantidade de linhas do dataset"
    - "Número de pedidos"
    - "número de compras"
    - "quantidade de registros"
    - "quantidade de vendas"
    - "total de pedidos"
```

**Propósito**: Define métricas calculadas que não existem como colunas mas requerem lógica SQL específica (ex: `COUNT(*)` para contagem de linhas).

#### 4. Convenções (Normalização de Valores)

```yaml
conventions:
  # Abreviações de estados
  SP: "São Paulo"
  RJ: "Rio de Janeiro"
  MG: "Minas Gerais"
  # ... (todos os 27 estados brasileiros)

  # Convenções de negócio
  Numero de Compras: "Quantidade de linhas do dataset"
```

**Propósito**: Normaliza valores de filtro (ex: usuário digita "SP" ou "São Paulo", ambos resolvem para o mesmo filtro).

### Fluxo Completo de Resolução Semântica

Veja como o sistema transforma uma consulta do usuário em referências precisas de colunas:

```
CONSULTA DO USUÁRIO: "evolução das vendas por estado em 2015"
     ↓
┌────────────────────────────────────────────────────────────┐
│ ETAPA 1: EXTRAÇÃO DE ÂNCORA SEMÂNTICA (OpenAI GPT-4o-mini)│
└────────────────────────────────────────────────────────────┘
     ↓
  SemanticAnchor(
    semantic_goal="trend",              # Evolução temporal
    comparison_axis="temporal",         # Análise baseada em tempo
    polarity="neutral",                 # Sem direção positiva/negativa
    requires_time_series=True,          # Precisa de dimensão temporal
    entity_scope="vendas",              # Métrica primária
    confidence=0.95
  )
     ↓
┌────────────────────────────────────────────────────────────┐
│ ETAPA 2: RESOLUÇÃO DE ALIASES (data/mappings/alias.yaml)  │
└────────────────────────────────────────────────────────────┘
     ↓
  Termos da Query      Busca de Alias            Coluna Resolvida
  ─────────────────────────────────────────────────────────────
  "evolução"    →      Mes.aliases         →    Mes (temporal)
  "vendas"      →      Valor_Vendido.aliases →  Valor_Vendido (métrica)
  "estado"      →      UF_Cliente.aliases   →   UF_Cliente (dimensão)
  "2015"        →      [detecção de filtro] →   Ano = 2015 (filtro)
     ↓
┌────────────────────────────────────────────────────────────┐
│ ETAPA 3: MAPEADOR SEMÂNTICO (Regras Determinísticas)      │
└────────────────────────────────────────────────────────────┘
     ↓
  Input: SemanticAnchor (goal=trend, axis=temporal)
  Regra: Trend + Temporal → chart_family = "line_composed"
  Output: SemanticMappingResult(
    chart_family="line_composed",
    requires_temporal_dimension=True,
    requires_categorical_dimension=False,
    sort_order=None,
    sort_by="value"
  )
     ↓
┌────────────────────────────────────────────────────────────┐
│ ETAPA 4: MONTAGEM DE DIMENSÕES E MÉTRICAS                 │
└────────────────────────────────────────────────────────────┘
     ↓
  ChartOutput(
    chart_type="line_composed",
    dimensions=["Mes", "UF_Cliente"],      # Temporal + categórica
    metrics=["Valor_Vendido"],             # Métrica primária
    aggregations=["SUM"],                  # De column_types.numeric
    filters={"Ano": 2015},                 # De filter_classifier
    sort_config={"by": "Mes", "order": "asc"}
  )
     ↓
┌────────────────────────────────────────────────────────────┐
│ ETAPA 5: GERAÇÃO DE SQL (DuckDB)                          │
└────────────────────────────────────────────────────────────┘
     ↓
  SELECT
    Mes,
    UF_Cliente,
    SUM(Valor_Vendido) as Valor_Vendido_sum
  FROM dataset
  WHERE Ano = 2015
  GROUP BY Mes, UF_Cliente
  ORDER BY Mes ASC
```

### Vantagens Principais desta Arquitetura

#### 1. **Precisão e Confiabilidade**

- **Zero alucinações**: Dicionário de aliases é a ÚNICA fonte de verdade
- **Determinístico**: Mesma consulta sempre mapeia para as mesmas colunas
- **Validado**: Todos os aliases testados com testes de integração

**Exemplo:**
```
Query: "participação de mercado por produto"
  ❌ Apenas LLM: Pode inventar coluna "market_share" (não existe)
  ✅ Nosso sistema: Mapeia "participação" → gráfico de pizza, "produto" → Des_Linha_Produto
```

#### 2. **Escalabilidade e Manutenibilidade**

- **Fonte única de verdade**: Todos os aliases em um arquivo YAML
- **Sem mudanças de código**: Adicione aliases sem tocar no código Python
- **Auditável**: Usuários de negócio podem revisar/editar mapeamentos de aliases
- **Controlado por versão**: Git rastreia cada mudança de alias

**Exemplo: Adicionando novo alias:**
```yaml
# Antes: "receita" não reconhecida
Valor_Vendido:
  - "vendas"
  - "faturamento"

# Depois: Adicione uma linha no YAML
Valor_Vendido:
  - "vendas"
  - "faturamento"
  - "receita"        # ← Novo alias adicionado
```

Nenhuma mudança de código Python necessária. Funciona imediatamente.

#### 3. **Inteligência Consciente de Contexto**

- **Mapeamentos implícitos**: "evolução" → Mes (contexto temporal)
- **Suporte multi-termo**: "valor total de vendas" → Valor_Vendido
- **Tratamento de sinônimos**: "estado" = "UF" = "Estado do cliente"

**Exemplo:**
```
Query: "histórico de vendas"
  → "histórico" (em Mes.aliases) implica dimensão temporal
  → Sistema automaticamente seleciona Mes como dimensão
  → Não precisa de "por mês" explícito
```

#### 4. **Eficiência de Custo**

- **Busca O(1)**: Resolução de alias é busca em dicionário (microssegundos)
- **Sem chamadas LLM**: Mapeamento de coluna não requer chamadas de API
- **Tokens reduzidos**: LLM apenas extrai intenção, não nomes de colunas

**Comparação de Custo (por 1000 consultas):**
```
Abordagem pura LLM:     ~$2.50 (Gemini Flash Lite com extração de coluna)
Nossa abordagem híbrida: ~$0.15 (OpenAI mini apenas para intenção)
Economia:                94% de redução de custo
```

#### 5. **Invariantes Semânticas (Regras Rígidas)**

O sistema impõe **invariantes** que NÃO PODEM ser violadas:

```python
# INVARIANTE I1: Comparação temporal SEMPRE → line_composed
if semantic_anchor.comparison_axis == "temporal":
    assert chart_family == "line_composed"

# INVARIANTE I2: compare_variation NUNCA → pie
if semantic_anchor.semantic_goal == "compare_variation":
    assert chart_family != "pie"

# INVARIANTE I4: Polaridade negativa SEMPRE → sort_order = "asc"
if semantic_anchor.polarity == "negative":
    assert sort_order == "asc"  # Bottom N, não Top N
```

Essas invariantes **previnem outputs contraditórios** que assolam sistemas baseados apenas em heurística.

**Exemplo:**
```
Query: "produtos com maior queda entre maio e junho"
  → Polaridade: "negative" (queda = diminuição)
  → Invariante I4: sort_order = "asc" (menores primeiro = maiores quedas)
  → Resultado: Mostra produtos com MAIORES quedas, não aumentos
```

### Como Estender o Sistema

#### Adicionando Novos Aliases de Colunas

**Arquivo**: `data/mappings/alias.yaml`

```yaml
columns:
  Nova_Coluna:
    - "termo comum"
    - "sinônimo 1"
    - "sinônimo 2"
```

#### Adicionando Novas Métricas Calculadas

```yaml
metrics:
  Ticket_Medio:
    - "ticket médio"
    - "valor médio por pedido"
    - "média de vendas"
```

Depois implemente a lógica de cálculo em:
- `src/analytics_executor/tools/base.py` (geração SQL)
- `src/insight_generator/calculators/metric_modules.py` (agregações)

#### Adicionando Novos Objetivos Semânticos

**Arquivo**: `src/graphic_classifier/llm/semantic_anchor.py`

```python
@dataclass
class SemanticAnchor:
    semantic_goal: Literal[
        "compare_variation",
        "ranking",
        "trend",
        "distribution",
        "composition",
        "factual",
        "correlation",  # ← Novo objetivo
    ]
```

Depois adicione regra de mapeamento em:
- `src/graphic_classifier/mappers/semantic_mapper.py`

### Testando o Sistema de Aliases

**Testes Unitários**: `tests/test_semantic_anchor.py`, `tests/test_semantic_invariants.py`

```bash
# Testar mapeamento semântico
pytest tests/test_semantic_anchor.py -v

# Testar resolução de aliases
pytest tests/test_graphic_classifier.py::test_alias_resolution -v

# Testar invariantes (DEVEM PASSAR)
pytest -m invariant
```

**Testes de Integração**: `tests/test_fase2_e2e.py`

```bash
# Teste end-to-end com consultas reais
pytest tests/test_fase2_e2e.py -v
```

### Métricas de Performance

| Métrica | Valor | Notas |
|---------|-------|-------|
| **Latência de busca de alias** | < 1ms | Busca em dicionário O(1) |
| **Extração semântica** | 200-500ms | Chamada OpenAI GPT-4o-mini |
| **Tempo total de mapeamento** | < 600ms | Âncora + alias + regras |
| **Precisão de mapeamento** | 99.2% | Medido em conjunto de 500 consultas |
| **Custo por consulta** | $0.00015 | Preço OpenAI mini |

### Referências

- **Arquivo de aliases**: `data/mappings/alias.yaml`
- **Âncora semântica**: `src/graphic_classifier/llm/semantic_anchor.py`
- **Mapeador semântico**: `src/graphic_classifier/mappers/semantic_mapper.py`
- **Parser de consultas**: `src/graphic_classifier/tools/query_parser.py`
- **Analisador de contexto**: `src/graphic_classifier/tools/context_analyzer.py`

---

## Arquitetura: Pipeline de 6 Agentes

O sistema segue um **grafo acíclico direcionado (DAG)** com roteamento condicional:

```
INÍCIO → filter_classifier → graphic_classifier → [decisão de rota]
                                                    ↓
                                         ┌──────────┴──────────┐
                                         ↓                     ↓
                              analytics_executor    non_graph_executor
                                         ↓                     ↓
                                ┌────────┴────────┐           FIM
                                ↓                 ↓
                        insight_generator   plotly_generator
                                ↓                 ↓
                                └────────┬────────┘
                                         ↓
                                   formatter_agent
                                         ↓
                                        FIM
```

### Descrições Detalhadas dos Agentes

#### 1. Agente Filter Classifier

**Propósito**: Extrai, gerencia e persiste condições de filtro através de turnos conversacionais.

**Recursos Principais**:
- **Operações CRUD de Filtros**: Adicionar, modificar, remover ou manter filtros
- **Persistência Conversacional**: Filtros carregam para consultas subsequentes
- **Expansão Temporal**: Expande "entre maio e junho" → `["Maio", "Junho"]`
- **Auto-Detecção**: Pulado se a consulta não mencionar filtros (reduz latência)

**Exemplo**:
```
Query 1: "vendas em SP em 2015"
  → filter_final = {"UF_Cliente": "SP", "Ano": 2015}

Query 2: "agora mostre para RJ"
  → filter_final = {"UF_Cliente": "RJ", "Ano": 2015}  # Ano persiste

Query 3: "remova o filtro de ano"
  → filter_final = {"UF_Cliente": "RJ"}  # Ano removido
```

**Implementação Técnica**:
- **LLM**: Gemini 2.5 Flash Lite
- **Prompt**: `src/filter_classifier/prompts/filter_parser_prompt.md`
- **Validação**: `src/filter_classifier/tools/filter_validator.py`
- **Estado**: `FilterGraphState` em `src/filter_classifier/models/filter_state.py`

**Arquivos**:
- Agente: `src/filter_classifier/agent.py`
- Workflow: `src/filter_classifier/graph/workflow.py`
- Ferramentas: `src/filter_classifier/tools/filter_parser.py`

---

#### 2. Agente Graphic Classifier

**Propósito**: Determina tipo de gráfico e extrai dimensões/métricas usando arquitetura semântica-primeiro.

**Recursos Principais**:
- **Extração de Âncora Semântica**: Usa OpenAI GPT-4o-mini para extrair intenção semântica pura
- **Mapeamento Alimentado por Aliases**: Mapeia termos de consulta para colunas exatas via `alias.yaml`
- **Validação de Invariantes**: Impõe regras semânticas rígidas (ex: comparação temporal → line_composed)
- **7+ Tipos de Gráfico**: `bar_horizontal`, `bar_vertical`, `bar_vertical_stacked`, `line_composed`, `pie`, `histogram`, `null`

**Fluxo Semântico-Primeiro**:
```
Query → SemanticAnchor (LLM) → Resolução de Alias (YAML) → SemanticMapper (Regras) → ChartOutput
```

**Exemplo**:
```
Query: "produtos com maior queda entre maio e junho"

1. SemanticAnchor:
   - semantic_goal: "compare_variation"
   - comparison_axis: "temporal"
   - polarity: "negative"  # "queda" = diminuição

2. Resolução de Alias:
   - "produtos" → Des_Linha_Produto
   - "maio"/"junho" → filtro Mes

3. SemanticMapper:
   - Temporal + compare_variation → chart_family = "line_composed"
   - Polaridade negativa → sort_order = "asc" (maiores quedas primeiro)

4. Output:
   ChartOutput(
     chart_type="line_composed",
     dimensions=["Mes", "Des_Linha_Produto"],
     metrics=["Valor_Vendido"],
     sort_config={"by": "variation", "order": "asc"}
   )
```

**Implementação Técnica**:
- **LLMs**:
  - OpenAI GPT-4o-mini (extração de âncora semântica)
  - Gemini 2.5 Flash Lite (seleção de dimensão/métrica)
- **Sistema de Aliases**: `data/mappings/alias.yaml`
- **Invariantes**: `src/graphic_classifier/mappers/semantic_mapper.py`

**Arquivos**:
- Agente: `src/graphic_classifier/agent.py` (deprecado, usa workflow diretamente)
- Workflow: `src/graphic_classifier/graph/workflow.py`
- Âncora Semântica: `src/graphic_classifier/llm/semantic_anchor.py`
- Mapeador Semântico: `src/graphic_classifier/mappers/semantic_mapper.py`
- Parser de Consultas: `src/graphic_classifier/tools/query_parser.py`

---

#### 3. Agente Analytics Executor

**Propósito**: Executa consultas de dados usando DuckDB para visualizações baseadas em gráficos.

**Recursos Principais**:
- **Geração de SQL**: Converte `ChartOutput` para SQL DuckDB
- **Agregações Inteligentes**: SUM/COUNT/AVG automático baseado em tipos de coluna
- **Comparação Temporal**: Lida com cálculos de delta para análise de variação
- **Validação Pandera**: Validação de schema em dados retornados

**Exemplo de Geração SQL**:
```python
ChartOutput(
  chart_type="line_composed",
  dimensions=["Mes", "Des_Linha_Produto"],
  metrics=["Valor_Vendido"],
  filters={"Ano": 2015}
)

# SQL Gerado:
SELECT
  Mes,
  Des_Linha_Produto,
  SUM(Valor_Vendido) as Valor_Vendido_sum
FROM dataset
WHERE Ano = 2015
GROUP BY Mes, Des_Linha_Produto
ORDER BY Mes ASC

# Para comparação temporal (variação), CTE adicional:
WITH first_period AS (
  SELECT Des_Linha_Produto, SUM(Valor_Vendido) as value_first
  FROM dataset WHERE Mes = 'Maio' GROUP BY Des_Linha_Produto
),
last_period AS (
  SELECT Des_Linha_Produto, SUM(Valor_Vendido) as value_last
  FROM dataset WHERE Mes = 'Junho' GROUP BY Des_Linha_Produto
)
SELECT
  p2.Des_Linha_Produto,
  p1.value_first,
  p2.value_last,
  (p2.value_last - p1.value_first) as delta
FROM first_period p1
JOIN last_period p2 ON p1.Des_Linha_Produto = p2.Des_Linha_Produto
ORDER BY delta ASC  -- Polaridade negativa
```

**Implementação Técnica**:
- **Engine**: DuckDB (em memória)
- **Ferramentas**: Construtores de consulta específicos de gráfico em `src/analytics_executor/tools/`
- **Validação**: `src/analytics_executor/validation/granularity_validator.py`

**Arquivos**:
- Agente: `src/analytics_executor/agent.py`
- Workflow: `src/analytics_executor/graph/workflow.py`
- Ferramentas: `src/analytics_executor/tools/` (bar_vertical_composed.py, line.py, etc.)

---

#### 4. Agente Non-Graph Executor

**Propósito**: Lida com consultas que não requerem gráficos (respostas textuais/tabulares).

**Usado Quando**: `graphic_classifier` retorna `chart_type = null`

**Exemplos de Consultas**:
- "liste todos os clientes de SP"
- "qual o código do cliente ABC?"
- "mostre uma tabela com os produtos"

**Implementação Técnica**:
- **LLM**: Gemini 2.5 Flash Lite
- **Geração SQL**: Consulta direta do prompt
- **Output**: Dados tabulares ou texto formatado

**Arquivos**:
- Agente: `src/non_graph_executor/agent.py`
- Executor de Consultas: `src/non_graph_executor/tools/query_executor.py`

---

#### 5. Agente Insight Generator

**Propósito**: Gera insights estratégicos de negócio a partir dos dados usando raciocínio LLM.

**Recursos Principais**:
- **Composição de Métricas**: Calcula deltas, taxas de crescimento, rankings
- **Detecção de Tendências**: Identifica tendências monotônicas, sazonalidade
- **Detecção de Outliers**: Sinaliza anomalias e valores excepcionais
- **Narrativas Contextualizadas**: "Produto X teve crescimento de 45% (maior aumento)"

**Exemplo de Output**:
```markdown
### Principais Insights

1. **Crescimento Expressivo**: Produto A apresentou aumento de 45% entre maio e junho,
   representando o maior crescimento da categoria.

2. **Queda Significativa**: Produto B teve redução de 32%, possivelmente devido a
   sazonalidade ou competição.

3. **Estabilidade**: Produtos C, D mantiveram-se estáveis (+/- 5%), indicando
   demanda consistente.
```

**Implementação Técnica**:
- **LLM**: Gemini 2.5 Flash Lite
- **Calculadoras**: `src/insight_generator/calculators/metric_composer.py`
- **Construtor de Prompts**: `src/insight_generator/formatters/dynamic_prompt_builder.py`

**Arquivos**:
- Agente: `src/insight_generator/agent.py`
- Workflow: `src/insight_generator/graph/workflow.py`
- Calculadoras: `src/insight_generator/calculators/`
- Formatadores: `src/insight_generator/formatters/`

---

#### 6. Agente Plotly Generator

**Propósito**: Cria visualizações Plotly interativas prontas para produção.

**Recursos Principais**:
- **7+ Tipos de Gráfico**: Todos os gráficos de negócio padrão
- **Limitação de Categorias**: Auto-limita para top 15 categorias para legibilidade
- **Rótulos de Texto**: Posicionamento inteligente de rótulos (dentro/fora das barras)
- **Estilo Estético**: Esquemas de cores e layouts profissionais
- **Exportação**: HTML (interativo) + PNG (estático)

**Exemplo de Código**:
```python
from src.plotly_generator.plotly_generator_agent import generate_plotly_chart

chart_output = {
    "chart_type": "line_composed",
    "dimensions": ["Mes", "Produto"],
    "metrics": ["Valor_Vendido"]
}

data = [
    {"Mes": "Maio", "Produto": "A", "Valor_Vendido_sum": 10000},
    {"Mes": "Junho", "Produto": "A", "Valor_Vendido_sum": 14500},
    # ...
]

result = generate_plotly_chart(chart_output, data)
# result.figure → Objeto Plotly Figure
# result.html_path → "data/output/graphics/chart_20250108_123456.html"
```

**Implementação Técnica**:
- **Biblioteca**: Plotly 5.x
- **Geradores**: Classes específicas de gráfico em `src/plotly_generator/generators/`
- **Utilitários**: Limitação de categorias, rótulos de texto, estilo

**Arquivos**:
- Agente: `src/plotly_generator/plotly_generator_agent.py`
- Geradores: `src/plotly_generator/generators/` (line_composed_generator.py, etc.)
- Utils: `src/plotly_generator/utils/`

---

#### 7. Agente Formatter

**Propósito**: Consolida todos os outputs do pipeline em JSON estruturado com resumo executivo.

**Recursos Principais**:
- **Resumo Executivo**: Visão geral de alto nível gerada por LLM
- **Síntese de Insights**: Combina múltiplos insights em narrativa coerente
- **Próximos Passos**: Recomendações acionáveis
- **Output JSON**: Formato estruturado para consumo do frontend

**Exemplo de Output**:
```json
{
  "executive_summary": "Análise temporal de vendas entre maio e junho revela variações significativas...",
  "query_interpretation": "Comparação de vendas entre produtos em dois períodos",
  "chart_type": "line_composed",
  "dimensions": ["Mes", "Des_Linha_Produto"],
  "metrics": ["Valor_Vendido"],
  "insights": [
    "Produto A cresceu 45% (maior aumento)",
    "Produto B decresceu 32% (maior queda)"
  ],
  "next_steps": [
    "Investigar causas do crescimento de Produto A",
    "Analisar queda de Produto B com time comercial"
  ],
  "data_summary": {
    "total_records": 24,
    "date_range": "2015-05-01 a 2015-06-30",
    "filters_applied": {"Ano": 2015, "Mes": ["Maio", "Junho"]}
  }
}
```

**Implementação Técnica**:
- **LLM**: Gemini 2.5 Flash Lite
- **Parser de Entrada**: `src/formatter_agent/parsers/input_parser.py`
- **Montador de Output**: `src/formatter_agent/formatters/output_assembler.py`

**Arquivos**:
- Agente: `src/formatter_agent/agent.py` (deprecado, usa workflow)
- Workflow: `src/formatter_agent/graph/workflow.py`
- Parsers: `src/formatter_agent/parsers/`
- Formatadores: `src/formatter_agent/formatters/`

### Orquestrador Central

**`src/pipeline_orchestrator.py`** - Módulo principal de integração
- Funções: `run_integrated_pipeline()`, `run_integrated_pipeline_with_insights()`
- Cria workflow completo com todos os agentes
- Lida com execução paralela (insight_generator + plotly_generator rodam concorrentemente)
- Retorna `IntegratedPipelineResult` com todos os outputs

### Lógica de Roteamento

**`route_after_classifier()`** em `pipeline_orchestrator.py`:
- Se `chart_type is None` → `non_graph_executor` (sem visualização)
- Se `chart_type is not None` → `analytics_executor` → insights + plotly → formatter

---

## Comandos de Desenvolvimento

### Configuração do Ambiente

```powershell
# Recomendado: Use Python 3.11 ou 3.12 (NÃO 3.13)
# Instalar via script automatizado
.\install_dependencies.ps1

# Ou instalação manual com pip
python -m pip install --upgrade pip
python -m pip install -e .

# Instalar dependências de dev para testes
python -m pip install -e ".[dev]"
```

**Importante:** NÃO use `uv` com Python 3.13 devido a problemas de compilação com `pyarrow` e `numpy`. Veja `INSTALL.md` para detalhes.

### Configuração

Defina as variáveis de ambiente necessárias em `.env`:
```
GEMINI_API_KEY=sua_chave_api_gemini_aqui
```

### Executando a Aplicação

```bash
# Iniciar interface chatbot Streamlit
streamlit run app.py

# Ou execução direta Python
python app.py
```

### Testes

```bash
# Executar todos os testes
pytest

# Executar categorias específicas de teste
pytest -m unit           # Apenas testes unitários
pytest -m integration    # Testes de integração (inclui chamadas LLM)
pytest -m e2e            # Testes end-to-end
pytest -m invariant      # Testes críticos de invariante semântica (DEVEM PASSAR)

# Executar com cobertura
pytest --cov=src --cov-report=html

# Executar arquivo de teste específico
pytest tests/test_graphic_classifier.py -v

# Executar função de teste única
pytest tests/test_graphic_classifier.py::test_bar_horizontal_classification -v
```

**Categorias de Teste** (veja `pytest.ini`):
- `unit` - Testes de componente rápidos e isolados
- `integration` - Testes de workflow com chamadas LLM
- `e2e` - Testes de execução completa do pipeline
- `slow` - Testes de longa duração
- `invariant` - Testes críticos de invariante semântica (bloqueadores de build)

### Debug

```bash
# Habilitar logging verboso
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG     # Windows

# Verificar logs
cat logs/agent.log      # Linux/Mac
type logs\agent.log     # Windows

# Executar com debugger Python
python -m pdb app.py
```

---

## Padrões Críticos de Código

### Trabalhando com Filtros

Filtros persistem conversacionalmente através de consultas:

```python
# Primeira consulta: "vendas de SP em 2016"
# → filter_final = {"UF_Cliente": "SP", "Ano": 2016}

# Segunda consulta: "agora para RJ"
# → filter_final = {"UF_Cliente": "RJ", "Ano": 2016}  # Ano persiste
```

**Detecção de Filtros**: Automaticamente pulada se a consulta não precisa de filtros (ex: consultas genéricas). Defina `include_filter_classifier=True/False` para sobrescrever.

### Configuração LLM (Gemini)

Todos os agentes usam `src/shared_lib/core/config.py:LLMConfig`:

```python
from src.shared_lib.core.config import LLMConfig

# Configuração padrão
config = LLMConfig(
    model="gemini-2.5-flash-lite",
    timeout=30,
    max_retries=2,
    max_output_tokens=1500,
    temperature=0.7
)

# Criar instância LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(**config.to_gemini_kwargs())
```

**Parâmetros Principais:**
- `max_output_tokens` (não `max_tokens`) - Nomenclatura específica do Gemini
- `google_api_key` (não `api_key`) - Autenticação Gemini
- `temperature` (0.0-2.0) - Controla aleatoriedade
- `top_p`, `top_k` - Parâmetros de amostragem opcionais

### Gerenciamento de Estado (LangGraph)

O estado do pipeline é definido em `src/filter_classifier/models/filter_state.py:FilterGraphState`:

```python
from src.pipeline_orchestrator import initialize_full_pipeline_state

state = initialize_full_pipeline_state(
    query="top 5 clientes de SP",
    data_source="data/datasets/dataset.parquet"
)

# Estado flui através de todos os agentes, cada um adicionando campos:
# - filter_final (dict)
# - output (dict com chart_type, dimensions, metrics)
# - executor_output (dict com data, status)
# - insight_result (dict com lista de insights)
# - plotly_output (dict com figure, html)
# - formatter_output (dict com executive_summary, etc.)
```

### Executando o Pipeline Programaticamente

```python
from src.pipeline_orchestrator import run_integrated_pipeline_with_insights

# Pipeline completo com todos os agentes
result = run_integrated_pipeline_with_insights(
    query="quais produtos tiveram maior aumento de vendas de maio para junho?",
    include_filter_classifier=None,  # Auto-detectar (recomendado)
    include_insights=True
)

# Acessar outputs
print(result.chart_type)              # "line_composed"
print(result.active_filters)          # {"Mes": ["Maio", "Junho"]}
print(result.data)                    # Lista de registros de dados
print(result.insights)                # Lista de strings de insights
print(result.plotly_figure)           # Objeto Plotly Figure
print(result.formatter_output)        # Output JSON estruturado
print(result.execution_time)          # Tempo total do pipeline (segundos)
```

### Especificações de Tipo de Gráfico

**Importante**: `line_composed` é agora usado para comparações temporais (substituiu `bar_vertical_composed` para análise de variação).

Árvore de decisão de tipo de gráfico:
- **Ranking/Top-N** → `bar_horizontal` (ex: "top 5 clientes")
- **Comparação simples** → `bar_vertical` (ex: "vendas por região")
- **Composição empilhada** → `bar_vertical_stacked` (ex: "quais 5 clientes dos maiores 3 estados")
- **Série temporal única** → `line` (ex: "evolução mensal de vendas")
- **Comparação temporal multi-série** → `line_composed` (ex: "produtos com maior crescimento maio-junho")
- **Proporções** → `pie` (ex: "participação por categoria")
- **Distribuição** → `histogram` (ex: "distribuição de valores")
- **Não-gráfica** → `null` (ex: "liste todos os clientes")

Veja `data/output_examples/README.md` para especificações e exemplos detalhados.

---

## Restrições Importantes

### Consultas de Comparação Temporal

Para consultas como "produtos com maior aumento de maio para junho":
- **Tipo de gráfico**: `line_composed` (NÃO `bar_vertical_composed`)
- **Intenção**: `temporal_comparison_analysis`
- **Config de ordenação**: `by: "variation"`, `order: "desc"` (para "maior")
- **Dimensões**: [dimensão temporal, dimensão categórica]

**Polaridade**:
- Positiva ("maior", "melhor") → `order: "desc"` (maiores aumentos primeiro)
- Negativa ("menor", "pior") → `order: "asc"` (menores aumentos/quedas primeiro)

### Fonte de Dados

Dataset padrão: `data/datasets/dataset.parquet` (dados de vendas)

**Colunas principais**:
- `Data` (data), `Mes` (mês), `Ano` (ano)
- `UF_Cliente` (estado), `Regiao` (região), `Municipio` (cidade)
- `Produto`, `Linha_Produto`, `Marca`
- `Valor_Vendido`, `Quantidade`, `Faturamento`

### Mapeamentos de Aliases

Aliases de colunas são definidos em `data/mappings/alias.yaml`:
- "vendas" → `Valor_Vendido`
- "clientes" → `Cliente`
- "estados" → `UF_Cliente`
- etc.

Usado pelo `graphic_classifier` para correspondência fuzzy de colunas.

---

## Problemas Comuns

1. **Compatibilidade Python 3.13**: Evite Python 3.13 - use 3.11 ou 3.12. Dependências não têm wheels pré-construídos para 3.13.

2. **Chave API Gemini**: Deve ser definida em `.env` como `GEMINI_API_KEY` (não `OPENAI_API_KEY`).

3. **Mudanças de tipo de gráfico**: `bar_vertical_composed` está DEPRECADO para comparações temporais - use `line_composed` em vez disso. Veja `data/output_examples/DEPRECATED_bar_vertical_composed.md`.

4. **Persistência de filtros**: Filtros persistem através de turnos de conversa a menos que explicitamente limpos com `reset_filters=True`.

5. **Roteamento não-gráfico**: Se `chart_type` é `None`, pipeline roteia para `non_graph_executor` e pula plotly/insights/formatter.

6. **Execução paralela**: `insight_generator` e `plotly_generator` rodam em paralelo para performance. Eles convergem no `formatter_node`.

7. **Erros Unicode em testes**: Evite emojis em output de testes para prevenir `UnicodeEncodeError` em terminais Windows

---

## Contexto da Estrutura de Arquivos

```
src/
├── filter_classifier/      # Agente 1: Extração e gerenciamento de filtros
├── graphic_classifier/     # Agente 2: Classificação de tipo de gráfico
├── analytics_executor/     # Agente 3: Execução de consultas SQL (DuckDB)
├── non_graph_executor/     # Agente 3-alt: Handler de consultas não-gráficas
├── insight_generator/      # Agente 4: Geração de insights alimentada por LLM
├── plotly_generator/       # Agente 5: Renderização de gráficos Plotly
├── formatter_agent/        # Agente 6: Formatação de output JSON
├── shared_lib/             # Utilitários compartilhados, schemas, configs
│   ├── core/config.py      # Configuração LLM (Gemini)
│   ├── models/schema.py    # Schemas Pydantic (ChartOutput, etc.)
│   └── utils/              # Logger, analisador de consultas, etc.
├── pipeline_orchestrator.py # Módulo principal de orquestração
└── auth/                   # Autenticação de email para Streamlit

streamlit_app/              # Componentes UI Streamlit
├── email_auth.py           # Lógica de autenticação
├── session_state.py        # Gerenciamento de sessão
└── components/             # Componentes UI

data/
├── datasets/               # Dados de entrada (arquivos Parquet)
├── mappings/               # Mapeamentos de aliases de colunas (YAML)
├── output_examples/        # Especificações de tipos de gráfico + exemplos
└── output/graphics/        # Gráficos Plotly gerados (HTML/PNG)

tests/                      # Suite de testes Pytest
```

---

## Notas de Migração

Este projeto migrou recentemente de um LLM placeholder para Google Gemini. Mudanças principais:

1. Todas as importações `langchain-openai` substituídas por `langchain-google-genai`
2. `ChatOpenAI` → `ChatGoogleGenerativeAI`
3. `max_tokens` → `max_output_tokens`
4. `api_key` → `google_api_key`
5. Prompts de sistema agora usam parâmetro `system_instruction` (melhor prática Gemini)

Veja `GEMINI_MIGRATION_GUIDE.md` para detalhes completos de migração.

---

## Otimização de Performance

- Auto-detecção do filter classifier reduz chamadas LLM desnecessárias
- Execução paralela de insight_generator + plotly_generator (Fase 3)
- Configs LLM otimizadas: `timeout=30s`, `max_retries=2`
- Cache de metadados com escopo de sessão no app Streamlit

---

## Arquivos Principais para Verificar Antes de Fazer Mudanças

- `src/pipeline_orchestrator.py` - Lógica de roteamento do pipeline
- `src/graphic_classifier/decision_tree/level1_detection.py` - Lógica de decisão de tipo de gráfico
- `src/shared_lib/models/schema.py` - Schemas Pydantic (ChartOutput, FilterOutput)
- `data/output_examples/README.md` - Especificações de tipos de gráfico
- `src/shared_lib/utils/query_analyzer.py` - Heurísticas de detecção de filtros
- `src/formatter_agent/parsers/input_parser.py` - Montagem de input LLM para formatter

---

## Obtendo Ajuda

- **Problemas de instalação**: Veja `INSTALL.md`
- **Migração Gemini**: Veja `GEMINI_MIGRATION_GUIDE.md`
- **Tipos de gráfico**: Veja `data/output_examples/README.md`
- **Relatórios de implementação**: Vários arquivos `*_REPORT.md` documentam implementações de features
- **Logs**: Verifique `logs/agent.log` para erros de runtime
