# Graphical Classifier Output Examples

Este diretorio contem exemplos completos e validados dos outputs JSON gerados pelo agente `graphical_classifier` para todos os 9 casos (8 tipos de graficos + 1 caso nao-grafico).

## Estrutura dos Arquivos

Cada arquivo contem:
- **Descricao**: Explicacao do tipo de grafico e quando usa-lo
- **Examples**: Multiplos exemplos com queries e outputs completos
- **Validation Notes**: Campos obrigatorios, opcionais e padroes comuns

## Lista de Arquivos

| Arquivo | Chart Type | Descrição Resumida |
|---------|-----------|--------------------|
| 01_bar_horizontal_examples.json | bar_horizontal | Rankings e comparações Top-N |
| 02_bar_vertical_examples.json | bar_vertical | Comparações diretas entre categorias |
| 04_bar_vertical_stacked_examples.json | bar_vertical_stacked | Composições empilhadas e proporções |
| 05_line_examples.json | line | Tendências temporais (série única) |
| 06_line_composed_examples.json | line_composed | Comparações temporais e múltiplas séries |
| 07_pie_examples.json | pie | Distribuições proporcionais |
| 08_histogram_examples.json | histogram | Distribuições de valores |
| 09_null_examples.json | null | Sem necessidade de visualização |

## Detalhamento de Casos de Uso por Tipo de Gráfico

### 1. Bar Horizontal (`bar_horizontal`)
**Quando usar:**
- Rankings e comparações Top-N (ex: "top 10 clientes", "5 melhores vendedores")
- Listas ordenadas por valor (maior para menor ou vice-versa)
- Comparações onde os rótulos são longos (nomes de produtos, clientes, etc.)

**Características:**
- **Dimensão:** 1 categórica (ex: Cliente, Produto, Vendedor)
- **Métrica:** 1 numérica agregada (ex: Valor_Vendido, Quantidade)
- **Ordenação:** Sempre por valor (desc para Top-N, asc para Bottom-N)
- **Visual:** Orientação horizontal, barras da esquerda para direita

**Keywords típicas:** "top", "ranking", "melhores", "maiores", "piores", "menores"

**Exemplo de query:** _"top 5 clientes por valor de vendas"_

---

### 2. Bar Vertical (`bar_vertical`)
**Quando usar:**
- Comparações diretas entre poucas categorias (2 a 7)
- Análises simples sem dimensão temporal
- Comparações lado a lado de valores absolutos

**Características:**
- **Dimensão:** 1 categórica (ex: Região, Categoria, Status)
- **Métrica:** 1 numérica agregada
- **Ordenação:** Opcional (por categoria ou por valor)
- **Visual:** Orientação vertical, barras de baixo para cima

**Keywords típicas:** "comparar", "vendas por", "quantidade de", "valores de"

**Exemplo de query:** _"vendas por região"_

---

### 3. Bar Vertical Stacked (`bar_vertical_stacked`)

**Quando usar:**
- Composições ou proporções **dentro de categorias**.
- Mostrar contribuição de subcategorias para um total.
- Comparar estruturas entre diferentes categorias ou entidades.
- Quando a pergunta envolve **mais de uma dimensão**, como "top 3 clientes dos 5 maiores estados".

**Características:**
- **Dimensões:** 2 ou mais categóricas – uma principal e outra(s) como subcategoria(s).
- **Métrica:** 1 numérica agregada (como `SUM`, `MAX` ou `COUNT`).
- **Visual:** Barras empilhadas verticalmente; cada segmento representa uma subcategoria.
- **Análise:** Permite ver tanto o total quanto a distribuição interna entre subcategorias.

**Keywords típicas:** "maiores N dos maiores X", "top N dos maiores X", "por categoria e subcategoria", "top N por categoria"

**Exemplo de query:** _"top 3 clientes dos 5 maiores estados"_  

---

### 4. Line (`line`)
**Quando usar:**
- Tendências temporais de uma única série
- Evolução ao longo do tempo sem comparação entre categorias
- Análise de sazonalidade ou padrões temporais

**Características:**
- **Dimensão:** 1 temporal (Mes, Ano, Data)
- **Métrica:** 1 numérica agregada
- **Ordenação:** Sempre ASC (cronológica)
- **Visual:** Linha contínua conectando pontos temporais

**Keywords típicas:** "evolução", "tendência", "ao longo do tempo", "histórico"

**Exemplo de query:** _"evolução de vendas mensais"_

---

### 5. Line Composed (`line_composed`)
**Quando usar:**
- **Comparações temporais entre categorias** (múltiplas séries), majoritaramente utilizado entre 2 periodos para representar variação de vendas entre categorias.
- **Análise de variação temporal** (crescimento, queda entre períodos)
- Identificar produtos/categorias com maior/menor crescimento
- Comparar desempenho de diferentes entidades ao longo do tempo

**Características:**
- **Dimensões:** 2 OBRIGATÓRIAS
  - **Primeira:** Temporal (Mes, Ano, Data) - eixo X
  - **Segunda:** Categórica (Produto, Cliente, Região) - séries
- **Métrica:** 1 numérica agregada
- **Ordenação:** 
  - `by: "temporal"` para ordem cronológica simples
  - `by: "variation"` para ranking por crescimento/variação
- **Visual:** Múltiplas linhas, cada uma representando uma categoria
- **Top-N:** Aplica-se às categorias (ex: Top 5 produtos com maior crescimento)

**Polaridade de Ordenação:**
- **Positiva** ("maior", "melhor", "top"): `order: desc` → Maiores crescimentos primeiro
- **Negativa** ("menor", "pior", "bottom"): `order: asc` → Menores crescimentos/quedas primeiro

**Keywords típicas:** 
- Variação: "maior aumento", "menor crescimento", "evolução de", "quais tiveram maior"
- Comparação: "comparar produtos", "tendência de cada", "evolução por categoria"

**Exemplos de queries:**
- _"quais produtos tiveram maior aumento de vendas de maio para junho de 2016?"_ → Top 5 com maior crescimento
- _"produtos com menor crescimento no último trimestre"_ → Bottom 5 (menores crescimentos)
- _"evolução de vendas dos estados SC, PR e RS por mês"_ → Comparação temporal de 3 estados

**IMPORTANTE:** Este é o chart type usado para análises de comparação temporal (anteriormente `bar_vertical_composed` para variações temporais foi migrado para `line_composed`).

---

### 6. Pie (`pie`)
**Quando usar:**
- Mostrar proporções e percentuais do total
- Distribuições com poucas categorias (ideal: 3 a 7)
- Enfatizar a participação relativa de cada parte

**Características:**
- **Dimensão:** 1 categórica
- **Métrica:** 1 numérica agregada
- **Visual:** Setores circulares proporcionais aos valores
- **Análise:** Foco em percentuais, não valores absolutos

**Keywords típicas:** "participação", "distribuição", "proporção", "percentual de"

**Exemplo de query:** _"participação de vendas por região"_

---

### 7. Histogram (`histogram`)
**Quando usar:**
- Distribuições de frequência de valores numéricos
- Análise de dispersão e concentração de dados
- Identificar padrões, outliers e agrupamentos

**Características:**
- **Dimensão:** 1 numérica (ex: Valor_Vendido, Idade, Preço)
- **Métrica:** Contagem (frequência)
- **Visual:** Barras representando faixas de valores (bins)
- **Parâmetro:** `bins` define o número de intervalos

**Keywords típicas:** "distribuição de valores", "frequência de", "faixas de"

**Exemplo de query:** _"distribuição de valores de pedidos"_

---

### 8. Null (`null`)
**Quando usar:**
- Queries que não requerem visualização gráfica
- Perguntas diretas, instruções, pedidos de dados tabulares
- Casos onde a resposta é melhor apresentada como tabela ou texto

**Características:**
- **chart_type:** `null`
- **Resposta:** Mensagem textual ou dados tabulares sem gráfico
- **Uso:** Fallback seguro quando nenhum gráfico é apropriado

**Keywords típicas:** "mostre os dados", "liste", "quais são", perguntas sem agregação visual

**Exemplo de query:** _"liste todos os clientes de São Paulo"_

---

## Uso

### Leitura de Exemplos

```python
import json

# Ler exemplos de bar_horizontal
with open('data/output_examples/01_bar_horizontal_examples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Acessar primeiro exemplo
example = data['examples'][0]
print(f"Query: {example['query']}")
print(f"Output: {json.dumps(example['output'], indent=2)}")
```

### Validacao de Outputs

```python
from src.shared_lib.models.schema import ChartOutput

# Validar output contra schema
output_dict = example['output']
validated_output = ChartOutput(**output_dict)
print(f"Valid: {validated_output.chart_type}")
```

## Padroes Comuns

### Campos Sempre Presentes

Todos os outputs (exceto null) devem ter:
- `chart_type`: Tipo de grafico (ou null)
- `metrics`: Lista de metricas (pode ser vazia para null)
- `dimensions`: Lista de dimensoes (pode ser vazia)
- `filters`: Dicionario de filtros (pode ser vazio {})
- `visual`: Configuracoes visuais

### Campos Opcionais

- `top_n`: Limite de registros (comum em rankings)
- `sort`: Configuracao de ordenacao
- `title`: Titulo do grafico
- `description`: Descricao detalhada
- `message`: Mensagem (usado em casos null)

## Referencias

- **Schema Oficial**: `src/shared_lib/models/schema.py:ChartOutput`
- **Specs por Chart Type**: `CHART_TYPE_SPECS.md`
- **Diagnostico Completo**: `GRAPHICAL_CLASSIFIER_DIAGNOSIS.md`
- **Contrato de Integracao**: `INTEGRATION_CONTRACT.md`

## Atualizações

- **v2.0** (2025-12-15): 
  - ✅ Removido `bar_vertical_composed` (migrado para `line_composed` para análises temporais)
  - ✅ Expandido `line_composed` com suporte a variação temporal e polaridade negativa
  - ✅ Adicionados casos de uso detalhados para cada chart type
  - ✅ Novos exemplos de comparações temporais com maior/menor crescimento
- **v1.0** (2025-11-10): Criação inicial com 9 tipos de gráficos e 28 exemplos totais
