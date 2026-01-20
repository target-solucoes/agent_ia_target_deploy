import pandas as pd
import plotly.express as px

df = pd.read_parquet("DadosComercial_resumido_v02.parquet")

# Filtrar o cliente 23700
df_cliente = df[df["Cod_Cliente"] == 23700]

# Agrupar por produto e somar o valor vendido
df_agg = df_cliente.groupby("Des_Linha_Produto", as_index=False)["Qtd_Vendida"].sum()

# Ordenar do maior para o menor
df_agg = df_agg.sort_values("Qtd_Vendida", ascending=False)

# Pegar os 10 produtos principais
top10 = df_agg.head(10)

# Somar os demais como "OUTROS"
outros_valor = df_agg["Qtd_Vendida"][10:].sum()
df_final = pd.concat(
    [
        top10,
        pd.DataFrame({"Des_Linha_Produto": ["OUTROS"], "Qtd_Vendida": [outros_valor]}),
    ],
    ignore_index=True,
)

# Criar gráfico de pizza
fig = px.pie(
    df_final,
    names="Des_Linha_Produto",
    values="Qtd_Vendida",
    title="Distribuição de Produtos do Cliente 23700 (Top 10 + Outros)",
    hover_data=["Qtd_Vendida"],
)

df_final
